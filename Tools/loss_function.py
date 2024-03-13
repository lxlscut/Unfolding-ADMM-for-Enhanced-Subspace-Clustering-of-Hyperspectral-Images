import torch
import torch.nn.functional as F


def preprocess_tensor_for_kl(tensor):
    # 将张量的所有值转换为绝对值
    tensor_abs = torch.abs(tensor)

    # 按列归一化张量，使每列的和为1
    column_sums = tensor_abs.sum(dim=0, keepdim=True)
    tensor_normalized = tensor_abs / column_sums

    return tensor_normalized

def kl_divergence(input, target):
    # 预处理张量
    p_tensor = preprocess_tensor_for_kl(input)
    q_tensor = preprocess_tensor_for_kl(target)

    # 计算KL散度
    kl_div = F.kl_div(p_tensor.log(), q_tensor, reduction='sum')

    return kl_div


def cosine_similarity_loss_columnwise(x1, x2):
    """
    Computes the cosine similarity loss between two tensors column-wise.

    Args:
    x1 (Tensor): A tensor.
    x2 (Tensor): Another tensor of the same size as x1.

    Returns:
    Tensor: Loss value.
    """
    # Normalize x1 and x2 along the first dimension (column-wise)
    x1_normalized = F.normalize(x1, p=2, dim=0)
    x2_normalized = F.normalize(x2, p=2, dim=0)

    # Compute cosine similarity column-wise
    cos_sim = torch.sum(x1_normalized * x2_normalized, dim=0)
    cos_sim_show = cos_sim.detach().cpu().numpy()
    # Calculate loss as 1 - cosine similarity
    loss = 1 - cos_sim

    return loss.mean()

def l1_regularization_columnwise(matrix, lambda_reg):
    """
    Apply L1 regularization column-wise to a 2D matrix.

    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    return lambda_reg * torch.mean(torch.norm(matrix, p=1, dim=0))


import torch

import torch

def l1_regularization_columnwise_masked(matrix, z, lambda_reg):
    """
    Apply L1 regularization to elements of a 2D matrix ('matrix') where corresponding elements in another
    2D matrix ('z') are zero.

    Args:
    matrix (Tensor): The input 2D tensor (matrix) for which regularization is applied.
    z (Tensor): A 2D tensor of the same shape as 'matrix'. Regularization is applied to elements of 'matrix'
                where corresponding elements in 'z' are zero.
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated for the specified elements.
    """
    assert matrix.shape == z.shape, "The shape of 'matrix' and 'z' must be the same."

    mask = z == 0  # Create a mask where elements of 'z' are zero
    selected_elements = matrix[mask]  # Select elements from 'matrix' using the mask

    selected_elements_show = selected_elements.detach().cpu().numpy()

    # Calculate L1 regularization loss for the selected elements
    return lambda_reg * torch.sum(torch.abs(selected_elements))/matrix.shape[1]



def l2_regularization_columnwise(matrix, lambda_reg):
    """
    Apply L1 regularization column-wise to a 2D matrix.
    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    return lambda_reg * torch.sum(torch.norm(matrix, p=2, dim=0))/matrix.shape[1]


import torch


def custom_loss(V, MASK):
    """
    自定义损失函数，旨在增大MASK矩阵中特定位置值所占的比例。

    参数:
    V - 值矩阵 (tensor)
    MASK - 掩码矩阵，有相同尺寸的二值矩阵 (tensor)

    返回:
    损失值，该值在所需比例增大时减小
    """
    # 确保V和MASK是tensor且具有相同的维度
    assert V.shape == MASK.shape, "V and MASK must have the same shape"

    # 计算sum(V[MASK==1])和sum(V)
    masked_sum = torch.sum(V * MASK)
    total_sum = torch.sum(V)

    # 计算比例
    ratio = masked_sum / total_sum

    # 由于我们想最大化这个比例，我们可以最小化其负值或倒数作为损失
    # 这里我们使用倒数，因为它对小的比例更敏感
    loss = 1-ratio  # 加上一个小的常数以避免除以零

    return loss


def error_function(C, N, lambda_param):
    """
    计算基于自表示矩阵C和最近邻矩阵N的误差函数。

    参数:
    X -- 数据矩阵 (tensor)
    C -- 自表示矩阵 (tensor)
    N -- 最近邻矩阵 (tensor)
    lambda_param -- 正则化参数 (float)

    返回:
    总误差 (tensor)
    """
    C_norm = F.normalize(input=C,p=2,dim=0)
    # 最近邻正则化
    C_diff = torch.matmul(C_norm.T,C_norm)  # 计算C中每对点的差异
    C_diff.fill_diagonal_(0)
    similarity = N*C_diff

    # loss = 1 - torch.sum(torch.abs(similarity))/(torch.sum(torch.abs(C_diff))+1e-8)

    similarity_mean = torch.sum(similarity)/torch.sum(N)
    loss = 1-similarity_mean

    return loss


def compute_graph_regularization(Z, A, lambda_reg):
    """
    Compute the graph regularization term using PyTorch.

    Parameters:
    Z (torch.Tensor): The self-representation matrix or feature matrix (size: N x N).
    A (torch.Tensor): The adjacency matrix of the graph (size: N x N).
    lambda_reg (float): The regularization parameter.

    Returns:
    torch.Tensor: The value of the graph regularization term.
    """
    mask = torch.ones(Z.shape[0], Z.shape[0], dtype=torch.bool,device='cuda')
    mask.fill_diagonal_(0)
    N = A.size(0)
    D = torch.diag(A.sum(1))  # Degree matrix
    L = D - A  # Laplacian matrix
    # L_mod = L - torch.diag(torch.diag(L))  # Modified Laplacian with zero diagonal

    reg_term = lambda_reg * torch.trace(torch.matmul(torch.matmul(Z.t(), L), Z))
    return reg_term