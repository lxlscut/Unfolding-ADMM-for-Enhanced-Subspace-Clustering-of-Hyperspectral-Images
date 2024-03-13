import torch
import torch.nn.functional as F


def preprocess_tensor_for_kl(tensor):
    # 将张量的所有值转换为绝对值
    tensor_abs = torch.abs(tensor)

    # 按列归一化张量，使每列的和为1
    column_sums = tensor_abs.sum(dim=0, keepdim=True)
    tensor_normalized = tensor_abs / column_sums

    return tensor_normalized


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
