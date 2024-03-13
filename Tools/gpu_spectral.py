import torch
import numpy as np
from sklearn.cluster import KMeans
from finch import FINCH



def spectral_clustering_torch(similarity_matrix, n_clusters, use_gpu=False):
    """
    使用 PyTorch 实现谱聚类

    参数:
    similarity_matrix (numpy array): 相似性矩阵
    n_clusters (int): 聚类数
    use_gpu (bool): 是否使用 GPU

    返回:
    numpy array: 聚类标签
    """

    # 将 NumPy 数组转换为 PyTorch 张量
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    similarity_matrix_torch = torch.from_numpy(similarity_matrix).float().to(device)

    # 计算拉普拉斯矩阵
    # L = D - A
    degree_matrix = torch.diag(torch.sum(similarity_matrix_torch, dim=1))
    laplacian_matrix = degree_matrix - similarity_matrix_torch

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)

    # 选取前 n_clusters+1 个特征向量
    n_components = n_clusters
    top_eigenvectors = eigenvectors[:, 1:n_components+1]

    # 对特征向量进行归一化处理
    # top_eigenvectors_normalized = top_eigenvectors / torch.norm(top_eigenvectors, dim=1, keepdim=True)

    # 转换为 NumPy 数组以使用 KMeans
    # top_eigenvectors_np = top_eigenvectors_normalized.cpu().numpy()

    # # 转换为 NumPy 数组以使用 KMeans
    top_eigenvectors_np = top_eigenvectors.cpu().numpy()

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(top_eigenvectors_np)
    labels = kmeans.labels_
    # labels, num_clust, req_c = FINCH(top_eigenvectors_np,req_clust=n_clusters)
    return labels+1

#
# def spectral_clustering_torch(similarity_matrix, n_clusters, use_gpu=False):
#     """
#     使用 PyTorch 实现谱聚类
#
#     参数:
#     similarity_matrix (numpy array): 相似性矩阵
#     n_clusters (int): 聚类数
#     use_gpu (bool): 是否使用 GPU
#
#     返回:
#     numpy array: 聚类标签
#     """
#
#     # 将 NumPy 数组转换为 PyTorch 张量
#     if use_gpu and torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     similarity_matrix_torch = torch.from_numpy(similarity_matrix).float().to(device)
#
#     # 计算拉普拉斯矩阵
#     degree_matrix = torch.diag(torch.sum(similarity_matrix_torch, dim=1))
#     laplacian_matrix = degree_matrix - similarity_matrix_torch
#
#     # 计算特征值和特征向量
#     eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
#
#     # 选取与最小非零特征值相对应的特征向量
#     sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues)
#     n_components = n_clusters
#     top_eigenvectors = eigenvectors[:, sorted_indices[1:n_components+1]]
#
#     # 对特征向量进行归一化处理
#     top_eigenvectors_normalized = top_eigenvectors / torch.norm(top_eigenvectors, dim=1, keepdim=True)
#
#     # 转换为 NumPy 数组以使用 KMeans
#     top_eigenvectors_np = top_eigenvectors_normalized.cpu().numpy()
#
#     # 使用 KMeans 进行聚类
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
#     kmeans.fit(top_eigenvectors_np)
#     labels = kmeans.labels_
#
#     return labels

# 使用示例
# similarity_matrix = <你的相似性矩阵>
# n_clusters = <希望的聚类数量>
# labels = spectral_clustering_torch(similarity_matrix, n_clusters, use_gpu=True)
