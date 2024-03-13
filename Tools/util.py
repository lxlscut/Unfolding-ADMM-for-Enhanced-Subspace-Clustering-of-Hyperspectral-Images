import faiss
import numpy as np
import torch
import torch.nn as nn

def find_nearest_neighbors(X, k=20):
    # Number of data points (columns in X)
    num_points = X.shape[1]

    # Convert the data to float32 as required by faiss
    X = X.astype(np.float32)

    # Create a faiss index (using L2 distance by default)
    index = faiss.IndexFlatL2(X.shape[0])

    # Add data points to the index
    index.add(X.T)  # Transpose because faiss expects each data point as a row

    # Search the index for the k nearest neighbors of each point
    _, I = index.search(X.T, k + 1)  # +1 because the first nearest neighbor is the point itself

    # Initialize the relationship matrix with zeros
    relationship_matrix = np.zeros((num_points, num_points))

    # Fill in the relationship matrix
    for i in range(num_points):
        for j in range(1, k + 1):  # Skip the first one (itself)
            nearest_neighbor = I[i][j]
            relationship_matrix[i, nearest_neighbor] = 1

    return relationship_matrix

# Example usage
X = np.random.rand(5, 10)  # 5x10 data matrix
relationship_matrix = find_nearest_neighbors(X)
print(relationship_matrix)


class AdaptiveSoftshrink(nn.Module):
    def __init__(self, lambd=0.5):
        super(AdaptiveSoftshrink, self).__init__()
        # 将lambda定义为一个可学习的参数
        self.lambd = nn.Parameter(torch.tensor([lambd],device='cuda'))
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算绝对值减去lambda后的结果，并应用ReLU
        shrunk = self.relu(torch.abs(x) - self.lambd)
        # 恢复原始值的符号
        z_s = shrunk * torch.sign(x)

        z_s_show = z_s.detach().cpu().numpy()

        return z_s




class ModifiedSoftshrink(nn.Module):
    def __init__(self, lambd=0.5):
        super(ModifiedSoftshrink, self).__init__()
        self.lambd = lambd
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算绝对值减去lambda后的结果，并应用ReLU
        shrunk = self.relu(torch.abs(x) - self.lambd)
        # 恢复原始值的符号
        return shrunk * torch.sign(x)



def efficient_initialize_self_representation_matrix(labels):
    """
    Efficiently initialize the self-representation matrix based on labels.

    :param labels: An array of labels for each data point.
    :return: Initialized self-representation matrix.
    """
    n = len(labels)
    # Create an NxN self-representation matrix initialized to zeros
    C = np.zeros((n, n))

    # Efficiently set elements to 1 where labels match (excluding diagonal)
    for label in set(labels):
        indices = np.where(labels == label)[0]
        C[np.ix_(indices, indices)] = 1

    # Reset diagonal to zero to avoid self-representation
    np.fill_diagonal(C, 0)

    return C


def cosine_similarity_normalized(X):
    """
    计算余弦相似度矩阵，并将对角线元素设置为0。

    参数:
    X -- 数据矩阵，每一行是一个数据点。

    返回:
    余弦相似度矩阵，对角线元素为0。
    """
    # 归一化每个向量到单位长度
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 计算余弦相似度
    S_cosine = np.dot(X_normalized, X_normalized.T)

    # 将余弦相似度归一化到0-1区间
    S_normalized = (S_cosine + 1) / 2  # 由于余弦值范围是[-1, 1]，通过这个变换将其映射到[0, 1]

    # 将对角线元素设置为0
    np.fill_diagonal(S_normalized, 0)

    return S_normalized



def scale_matrix_columns_to_max_length(data):
    """
    将矩阵的每列缩放到最大列长度。

    参数:
    data (torch.Tensor): 需要缩放列的矩阵。

    返回:
    torch.Tensor: 列缩放后的矩阵。
    """
    # 计算每列的长度（L2 范数）
    col_lengths = torch.norm(data, p=2, dim=0)
    # print(torch.isnan(col_lengths).any())

    # 找出最大长度
    max_length = col_lengths.max()
    # print(torch.isnan(max_length).any())
    # 缩放每列到最大长度
    scale = (max_length / col_lengths)
    # print(torch.isnan(scale).any())
    scaled_data = data * scale
    # data_show = data.detach().cpu().numpy()
    # scale_show = scale.detach().cpu().numpy()
    # print(torch.isnan(data).any())
    #
    # print(torch.isnan(scaled_data).any())
    # print(torch.where(torch.isnan(scaled_data)))
    return scaled_data


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_clustered_heatmap(C, labels, title='Block Structure of Self-Representation Matrix'):
    """
    根据表示矩阵和聚类标签绘制块状结构的热图。

    参数:
    C -- 表示矩阵 (numpy.ndarray)
    labels -- 聚类标签 (numpy.ndarray)
    title -- 热图标题 (str)
    """
    # 根据聚类标签对表示矩阵的行和列进行排序
    sorted_indices = np.argsort(labels)
    C_sorted = C[sorted_indices, :][:, sorted_indices]

    # 计算不同类别之间的边界
    unique_labels = np.unique(labels)
    boundaries = [np.where(labels[sorted_indices] == label)[0][-1] for label in unique_labels[:-1]]

    # 绘制热图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(C_sorted, cmap='viridis', xticklabels=False, yticklabels=False, cbar=True)

    # 在类别边界处添加分界线
    for boundary in boundaries:
        plt.axhline(boundary + 1, color='white', lw=2)
        plt.axvline(boundary + 1, color='white', lw=2)

    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Data Points')
    plt.show()


def loss_graph(c,g):
    pass