import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
import numpy as np
from dask_ml.cluster import SpectralClustering as sc
from Tools.gpu_spectral import spectral_clustering_torch
from Tools.evaluation import cluster_accuracy

def post_proC(C, K, d, alpha):
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    if True:
        label = spectral_clustering_torch(similarity_matrix=L, n_clusters=K, use_gpu=True)
        return label, L
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='amg', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

#
# def thrC(C, ro):
#     if ro < 1:
#         N = C.shape[1]
#         Cp = np.zeros((N, N))
#         S = np.abs(np.sort(-np.abs(C), axis=0))
#         Ind = np.argsort(-np.abs(C), axis=0)
#         for i in range(N):
#             cL1 = np.sum(S[:, i]).astype(float)
#             stop = False
#             csum = 0
#             t = 0
#             while not stop:
#                 csum = csum + S[t, i]
#                 if csum > ro * cL1:
#                     stop = True
#                     Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
#                 t = t + 1
#     else:
#         Cp = C
#     return Cp

#
def thrC(C, ro):
    C = np.abs(C)
    # C_norm = normalize_columns_numpy(C)
    C[C<ro] = 0
    return C


def normalize_columns_numpy(C):
    C_min = C.min(axis=0, keepdims=True)
    C_max = C.max(axis=0, keepdims=True)
    epsilon = 1e-10
    C_norm = (C - C_min) / (C_max - C_min + epsilon)
    return C_norm






def performace(C, label, args):
    C = C.cpu().detach().numpy()
    Coef = thrC(C, args.threshold)
    colomn_max = np.max(Coef,axis=0)
    y_pred, C_best = post_proC(Coef, args.n_clusters, args.n_input, args.alpha)
    y_best, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_true=label, y_pre=y_pred, return_aligned=True)

    return y_best, acc, kappa, nmi, ari, pur, ca


def perform_spectral_clustering(similarity_matrix, n_clusters):
    """
    使用 Dask-ML 的 SpectralClustering 对给定的相似矩阵进行谱聚类。

    参数:
    similarity_matrix (array-like): 代表要聚类的数据的相似矩阵。
    n_clusters (int): 要形成的聚类数量。

    返回:
    array: 每个数据点的聚类标签。
    """
    from dask_ml.cluster import SpectralClustering

    # 创建 SpectralClustering 对象
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')

    # 拟合模型并预测聚类标签
    labels = clustering.fit_predict(similarity_matrix)

    return labels + 1
