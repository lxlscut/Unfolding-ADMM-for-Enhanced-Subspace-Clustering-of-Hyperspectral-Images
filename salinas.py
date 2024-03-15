import argparse
import numpy as np
import torch

from Tools.clustering import performace
from Tools.draw_prediction import draw_prediction_with_plt
from Tools.getdata import Load_my_Dataset
from Tools.util import find_nearest_neighbors
from model.model import Admm, train


seed = 42

# 设置NumPy随机生成器的种子
np.random.seed(seed)

# 设置PyTorch随机生成器的种子
torch.manual_seed(seed)

# 如果你的代码在CUDA上运行，还需要
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    # 为确保在CUDA上的确定性，可能需要设置以下选项：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    # 数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='ADMM for Subspace Clustering')

    parser.add_argument('--n_clusters', default=4, type=int, help='Number of clusters')
    parser.add_argument('--n_input', default=8, type=int, help='Input dimension')
    parser.add_argument('--threshold', default=0.005, type=float, help='Threshold')
    parser.add_argument('--alpha', default=18, type=int, help='Alpha for post-proC')
    parser.add_argument('--lamda', default=1, type=float, help='lamda')

    parser.add_argument('--number_layer', default=2, type=int, help='ADMM iteration')
    parser.add_argument('--lr', default=0.0000005, type=float, help='Learning rate')
    parser.add_argument('--rho', default=0.1, type=float, help='Initial rho')
    parser.add_argument('--bata', default=40, type=float, help='weight loss_recon')
    parser.add_argument('--theta', default=0.6, type=float, help='weight loss_c2')
    parser.add_argument('--gama', default=0.001, type=float, help='weight loss_g')



    args = parser.parse_args()
    dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/SalinasA/SalinasA_corrected.mat",
                              "/home/xianlli/dataset/HSI/SalinasA/SalinasA_gt.mat")
    # Get the location of different dataset
    data = dataset.train
    data_2 = data.reshape(data.shape[0], -1).T  # Shape: [392, 6104]
    z_ = find_nearest_neighbors(data_2, k=30)
    z1 = find_nearest_neighbors(data_2, k=10)
    g = np.logical_or(z1 == 1, z1.T == 1).astype(int)
    g = torch.tensor(g, device=device, dtype=torch.float32)
    z_0 = torch.tensor(z_, device=device, dtype=torch.float32) * 1e-1
    data = torch.tensor(data, device=device)
    label = dataset.y

    N_CLASS,CLASS_NUMER = np.unique(label,return_counts=True)
    args.n_clusters = len(N_CLASS)


    # 定义输入和输出的形状，以及网络的层数
    u_0 = torch.ones((data.shape[0], data.shape[0]), device=device) * 1e-3
    model = Admm(g=g,label=label,args=args)
    model.AE.initialize_weights()
    model, C, Z = train(model=model, dataset=dataset, u_0=u_0, z_0=z_0, epochs=500)

    C_show = C.cpu().detach().numpy()
    y_best, acc, kappa, nmi, ari, pur, ca = performace(C, label, args)

    # path_pred = "/home/xianlli/code/Journel_code/k-means/result/" + "india" + "_" + "unrolling_admm" + "_" "predict" + ".jpg"
    # path_true = "/home/xianlli/code/Journel_code/k-means/result/" + "india" + "_" + "unrolling_admm" + "_" + "label" + ".jpg"
    # draw_prediction_with_plt(location=dataset.index, pred=y_best, y_true=dataset.y, image_size=[85,70],
    #                          path_pred=path_pred, path_true=path_true)

    print("acc:", acc, "nmi:", nmi, "kappa:", kappa)
