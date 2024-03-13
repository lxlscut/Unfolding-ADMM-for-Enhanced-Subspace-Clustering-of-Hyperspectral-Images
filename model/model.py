import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 设定一个种子值
from Tools.clustering import performace
from Tools.loss_function import l1_regularization_columnwise, compute_graph_regularization
from Tools.util import AdaptiveSoftshrink
from model.AE import Ae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class C_update(nn.Module):
    def __init__(self):
        super(C_update, self).__init__()

    def innitialize(self, W_initializer, B_initializer):
        self.W = nn.Parameter(W_initializer)
        self.B = nn.Parameter(B_initializer)

    def forward(self, input):
        X, u, Z, rho = input
        C = torch.matmul(self.W, X) - torch.matmul(self.B, u - rho * Z)
        return C


def keep_top_k_values_per_column(input_matrix, k=50):
    values, indices = torch.topk(torch.abs(input_matrix), k, dim=0)  # 沿列维度计算
    threshold = torch.abs(values[k - 1, :])  # 取出每列第k个值作为阈值
    binary_mask = (torch.abs(input_matrix) >= threshold).float()
    return input_matrix * binary_mask, binary_mask


class Z_update(nn.Module):
    def __init__(self):
        super(Z_update, self).__init__()

    def forward(self, input):
        """
        :param input:
        :return: z_s: the sparse value that must reserve
                z_m: the margin value that increase robustness and reduce instability
        """
        lamda, rho, C, u = input
        Z = C + u / rho
        Z = Z - torch.diag_embed(torch.diag(Z))
        return Z


class u_update(nn.Module):
    def __init__(self):
        super(u_update, self).__init__()

    def forward(self, input):
        u, rho, C, Z = input
        u = u + rho * (C - Z)
        return u

class admm_norm(nn.Module):
    def __init__(self):
        super(admm_norm, self).__init__()

    def forward(self, x):
        x = x + 1e-8  # 防止除零
        norm = torch.norm(x, p=1, dim=-1, keepdim=True)  # L1 范数
        return x / norm


class Admm(nn.Module):
    def __init__(self, label, g, args):
        super(Admm, self).__init__()
        self.AE = Ae(n_input=8, n_z=4).to(device=device)
        self.C_update = C_update()
        self.Z_update = Z_update()
        self.u_update = u_update()
        self.Norm = admm_norm()
        self.number_layer = args.number_layer
        self.lamda = args.lamda
        self.rho = nn.Parameter(torch.tensor([args.rho], device='cuda'))
        self.softshrink = AdaptiveSoftshrink(lambd=0.005)
        self.args = args
        self.g = g
        self.label = label


    def innitialize(self, W, B):
        self.C_update.innitialize(W_initializer=W, B_initializer=B)

    def pretrain_ae(self, train_loader, num_epochs=10, lr=0.001):
        """
        预训练自编码器
        """
        optimizer = torch.optim.Adam(self.AE.parameters(), lr=lr)
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (data, _, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                X_pr, _ = self.AE(data)
                loss = F.mse_loss(input=X_pr, target=data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def forward(self, X_p, u_0, z_0, epoch):
        u = u_0
        Z = z_0
        X_pr, H = self.AE(X_p)
        X = H.view(H.shape[0], -1).T
        X_p1 = X_p.view(H.shape[0], -1).T
        norms = torch.norm(X, p=2, dim=0)
        X = X / norms
        for i in range(self.number_layer):
            c_i = self.C_update([X, u, Z, self.rho])
            z_i = self.Z_update([self.lamda, self.rho, c_i, u])
            z_s = self.softshrink(z_i)
            u_i = self.u_update([u, self.rho, c_i, z_s])
            u = u_i
            Z = z_s
        if epoch < 0:
            y_i = torch.matmul(X_p1, c_i)
        else:
            y_i = torch.matmul(X, c_i)
        return c_i, y_i, z_s, u_i, X, X_pr

    def train(self, X_p, u_0, z_0, epochs):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)  # You can adjust the learning rate
        c_best = None
        z_best = None
        loss_min = 100000
        c_last = None
        c_difference = torch.tensor(0)

        for epoch in range(epochs):
            print("current epoch:", epoch)
            # Forward pass
            c_i, y_i, z_i, u_i, X, X_pr = self.forward(X_p, u_0, z_0, epoch)
            if c_last == None:
                c_last = c_i
            else:
                c_difference = torch.sum(torch.abs(c_i - c_last))
                c_last = c_i
            if (epoch % 10 == 0 or epoch==0) and self.label is not None:
                y_best, acc, kappa, nmi, ari, pur, ca = performace(c_i, self.label, self.args)
                # plot_clustered_heatmap(C=c_i_show, labels=y_best)
                print("epoch", epoch, "acc", acc)
                print("self.softshrink.lambd: ", self.softshrink.lambd, "self.rho: ", self.rho)
                print("c_difference", c_difference)
                # break
            # Calculate the loss
            loss = self.loss(X_p, X_pr, X, c_i, y_i, z_i, epoch)
            if loss < loss_min:
                loss_min = loss
                c_best = c_i
                z_best = z_i
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            # Print loss for every few epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            torch.cuda.empty_cache()
        print("loss_min", loss_min)
        return self, c_best, z_best

    def loss(self, X_p, X_pr, X, c_i, y_i, z_i, epoch):

        # AE RECONS_LOSS
        loss_ae = F.mse_loss(input=X_pr, target=X_p)
        loss_recon = torch.mean(torch.norm(X - y_i, p=2, dim=0))
        # c_loss
        loss_c1 = F.l1_loss(input=c_i, target=z_i,reduction='sum')/X.shape[0]
        loss_c2 = l1_regularization_columnwise(matrix=c_i, lambda_reg=1)
        loss_g = compute_graph_regularization(Z=c_i, A=self.g, lambda_reg=1)

        loss = loss_ae + self.args.bata*loss_recon + self.args.theta*loss_c2 + self.args.gama*loss_g

        print("recon_loss:", loss_recon, "loss_g", loss_g, "loss_c1:", loss_c1, "loss_c2:", loss_c2, "loss_ae:",
              loss_ae, "loss:", loss)

        return loss

    def innitialize_c(self, X, rho):
        m = 2 * torch.matmul(X.T, X) + rho * torch.eye(X.shape[1], device=device)
        m_1 = torch.inverse(m)
        W_c = torch.matmul(m_1, 2 * X.T)
        B_c = m_1
        return W_c, B_c


def train(model, dataset, u_0, z_0, epochs):
    # pre-train the auto-encoder of model
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model.pretrain_ae(train_loader=train_loader, num_epochs=100)
    # innitialize the w, b
    X_p = torch.tensor(dataset.train, device=device)
    _, H = model.AE(X_p)
    X = H.view(H.shape[0], -1).T
    X_n = F.normalize(input=X, p=2, dim=0)
    W_c, B_c = model.innitialize_c(X_n, model.rho)
    model.innitialize(W=W_c, B=B_c)
    # train the whole model
    model, C, Z = model.train(X_p=X_p, u_0=u_0, z_0=z_0, epochs=epochs)
    return model, C, Z

