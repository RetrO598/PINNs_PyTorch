from collections import OrderedDict
from itertools import combinations, product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotting
import scipy.io
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

np.random.seed(1234)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        X = np.concatenate([x, y, t], 1)

        lb = X.min(0)
        ub = X.max(0)

        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.X = X

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.u = torch.tensor(u, requires_grad=True).float().to(device)
        self.v = torch.tensor(v, requires_grad=True).float().to(device)

        self.layers = layers

        self.lambda1 = torch.tensor([0.0], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([0.0], requires_grad=True).float().to(device)

        self.lambda1 = torch.nn.Parameter(self.lambda1)
        self.lambda2 = torch.nn.Parameter(self.lambda2)

        self.iter_list = []
        self.loss_list = []
        self.lambda1_list = []
        self.lambda2_list = []

        self.dnn = DNN(layers).to(device)

        self.dnn.register_parameter("lambda1", self.lambda1)
        self.dnn.register_parameter("lambda2", self.lambda2)

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=500000,
            max_eval=500000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        self.iter = 0

    def net_NS(self, x, y, t):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        psi_and_p = self.dnn(torch.cat([x, y, t], dim=1))
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = torch.autograd.grad(
            psi,
            y,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True,
        )[0]
        v = (
            -1.0
            * torch.autograd.grad(
                psi,
                x,
                grad_outputs=torch.ones_like(psi),
                retain_graph=True,
                create_graph=True,
            )[0]
        )
        u_t = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_y = torch.autograd.grad(
            u,
            y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u_y,
            y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        v_t = torch.autograd.grad(
            v,
            t,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_x = torch.autograd.grad(
            v,
            x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_y = torch.autograd.grad(
            v,
            y,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_xx = torch.autograd.grad(
            v_x,
            x,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_yy = torch.autograd.grad(
            v_y,
            y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        p_x = torch.autograd.grad(
            p,
            x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]
        p_y = torch.autograd.grad(
            p,
            y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True,
        )[0]

        f_u = u_t + lambda1 * (u * u_x + v * u_y) + p_x - lambda2 * (u_xx + u_yy)
        f_v = v_t + lambda1 * (u * v_x + v * v_y) + p_y - lambda2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        loss = (
            torch.mean((self.u - u_pred) ** 2)
            + torch.mean((self.v - v_pred) ** 2)
            + torch.mean(f_u_pred**2)
            + torch.mean(f_v_pred**2)
        )

        loss.backward()
        self.iter += 1
        self.iter_list.append(self.iter)
        self.loss_list.append(loss.item())
        self.lambda1_list.append(self.lambda1.item())
        self.lambda2_list.append(self.lambda2.item())
        if self.iter % 100 == 0:
            print(
                "Iter %d, Loss: %.5e, L1: %.5e, L2: %.5e"
                % (self.iter, loss.item(), self.lambda1.item(), self.lambda2.item())
            )
        return loss

    def train(self):
        self.dnn.train()

        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, x_star, y_star, t_star):
        x = torch.tensor(x_star, requires_grad=True).float().to(device)
        y = torch.tensor(y_star, requires_grad=True).float().to(device)
        t = torch.tensor(t_star, requires_grad=True).float().to(device)

        self.dnn.eval()
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        u = u_pred.detach().cpu().numpy()
        v = v_pred.detach().cpu().numpy()
        p = p_pred.detach().cpu().numpy()

        return u, v, p


def plot_solution(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method="cubic")

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap="jet")
    plt.colorbar()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":
    N_train = 5000

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    # Load Data
    data = scipy.io.loadmat(
        r"C:\Users\songx\OneDrive\CODES\Python\PINNs\NS\Data\cylinder_nektar_wake.mat"
    )

    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(t_star, (1, N)).T

    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]

    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]

    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train()

    # Test Data
    snap = np.array([100])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print("Error u: %e" % (error_u))
    print("Error v: %e" % (error_v))
    print("Error p: %e" % (error_p))

    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method="cubic")
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method="cubic")
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method="cubic")
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method="cubic")

    data_vort = scipy.io.loadmat(
        r"C:\Users\songx\OneDrive\CODES\Python\PINNs\NS\Data\cylinder_nektar_t0_vorticity.mat"
    )

    x_vort = data_vort["x"]
    y_vort = data_vort["y"]
    w_vort = data_vort["w"]
    modes = data_vort["modes"].item()
    nel = data_vort["nel"].item()

    xx_vort = np.reshape(x_vort, (modes + 1, modes + 1, nel), order="F")
    yy_vort = np.reshape(y_vort, (modes + 1, modes + 1, nel), order="F")
    ww_vort = np.reshape(w_vort, (modes + 1, modes + 1, nel), order="F")

    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])

    fig, ax = plotting.newfig(1.0, 1.2)
    ax.axis("off")

    ####### Row 0: Vorticity ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 2 / 4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])

    for i in range(0, nel):
        h = ax.pcolormesh(
            xx_vort[:, :, i],
            yy_vort[:, :, i],
            ww_vort[:, :, i],
            cmap="seismic",
            shading="gouraud",
            vmin=-3,
            vmax=3,
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], "k", linewidth=1)
    ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], "k", linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], "k", linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], "k", linewidth=1)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Vorticity", fontsize=10)

    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 2 / 4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0], projection="3d")
    ax.axis("off")

    r1 = [x_star.min(), x_star.max()]
    r2 = [data["t"].min(), data["t"].max()]
    r3 = [y_star.min(), y_star.max()]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if (
            np.sum(np.abs(s - e)) == r1[1] - r1[0]
            or np.sum(np.abs(s - e)) == r2[1] - r2[0]
            or np.sum(np.abs(s - e)) == r3[1] - r3[0]
        ):
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, t_train, y_train, s=0.1)
    ax.contourf(
        X, UU_star, Y, zdir="y", offset=t_star.mean(), cmap="rainbow", alpha=0.8
    )

    ax.text(x_star.mean(), data["t"].min() - 1, y_star.min() - 1, "$x$")
    ax.text(x_star.max() + 1, data["t"].mean(), y_star.min() - 1, "$t$")
    ax.text(x_star.min() - 1, data["t"].min() - 0.5, y_star.mean(), "$y$")
    ax.text(x_star.min() - 3, data["t"].mean(), y_star.max() + 1, "$u(t,x,y)$")
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    ########      v(t,x,y)     ###################
    ax = plt.subplot(gs1[:, 1], projection="3d")
    ax.axis("off")

    r1 = [x_star.min(), x_star.max()]
    r2 = [data["t"].min(), data["t"].max()]
    r3 = [y_star.min(), y_star.max()]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if (
            np.sum(np.abs(s - e)) == r1[1] - r1[0]
            or np.sum(np.abs(s - e)) == r2[1] - r2[0]
            or np.sum(np.abs(s - e)) == r3[1] - r3[0]
        ):
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, t_train, y_train, s=0.1)
    ax.contourf(
        X, VV_star, Y, zdir="y", offset=t_star.mean(), cmap="rainbow", alpha=0.8
    )

    ax.text(x_star.mean(), data["t"].min() - 1, y_star.min() - 1, "$x$")
    ax.text(x_star.max() + 1, data["t"].mean(), y_star.min() - 1, "$t$")
    ax.text(x_star.min() - 1, data["t"].min() - 0.5, y_star.mean(), "$y$")
    ax.text(x_star.min() - 3, data["t"].mean(), y_star.max() + 1, "$v(t,x,y)$")
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    plt.savefig("fig1", dpi=500)
    plt.show()
    # plotting.savefig("figures/NavierStokes_data")

    fig, ax = plotting.newfig(1.015, 0.8)
    ax.axis("off")

    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ###########
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(
        PP_star,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Predicted pressure", fontsize=10)

    ########     Exact p(t,x,y)     ###########
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(
        P_exact,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Exact pressure", fontsize=10)

    # ######## Row 3: Table #######################
    # gs3 = gridspec.GridSpec(1, 2)
    # gs3.update(top=1 - 1 / 2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    # ax = plt.subplot(gs3[:, :])
    # ax.axis("off")

    # s = r"$\begin{tabular}{|c|c|}"
    # s = s + r" \hline"
    # s = s + r" Correct PDE & $\begin{array}{c}"
    # s = s + r" u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\"
    # s = s + r" v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})"
    # s = s + r" \end{array}$ \\ "
    # s = s + r" \hline"
    # s = s + r" Identified PDE (clean data) & $\begin{array}{c}"
    # s = s + r" u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})" % (
    #     model.dnn.state_dict()["lambda1"].item(),
    #     model.dnn.state_dict()["lambda2"].item(),
    # )
    # s = s + r" \\"
    # s = s + r" v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})" % (
    #     model.dnn.state_dict()["lambda1"].item(),
    #     model.dnn.state_dict()["lambda2"].item(),
    # )
    # s = s + r" \end{array}$ \\ "
    # s = s + r" \hline"
    # s = s + r" \end{tabular}$"

    # ax.text(0.015, 0.0, s)
    plt.savefig("pressure", dpi=500)
    plt.show()

    iter_plot = np.array(model.iter_list)
    loss_plot = np.array(model.loss_list)
    lambda1_plot = np.array(model.lambda1_list)
    lambda2_plot = np.array(model.lambda2_list)

    plt.loglog(iter_plot, loss_plot, color="b", label="Loss")
    plt.legend()
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.savefig("loss", dpi=500)
    plt.show()

    plt.plot(iter_plot, lambda1_plot, color="r", label="$\lambda_1$")
    plt.legend()
    plt.xlabel("Iter")
    plt.ylabel("$\lambda_1$")
    plt.savefig("lambda1", dpi=500)
    plt.show()

    plt.plot(iter_plot, lambda2_plot, color="b", label="$\lambda_2$")
    plt.legend()
    plt.xlabel("Iter")
    plt.ylabel("$\lambda_2$")
    plt.savefig("lambda2", dpi=500)
    plt.show()
