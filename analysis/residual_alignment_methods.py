from typing import Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.functional as F
import torch.nn as nn
from torch.autograd import grad
from torcheval.metrics import MulticlassAccuracy

from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
)


@torch.no_grad()
def compare_model_autoencoder_acc(
    model, autoencoder, k, num_classes, mlp_inputs, ae_inputs, labels
):
    # Feed to MLP
    mlp_metric = MulticlassAccuracy(average=None, num_classes=num_classes)
    mlp_output = model(mlp_inputs)
    mlp_metric.update(mlp_output, labels.squeeze())

    # Feed to autoencoder
    koopman_metric = MulticlassAccuracy(average=None, num_classes=num_classes)
    pred_act = autoencoder(ae_inputs, k=k).predictions[-1]
    koopman_output = model.modules[-2:](pred_act)
    koopman_metric.update(koopman_output, labels.squeeze())

    return mlp_metric.compute(), koopman_metric.compute()


def trajectories(model, dataloader, num_classes, samples_per_class=1, device="cpu"):
    # Get tableau colors for class visualization
    class_colors = sorted(list(mcolors.TABLEAU_COLORS.values()))

    # Initialize data structures
    activations_by_class = [[] for _ in range(num_classes)]  # Activations per class
    inputs_by_class = [[] for _ in range(num_classes)]  # Data points per class
    samples_collected = [0 for _ in range(num_classes)]  # Counter for points per class

    # Collect activations for each class
    for data, target in dataloader:
        # Exit if we've collected enough samples from all classes
        if sum(samples_collected) == -num_classes:
            break

        # Move data to device and get model activations
        data, target = data.to(device), target.to(device)
        _ = model(data)
        activations = model.get_fwd_activations()
        layer_outputs = list(activations.values())  # Layer activations

        # Process each class
        for class_idx in range(num_classes):
            # Find samples of this class in the batch
            class_mask = target == class_idx
            num_samples = class_mask.sum()

            # Skip if no matches or class already completed
            if not num_samples:
                continue
            if samples_collected[class_idx] < 0:
                continue

            # Mark class as complete if we have enough samples
            if samples_collected[class_idx] > samples_per_class:
                samples_collected[class_idx] = -1
                continue

            # Update counter and collect activations
            samples_collected[class_idx] += num_samples
            activations_by_class[class_idx].append([layer[class_mask] for layer in layer_outputs])
            inputs_by_class[class_idx].append(data[class_mask])

    # Concatenate and limit samples per class
    activations_by_class = [
        [torch.cat(layer_data, dim=0)[:samples_per_class] for layer_data in list(zip(*class_data))]
        for class_data in activations_by_class
    ]
    inputs_by_class = [
        torch.cat(class_data, dim=0)[:samples_per_class] for class_data in inputs_by_class
    ]

    # Prepare for trajectory visualization
    projection_vectors = []
    need_projection_vectors = True

    # Project and plot trajectories for each class
    for class_idx in range(len(activations_by_class)):
        trajectory_points = []

        # Project each layer's activations to 2D
        for layer_idx in range(len(activations_by_class[class_idx])):
            layer_activations = activations_by_class[class_idx][layer_idx].reshape(
                samples_per_class, -1
            )
            feature_dim = layer_activations.shape[1]

            # Create projection vectors if needed
            if need_projection_vectors:
                vector_exists = False
                for vectors in projection_vectors:
                    if vectors.shape[1] == feature_dim:
                        vector_exists = True
                if not vector_exists:
                    projection_vectors.append(torch.randn(2, feature_dim).to(device))

            # Find matching projection vectors and project
            for vectors in projection_vectors:
                if vectors.shape[1] == feature_dim:
                    projected_points = vectors @ layer_activations.T
                    break
            trajectory_points.append(projected_points)

        # Convert points to numpy for plotting
        trajectory_points = torch.stack(trajectory_points).detach().cpu().numpy()
        need_projection_vectors = False

        # Plot trajectories for each sample
        for sample_idx in range(samples_per_class):
            plt.plot(
                trajectory_points[:, 0, sample_idx],
                trajectory_points[:, 1, sample_idx],
                marker="o",
                color=class_colors[class_idx],
                markersize=3,
            )

    plt.show()


# credit to Elad Hoffer
def remove_bn_params(bn_module):
    bn_module.running_mean.fill_(0)
    bn_module.running_var.fill_(1)
    bn_module.register_parameter("weight", None)
    bn_module.register_parameter("bias", None)
    bn_module.removed = True


def init_bn_params(bn_module):
    bn_module.running_mean.fill_(0)
    bn_module.running_var.fill_(1)
    if bn_module.affine:
        bn_module.weight.fill_(1)
        bn_module.bias.fill_(0)


def absorb_bn(module, bn_module, remove_bn=True, verbose=True, center=None):
    with torch.no_grad():
        if hasattr(bn_module, "removed"):
            if bn_module.removed:
                return

        w = module.weight
        if module.bias is None:
            if isinstance(module, nn.Linear):
                out_ch = module.out_features
            else:
                out_ch = module.out_channels

            zeros = torch.zeros(out_ch, dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter("bias", bias)
        b = module.bias

        if len(w.shape) == 2:
            w_shape = [w.size(0), 1]
        else:
            if "Transpose" in str(type(module)):
                w_shape = [1, w.size(1), 1, 1]
            else:
                w_shape = [w.size(0), 1, 1, 1]

        if center == "self":
            w.add(-w.mean(0, keepdim=True))
            w.add(-w.mean(1, keepdim=True))

        if hasattr(bn_module, "running_mean"):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, "running_var"):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w_shape))
            b.mul_(invstd)

        if hasattr(bn_module, "weight"):
            w.mul_(bn_module.weight.view(w_shape))
            b.mul_(bn_module.weight)
        if hasattr(bn_module, "bias"):
            b.add_(bn_module.bias)

        if center == "bias":
            w.add(-b.view(w_shape))

        if remove_bn:
            remove_bn_params(bn_module)
        else:
            init_bn_params(bn_module)

        if verbose:
            print("BN module %s was asborbed into layer %s" % (bn_module, module))


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)


def is_id(m):
    return isinstance(m, nn.Identity)


def sab(model, prev=None, remove_bn=True, verbose=True, center=None):
    with torch.no_grad():
        for m in model.children():
            if is_bn(m) and is_absorbing(prev):
                absorb_bn(prev, m, remove_bn=remove_bn, verbose=verbose, center=center)
            sab(m, remove_bn=remove_bn, verbose=verbose, center=center)
            prev = m


############################################################################################


def randomevd(mm, d, k, l, its=0, device="cuda"):
    """
    Approximates the top eigenvalues(vectors) of a matrix M.

    === Input parameters ==
    mm:
        function that takes in a matrix Q and computes MQ
    d:
        width of matrix M
    k:
        number of principal components to extract
    l:
        number of random vectors to project. Usually k + 3
    its:
        number of power iterations. Usually even 0 is good
    device:
        cpu or cuda

    === Output variables ===
    (Ds, Vs)
    Ds:
        Top eigenvalues of M, 1-D tensor of (k,)
    Vs:
        Top eigenvectors of M, 2-D tensor of (d, k)
    """

    if l < k:
        raise ValueError("l={} less than k={}".format(l, k))
    Q = torch.randn(d, l).to(device)
    Q = mm(Q)
    Q, _ = torch.qr(Q)
    for i in range(its):
        Q = mm(Q)
        Q, _ = torch.qr(Q)
    R = Q.T @ mm(Q)
    R = (R + R.T) / 2
    D, S = torch.linalg.eigh(R, UPLO="U")

    V = Q @ S
    D, V = D[-k:], V[:, -k:]
    return D.flip(-1), V.flip(-1)


# K = 10
# L = 5
# E = 5
# ITS = 1
# K = 30
K = 12
L = 20
E = 5
ITS = 20


def randomsvd_each(J):
    dl = J.shape[0]
    dr = J.shape[1]
    k = K + E
    l = K + E + L
    its = ITS

    def mml(Q):
        tmp = J.T @ Q
        return J @ tmp

    def mmr(Q):
        tmp = J @ Q
        return J.T @ tmp

    Dl, U = randomevd(mml, dl, k, l, its, "cpu")
    Dr, V = randomevd(mmr, dr, k, l, its, "cpu")

    return U, Dl**0.5, V


def randomsvd(J):
    Us, Ss, Vs = [], [], []
    for j in range(len(J)):
        # i = len(J) - j - 1
        i = j
        print("svding layer {}".format(i))
        U, S, V = randomsvd_each(J[i])
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
    return Us, Ss, Vs


SVD = randomsvd


def trajectory(h, fh, title, device):
    vecs = torch.randn(2, h[0].numel()).to(device)
    zs = []
    for i, x in enumerate(h):
        z = vecs @ (h[i].view(-1, 1))
        zs.append(z)
    z = torch.cat(zs, dim=1).detach().cpu().numpy()
    plt.plot(z[0], z[1])
    plt.savefig(title + "projected.png")
    plt.close()


def jacobian(output, input, device):
    J = []
    print(input.shape)
    for i in range(output.numel()):
        I = torch.zeros_like(output)
        ind = np.unravel_index(i, I.shape)
        I[ind] = 1
        j = grad(output, input, I, retain_graph=True)[0]
        I[ind] = 0
        J.append(j.flatten())
    return torch.stack(J, dim=0)


def plotsvals(h, fh, title=None, device="cuda", SVD=SVD, label="0"):
    import matplotlib.colors as mcolors

    pcolors = sorted(list(mcolors.TABLEAU_COLORS.values()))
    svr = 1
    if svr:
        J = [jacobian(fh[l], h[l - 1], device).cpu() for l in range(1, len(h))]
    else:
        J = [jacobian(h[l], h[0], device).cpu() for l in range(1, len(h))]
    U, S, V = SVD(J)
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in J]
    U, V = [u[..., : K + E] for u in U], [v[..., : K + E] for v in V]

    d = torch.arange(len(S)).float() + 1
    csfont = {"fontname": "Times New Roman"}
    fs = 40
    fig, axs = plt.subplots(1, 2, figsize=(30, 12))
    for i in range(15):
        if svr:
            if i == 0:
                axs[0].scatter(d, 1 / S[:, i])
            axs[1].scatter(d, S[:, i], color=pcolors[9 * int(i < 10)])
        else:
            axs[0].scatter(d, S[:, i])
    if svr:
        axs[0].tick_params(labelsize=40)
        axs[1].tick_params(labelsize=40)
        axs[0].set_xlabel("Depth", fontsize=fs, **csfont)
        axs[0].set_ylabel("1 / Singular Value", fontsize=fs, **csfont)
        axs[1].set_xlabel("Depth", fontsize=fs, **csfont)
        axs[1].set_ylabel("Singular Value", fontsize=fs, **csfont)
        # axs[0].legend(np.arange(K+E)+1, fontsize=40)
        axs[1].legend(np.arange(K + E) + 1, fontsize=17)
    else:
        axs[0].set_xlabel("Depth", fontsize=fs, **csfont)
        axs[0].set_ylabel("Singular Value", fontsize=fs, **csfont)
        axs[0].legend(np.arange(K + E) + 1, fontsize=15)

    def fit_line(x, y, ax, plot=False):
        xm = torch.mean(x)
        ym = torch.mean(y)
        x_ = x - xm
        a = torch.sum(x_ * y) / torch.sum(x_**2)
        b = ym - a * xm
        y_hat = a * x + b
        if plot:
            ax.plot(x, y_hat)
        rss = torch.sum((y - y_hat) ** 2)
        tss = torch.sum((y - ym) ** 2)
        rsq = 1 - rss / tss
        return rsq.item()

    for i in range(1):
        if svr:
            print("fit_line", fit_line(d[9:], 1 / S[9:, i], axs[0], True))
        else:
            print("fit_line", fit_line(d, S[:, i], axs[0], True))

    plt.savefig(title + "svals{}.png".format(label))
    plt.close()

    if svr:
        return J, U, S, V
    else:
        return None


def alignment(h, fh, title=None, device="cuda", JUSV=None, SVD=SVD, label="0"):
    if JUSV is None:
        J = [jacobian(fh[l], h[l - 1], device).cpu() for l in range(1, len(h))]
        U, S, V = SVD(J)
        S = torch.stack(S).cpu()
    else:
        J, U, S, V = JUSV
    J = [j.cpu() for j in J]
    U = [u.cpu() for u in U]
    V = [v.cpu() for v in V]
    V = V

    fig, axs = plt.subplots(len(S), len(S), figsize=(20, 20))
    for i in range(len(S)):
        for j in range(len(S)):
            ax = axs[i, j]
            ax.tick_params(
                left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            uj, ji, vj = U[j], J[i], V[j]
            if uj.shape[0] != ji.shape[0] or vj.shape[0] != ji.shape[1]:
                continue
            m = uj[:, :K].T @ ji @ vj[:, :K]
            a = ax.matshow(torch.abs(m).detach().cpu().numpy(), cmap="RdBu")
    plt.savefig(title + "UJV{}.png".format(label))
    plt.close()

    fig, axs = plt.subplots(len(S), len(S), figsize=(20, 20))
    for i in range(len(S)):
        for j in range(len(S)):
            ax = axs[i, j]
            ax.tick_params(
                left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            uj, ji, vj = U[j], J[i], V[j]
            if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
                continue
            m = vj[:, :K].T @ ji @ uj[:, :K]
            a = ax.matshow(torch.abs(m).detach().cpu().numpy(), cmap="RdBu")
    plt.savefig(title + "VJU{}.png".format(label))
    plt.close()
