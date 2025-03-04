from typing import Optional, Union

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


K = 30
L = 20
E = 5
ITS = 50


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
    print(Dl - Dr)
    print(Dl)
    return U, Dl**0.5, V


def randomsvd(J):
    Us, Ss, Vs = [], [], []
    for j in range(len(J)):
        # i = len(J) - j - 1
        i = j
        print("svding layer {}".format(i))
        # U, S, V = randomsvd_each(J[i])
        U, S, V = torch.svd_lowrank(J[i], q=K, niter=ITS)
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
    return Us, Ss, Vs


SVD = randomsvd


def jacobian_torch(output, input, device):
    print(output.shape)
    output_numel = output.numel()

    # Pre-allocate list with correct size to avoid dynamic resizing
    J = [None] * output_numel

    for i in range(output_numel):
        # Create one-hot vector
        I = torch.zeros_like(output)
        # Convert flat index to multi-dimensional index using PyTorch
        ind = torch.tensor(np.unravel_index(i, I.shape))
        I[tuple(ind)] = 1.0

        # Compute gradient
        j = grad(output, input, I, retain_graph=True)[0]

        # Store gradient - no need to zero I[ind] since I is recreated each iteration
        J[i] = j.flatten()

    return torch.stack(J, dim=0)


def plotsvals(h, fh, title=None, device="cuda", SVD=SVD, label="0"):
    import matplotlib.colors as mcolors

    pcolors = sorted(list(mcolors.TABLEAU_COLORS.values()))
    svr = 1
    if svr:
        J = [jacobian_torch(fh[l], h[l - 1], device).cpu() for l in range(1, len(h))]
    else:
        J = [jacobian_torch(h[l], h[0], device).cpu() for l in range(1, len(h))]
    U, S, V = SVD(J)
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in J]
    U, V = [u[..., : K + E] for u in U], [v[..., : K + E] for v in V]
    print(S.shape)

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
        J = [jacobian_torch(fh[l], h[l - 1], device).cpu() for l in range(1, len(h))]
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


def is_bn(m):
    return isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d)


def is_linear(m):
    return isinstance(m, nn.Linear)


def is_absorbing(m):
    return isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)


def remove_bn_params(bn_module):
    """Remove parameters from a BatchNorm module by setting them to identity transform."""
    with torch.no_grad():
        bn_module.running_mean.fill_(0)
        bn_module.running_var.fill_(1)
        bn_module.register_parameter("weight", None)
        bn_module.register_parameter("bias", None)
        bn_module.eval()  # Set to eval mode to ensure it uses running stats
        setattr(bn_module, "removed", True)  # Mark as removed


def absorb_bn(module, bn_module, remove_bn=True, verbose=True):
    """Absorb a BatchNorm module into a preceding Linear/Conv layer."""
    with torch.no_grad():
        if hasattr(bn_module, "removed") and bn_module.removed:
            return

        # Get weights and biases
        w = module.weight
        if module.bias is None:
            # Create a bias if none exists
            if isinstance(module, nn.Linear):
                out_ch = module.out_features
            else:  # Conv layer
                out_ch = module.out_channels

            zeros = torch.zeros(out_ch, dtype=w.dtype, device=w.device)
            bias = nn.Parameter(zeros)
            module.register_parameter("bias", bias)
        b = module.bias

        # Reshape for correct broadcasting
        if len(w.shape) == 2:  # Linear layer
            w_shape = [w.size(0), 1]
        else:  # Conv layer
            if "Transpose" in str(type(module)):
                w_shape = [1, w.size(1), 1, 1]
            else:
                w_shape = [w.size(0), 1, 1, 1]

        # Absorb BatchNorm parameters
        if hasattr(bn_module, "running_mean"):
            b.add_(-bn_module.running_mean)
        if hasattr(bn_module, "running_var"):
            invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
            w.mul_(invstd.view(w_shape))
            b.mul_(invstd)

        if hasattr(bn_module, "weight") and bn_module.weight is not None:
            w.mul_(bn_module.weight.view(w_shape))
            b.mul_(bn_module.weight)

        if hasattr(bn_module, "bias") and bn_module.bias is not None:
            b.add_(bn_module.bias)

        # Remove or initialize BatchNorm parameters
        if remove_bn:
            remove_bn_params(bn_module)

        if verbose:
            print(f"BN module {bn_module} was absorbed into layer {module}")


def find_bn_and_preceding_layer(module):
    """
    Find BatchNorm layers and their preceding layers in a module.
    Returns a list of (prev_layer, bn_layer) tuples.
    """
    absorb_pairs = []
    prev_layer = None

    # Handle the case where LinearLayer has internal BatchNorm
    if hasattr(module, "bn") and module.bn is not None and is_absorbing(module.linear):
        absorb_pairs.append((module.linear, module.bn))

    # Recursively search for layers to absorb
    for child in module.children():
        # Check if this child has its own BatchNorm
        if (
            hasattr(child, "bn")
            and child.bn is not None
            and hasattr(child, "linear")
            and is_absorbing(child.linear)
        ):
            absorb_pairs.append((child.linear, child.bn))

        # Standard sequential checking
        if prev_layer is not None and is_bn(child) and is_absorbing(prev_layer):
            absorb_pairs.append((prev_layer, child))

        # Recurse into child's children
        child_pairs = find_bn_and_preceding_layer(child)
        absorb_pairs.extend(child_pairs)

        # Update previous layer (only if it's an absorbing type)
        if is_absorbing(child) or is_bn(child):
            prev_layer = child

    return absorb_pairs


def sequential_absorb_batchnorm(model, remove_bn=True, verbose=True):
    """
    Find and absorb all BatchNorm layers into their preceding layers across the model.

    Args:
        model: The model to process
        remove_bn: Whether to remove the BatchNorm parameters after absorption
        verbose: Whether to print information about absorbed layers

    Returns:
        The modified model with BatchNorm layers absorbed
    """
    # Find all BatchNorm and preceding layer pairs to absorb
    absorb_pairs = find_bn_and_preceding_layer(model)

    # Absorb each pair
    for module, bn_module in absorb_pairs:
        absorb_bn(module, bn_module, remove_bn=remove_bn, verbose=verbose)

    if verbose:
        print(f"Absorbed {len(absorb_pairs)} BatchNorm layers")

    return model
