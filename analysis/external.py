# %%
import argparse
import copy
import os
import random
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as t
from IPython.display import Image
from scipy.optimize import curve_fit
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

# %%
device = "cuda"

seed = 0
dset = "cifar3"
res = 1
conv = 2
depth = 8
cha = 64
ker = 3
pad = 1
btn = 0
lpl = 1
ep = 50
bs = 1_024
op = "sgd"
lr = 0.1
wd = 1e-2
ga = 0.1
sc = "cosine"

# %%
"""
## utils
"""


# %%
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def getbnp(model):
    l = []
    for m in model.children():
        if is_bn(m):
            l.append(m.weight)
            l.append(m.bias)
    return l


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
K = 30
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
    print(Dl - Dr)
    print(Dl)
    return U, Dl**0.5, V


def randomsvd(J):
    Us, Ss, Vs = [], [], []
    for j in range(len(J)):
        # i = len(J) - j - 1
        i = j
        tt = time.time()
        print("svding layer {}".format(i))
        U, S, V = randomsvd_each(J[i])
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
        print(time.time() - tt)
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
    print(output.shape)
    J = []
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


def plots(h, fh, title=None, device="cuda"):
    JUSV = plotsvals(h, fh, title, device, randomsvd, "1")
    alignment(h, fh, title, device, JUSV, randomsvd, "1")


def trajectories(model, track_loader, C, logdir, device):
    import matplotlib.colors as mcolors

    pcolors = sorted(list(mcolors.TABLEAU_COLORS.values()))
    tp = 10
    colours = []
    hpclass = [[] for _ in range(C)]
    ppclass = [[] for _ in range(C)]
    cpclass = [0 for _ in range(C)]
    for data, target in track_loader:
        if sum(cpclass) == -C:
            break
        data, target = data.to(device), target.to(device)
        _, h, _ = model(data)
        for c in range(C):
            matches = target == c
            smatches = matches.sum()
            if not smatches:
                continue
            if cpclass[c] < 0:
                continue
            if cpclass[c] > tp:
                cpclass[c] = -1
                continue
            cpclass[c] += smatches
            hpclass[c].append([hh[matches] for hh in h])
            ppclass[c].append(data[matches])
    hpclass = [[torch.cat(ll, dim=0)[:tp] for ll in list(zip(*l))] for l in hpclass]
    ppclass = [torch.cat(l, dim=0)[:tp] for l in ppclass]
    vecss = []
    novec = 1
    for c in range(len(hpclass)):
        points = []
        for l in range(len(hpclass[c])):
            hh = hpclass[c][l].reshape(tp, -1)
            sp = hh.shape
            if novec == 1:
                foundvec = False
                for vecs in vecss:
                    if vecs.shape[1] == sp[1]:
                        foundvec = True
                if not foundvec:
                    vecss.append(torch.randn(2, sp[1]).to(device))
            for vecs in vecss:
                if vecs.shape[1] == sp[1]:
                    z = vecs @ hh.T
                    break
            points.append(z)
        points = torch.stack(points).detach().cpu().numpy()
        novec = 0
        for l in range(tp):
            plt.plot(points[:, 0, l], points[:, 1, l], marker="o", color=pcolors[c], markersize=3)

    plt.savefig(logdir + "trajectories.png")
    plt.close()


# %%
# Modified from https://github.com/DIAGNijmegen/StreamingCNN/blob/master/Imagenette%20example.ipynb
class ImagenetteDataset(object):
    def __init__(self, dset, data_dir, imsize=224, validation=False, should_normalize=True):
        suff = {"nette": "nette2", "woof": "woof", "wang": "wang"}[dset[5:]]
        trainfolder = Path(data_dir + "image" + suff + "-320/train")
        validfolder = Path(data_dir + "image" + suff + "-320/val")
        self.folder = trainfolder if not validation else validfolder
        self.classes = sorted([pp.stem for pp in list(trainfolder.glob("*/"))])
        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + "/*.JPEG"))
            self.images.extend(cls_images)
        self.imsize = imsize
        self.validation = validation
        self.rrc = t.Compose([t.Resize(imsize), t.CenterCrop(imsize)])
        self.cc = t.Compose([t.Resize(imsize), t.CenterCrop(imsize)])
        self.should_normalize = should_normalize
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, index):
        image_fname = self.images[index]
        image = PIL.Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)
        if not self.validation:
            image = self.rrc(image)
        else:
            image = self.cc(image)
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1:
            image = image.expand(3, self.imsize, self.imsize)
        if self.should_normalize:
            image = self.normalize(image)
        return image, label

    def __len__(self):
        return len(self.images)


# Adapted from https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f
class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=None):
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        print(self.lengths)
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index):
        accum = np.add.accumulate(bin_sizes)
        bin_index = len(np.argwhere(accum <= absolute_index))
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        return bin_index, index_wrt_class


def get_class_i(x, y, i):
    y = np.array(y)
    pos_i = np.argwhere(y == i)
    pos_i = list(pos_i[:, 0])
    x_i = [x[j] for j in pos_i]
    return x_i


def get_subsets(trainset, testset, C, train_t=None, test_t=None):
    classDict = {
        "plane": 0,
        "car": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets

    RC = t.RandomCrop(32, padding=4)
    RS = t.Resize(32)
    RHF = t.RandomHorizontalFlip(p=0.5)
    RVF = t.RandomVerticalFlip()
    RAF = t.RandomAffine(20, shear=20, scale=(0.8, 1.2))
    CJ = t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    NRM = t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = t.ToTensor()
    TPIL = t.ToPILImage()

    train_t = t.Compose([TPIL, RC, RHF, RAF, CJ, TT, NRM])
    # train_t = t.Compose([TPIL, RS, TT, NRM])
    test_t = t.Compose([TPIL, RS, TT, NRM])

    s1 = DatasetMaker(
        [
            get_class_i(x_train, y_train, 1),
            get_class_i(x_train, y_train, 2),
            get_class_i(x_train, y_train, 3),
            get_class_i(x_train, y_train, 4),
            get_class_i(x_train, y_train, 5),
            get_class_i(x_train, y_train, 6),
            get_class_i(x_train, y_train, 7),
            get_class_i(x_train, y_train, 8),
            get_class_i(x_train, y_train, 9),
            get_class_i(x_train, y_train, 0),
        ][:C],
        train_t,
    )
    s2 = DatasetMaker(
        [
            get_class_i(x_test, y_test, 1),
            get_class_i(x_test, y_test, 2),
            get_class_i(x_test, y_test, 3),
            get_class_i(x_test, y_test, 4),
            get_class_i(x_test, y_test, 5),
            get_class_i(x_test, y_test, 6),
            get_class_i(x_test, y_test, 7),
            get_class_i(x_test, y_test, 8),
            get_class_i(x_test, y_test, 9),
            get_class_i(x_test, y_test, 0),
        ][:C],
        test_t,
    )

    return s1, s2


# %%
class BasicBlock(nn.Module):
    def __init__(self, res, conv, chanum, kersiz, padding, dsample=0):
        super().__init__()
        self.res = res
        outcha = chanum * 2 if dsample else chanum
        stride = 2 if dsample else 1
        self.dsample = (
            nn.Sequential(
                nn.Conv2d(chanum, outcha, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(outcha),
            )
            if dsample
            else nn.Identity()
        )
        if conv:
            self.w1 = nn.Conv2d(chanum, chanum, kersiz, stride=stride, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm2d(chanum)
            self.w2 = nn.Conv2d(chanum, outcha, kersiz, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm2d(outcha)
        else:
            self.w1 = nn.Linear(chanum, chanum, bias=False)
            self.bn1 = nn.BatchNorm1d(chanum)
            self.w2 = nn.Linear(chanum, outcha, bias=False)
            self.bn2 = nn.BatchNorm1d(outcha)

    def forward(self, x):
        fx = F.relu(self.bn1(self.w1(x)))
        fx = self.bn2(self.w2(fx))
        skip = self.dsample(x)
        x, fx = (
            (F.relu(skip + fx), fx * (skip + fx > 0)) if self.res else (F.relu(fx), fx * (fx > 0))
        )
        return x, fx


class ResNet(nn.Module):
    def __init__(self, res, conv, depth, inchan, chanum, kersiz, padding, cls, imsize, lpl):
        super(ResNet, self).__init__()
        self.res = res
        self.conv = conv
        pol = 2**lpl
        if conv == 0:
            self.w = nn.Linear(inchan, chanum, bias=False)
            self.bn = nn.BatchNorm1d(chanum)
            self.blocks = nn.ModuleList(
                [BasicBlock(res, conv, chanum, kersiz, padding) for _ in range(depth)]
            )
            self.clf = nn.Linear(chanum, cls, bias=True)
        elif conv == 1:
            if imsize == 32:
                self.w = nn.Conv2d(
                    inchan, chanum, kernel_size=2 * pol + 1, stride=pol, padding=pol, bias=False
                )
            elif imsize == 28:
                self.w = nn.Conv2d(
                    inchan, chanum, kernel_size=2 * pol + 1, stride=pol, padding=pol + 2, bias=False
                )
            else:
                raise NotImplementedError("bad input size")
            nimsize = 32 // pol
            pos = min(4, nimsize)
            self.bn = nn.BatchNorm2d(chanum)
            self.blocks = nn.ModuleList(
                [BasicBlock(res, conv, chanum, kersiz, padding) for _ in range(depth)]
            )
            self.avgpool = nn.AvgPool2d(pos)
            self.clf = nn.Linear(chanum * nimsize**2 // pos**2, cls, bias=True)
        elif conv == 2:
            if imsize == 32:
                self.w = nn.Conv2d(inchan, chanum, kernel_size=5, stride=2, padding=2, bias=False)
            elif imsize == 28:
                self.w = nn.Conv2d(inchan, chanum, kernel_size=5, stride=2, padding=4, bias=False)
            else:
                raise NotImplementedError("bad input size")
            nimsize = 32 // pol
            pos = min(4, nimsize)
            self.bn = nn.BatchNorm2d(chanum)
            blocks = []
            blocks.extend(
                [BasicBlock(res, conv, chanum, kersiz, padding) for _ in range(depth // 2)]
            )
            blocks.extend([BasicBlock(res, conv, chanum, kersiz, padding, dsample=1)])
            blocks.extend(
                [
                    BasicBlock(res, conv, chanum * 2, kersiz, padding)
                    for _ in range(depth - depth // 2 - 1)
                ]
            )
            self.blocks = nn.ModuleList(blocks)
            self.avgpool = nn.AvgPool2d(2)
            self.clf = nn.Linear(chanum * nimsize**2 // pos**2 * 2, cls, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x if self.conv else x.view(x.shape[0], -1)
        x = F.relu(self.bn(self.w(x)))
        h = [x]
        fh = [x]
        for i in range(len(self.blocks)):
            x, fx = self.blocks[i](x)
            h.append(x)
            fh.append(fx)
        if self.conv:
            x = self.avgpool(x).reshape(x.shape[0], -1)
        return self.clf(x), h, fh


# %%
"""
## run
"""

# %%
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if "mnist" in dset.lower():
    inchan = 1
    imsize = 28
    C = 10
elif "cifar" in dset.lower():
    inchan = 3
    imsize = 32
    if "100" in dset.lower():
        C = 100
    elif "10" in dset.lower():
        C = 10
    else:
        assert len(dset) == 6
        C = int(dset[-1])
elif "image" in dset.lower():
    inchan = 3
    imsize = 32
    if "wang" in dset.lower():
        C = 20
    else:
        C = 10

datadir = "."
if "nette" in dset.lower():
    train_bs = bs
    test_bs = 256
    trainset = ImagenetteDataset(dset, datadir, imsize)
    testset = ImagenetteDataset(dset, datadir, imsize, validation=True)
else:
    train_bs = bs
    test_bs = 256
    train_t = [
        torchvision.transforms.Resize(imsize),
        torchvision.transforms.ToTensor(),
    ]
    test_t = [
        torchvision.transforms.Resize(imsize),
        torchvision.transforms.ToTensor(),
    ]
    train_t = torchvision.transforms.Compose(train_t)
    test_t = torchvision.transforms.Compose(test_t)
    if dset.lower() == "mnist":
        trainset = torchvision.datasets.MNIST(datadir, train=True, download=True, transform=train_t)
        testset = torchvision.datasets.MNIST(datadir, train=False, download=True, transform=test_t)
    elif dset.lower() == "fashionmnist":
        trainset = torchvision.datasets.FashionMNIST(
            datadir, train=True, download=True, transform=train_t
        )
        testset = torchvision.datasets.FashionMNIST(
            datadir, train=False, download=True, transform=test_t
        )
    elif dset.lower() == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            datadir, train=True, download=True, transform=train_t
        )
        testset = torchvision.datasets.CIFAR100(
            datadir, train=False, download=True, transform=test_t
        )
    elif dset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            datadir, train=True, download=True, transform=train_t
        )
        testset = torchvision.datasets.CIFAR10(
            datadir, train=False, download=True, transform=test_t
        )
    elif dset.lower()[:-1] == "cifar":
        trainset = torchvision.datasets.CIFAR10(datadir, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(datadir, train=False, download=True)
        trainset, testset = get_subsets(trainset, testset, C)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=train_bs, shuffle=True, pin_memory=True, num_workers=4
)
track_loader = torch.utils.data.DataLoader(
    trainset, batch_size=test_bs, shuffle=False, pin_memory=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=test_bs, shuffle=False, pin_memory=True, num_workers=4
)

if not conv:
    inchan = inchan * imsize**2

model = ResNet(res, conv, depth, inchan, cha, ker, pad, C, imsize, lpl).to(device)
model.train()

if op.lower() == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
elif op.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
elif op.lower() == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

milestones = [int(ep * 0.25), int(ep * 0.5 // 1)]
if sc == "multistep":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=ga)
elif sc == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ep)

criterion = torch.nn.CrossEntropyLoss()


start_epoch = 0


class Logged:
    def __init__(self):
        pass


logged = Logged()
logged.losses = []
logged.testls = []
logged.testas = []
logged.filenames = []
dlogged = vars(logged)

print("finding checkpoint")
fname = "model.pt"
if fname in os.listdir("."):
    print("loading checkpoint")
    loaded = torch.load(fname)
    if "dlogged" in loaded:
        dlogged = loaded["dlogged"]
    if "epoch" in loaded:
        start_epoch = loaded["epoch"] + 1
    if "model" in loaded:
        model.load_state_dict(loaded["model"])
    if "optim" in loaded:
        optimizer.load_state_dict(loaded["optim"])
    scheduler.last_epoch = start_epoch

for key in dlogged.keys():
    exec(key + "=dlogged[key]")

for i in range(start_epoch, ep):
    tt = time.time()

    cor = 0
    all = 0
    meter = AverageMeter()
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(data)
        # print((logits.argmax(1) == target).sum().cpu().numpy())
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), len(target))
        cor += (logits.argmax(1) == target).sum().cpu().numpy()
        all += len(logits)
    scheduler.step()
    print(i, cor / all, meter.avg)
    logged.losses.append(meter.avg)
    if not (i) % 1:
        cor = 0
        all = 0
        meter = AverageMeter()
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _, _ = model(data)
            loss = criterion(logits, target)
            meter.update(loss.item(), len(target))
            cor += (logits.argmax(1) == target).sum().cpu().numpy()
            all += len(logits)
        print(i, cor / all, meter.avg)
        logged.testas.append(cor / all)
        logged.testls.append(meter.avg)

    print(i, "traineval", time.time() - tt)
    tt = time.time()

    if not (i + 1) % 50:
        torch.save(
            {
                "dlogged": dlogged,
                "epoch": i,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            },
            "model.pt",
        )


tt = time.time()

plt.figure()
plt.semilogy(logged.losses)
plt.savefig("train_loss.png")
plt.close()
plt.figure()
plt.semilogy(logged.testls)
plt.savefig("test_loss.png")
plt.close()
plt.figure()
plt.semilogy(logged.testas)
plt.savefig("test_acc.png")
plt.close()

print("curves", time.time() - tt)
tt = time.time()

model.eval()
sab(model)

x, _ = trainset[0]
x = x.to(device).unsqueeze(0)
x.requires_grad = True
_, h, fh = model(x)

plots(h, fh, "no_opt", device)
print("plot1", tt - time.time())
tt = time.time()

trajectories(model, track_loader, C, ".", device)
print("plot1", tt - time.time())
