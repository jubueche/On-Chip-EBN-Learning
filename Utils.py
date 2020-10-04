import numpy as np
import matplotlib.pyplot as plt

def plot_matrices(A,B,C=None, title_A=None, title_B=None, title_C=None):
    if(C is not None):
        base = 130
    else:
        base = 120
    plt.subplot(base +1)
    plt.title(title_A)
    im = plt.matshow(A, fignum=False)
    plt.xticks([], [])
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.subplot(base+2)
    plt.title(title_B)
    im = plt.matshow(B, fignum=False)
    plt.xticks([], [])
    plt.colorbar(im,fraction=0.046, pad=0.04)
    if(C is not None):
        plt.subplot(base+3)
        plt.title(title_C)
        im = plt.matshow(C, fignum=False)
        plt.xticks([], [])
        plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def round_to_nearest(x,l,r):
    if(round((x-l)/(r-l)) == 0):
        return l
    else:
        return r

def discretize(Omega, bin_edges):
    C = np.copy(Omega)
    diag_C = np.copy(np.diagonal(C))
    np.fill_diagonal(C,0)
    C_flat = C.ravel()
    indices = np.digitize(C_flat, bins = bin_edges, right = True)
    n = len(bin_edges)
    for idx,i in enumerate(indices):
        if(i >= n):
            i = n-1
        C_flat[idx] = round_to_nearest(C_flat[idx], bin_edges[i-1], bin_edges[i])
    C = C_flat.reshape(C.shape)
    np.fill_diagonal(C, diag_C)
    return C
