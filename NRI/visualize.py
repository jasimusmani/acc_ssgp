#adapted from https://github.com/juexinwang/NRI-MD
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(
    'Visualize the distribution of learned edges between the PCA modes.')
parser.add_argument('--num-residues', type=int, default=16,
                    help='Number of residues of the PDB.')
parser.add_argument('--windowsize', type=int, default=218,
                    help='window size')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='threshold for plotting')
parser.add_argument('--dist-threshold', type=int, default=1,
                    help='threshold for shortest distance')
parser.add_argument('--filename', type=str, default='./NRI-MD-main/logs/out_probs_train.npy',
                    help='File name of the probs file.')
args = parser.parse_args()


def getEdgeResults(threshold=False):
    a = np.load(args.filename)
    print(a.shape)

    b = a[:, :, 1]
    c = a[:, :, 2]
    d = a[:, :, 3]
    # e = a[:, :, 4]
    # f = a[:, :, 5]
    # g = a[:, :, 6]

    # There are four types of edges, eliminate the first type as the non-edge
    probs = b + c + d  # +e+f+g
    residueR2 = args.num_residues * (args.num_residues - 1)
    probs = np.reshape(probs, (args.windowsize, residueR2))

    # Calculate the occurence of edges
    edges_train = probs / args.windowsize

    results = np.zeros((residueR2))
    for i in range(args.windowsize):
        results = results + edges_train[i, :]

    if threshold:
        # threshold, default 0.6
        index = results < (args.threshold)
        results[index] = 0

    # Calculate prob for figures
    edges_results = np.zeros((args.num_residues, args.num_residues))
    count = 0
    for i in range(args.num_residues):
        for j in range(args.num_residues):
            if not i == j:
                edges_results[i, j] = results[count]
                count += 1
            else:
                edges_results[i, j] = 0

    return edges_results

# Load distribution of learned edges
edges_results_visual = getEdgeResults(threshold=True)
# Step 1: Visualize results
ax = sns.heatmap(edges_results_visual, linewidth=0.5,
                 cmap="Blues", vmax=1.0, vmin=0.0)
plt.savefig('./NRI-MD-main/logs/probs.png', dpi=600)
# plt.show()
plt.close()