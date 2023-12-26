import sklearn
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
LOCATION = '/home/jasimusmani/Documents/Subspace_Graph_Physics_Main/learning_to_simulate/datasets/ExcavationKPCA/'


def kpca(X2D):
    """This function is the implementation of the Kernel Principal component Analysis.
        Arguments:
        X2D is the 2 dimensional data
        return:
        This function returns the eigen values, eigenvectors and the mean of the data
        """

    observations = X2D.shape[0]
    dimensions = X2D.shape[1]

    m1 = np.mean(X2D, axis=0)
    m2 = np.mean(m1, axis=0)
    raw_data = KernelPCA(n_components=8,  kernel="rbf", gamma=1e-3)
    raw_data.fit_transform(X2D)

    evalues = raw_data.eigenvalues_
    evectors = raw_data.eigenvectors_

    idx = evalues.argsort()[::-1]  
    evalues = evalues[idx]
    evectors = evectors[:, idx]

    energy = []
    for i in range(1, dimensions):
        energy.append(np.sum(evalues[:i]) / np.sum(evalues))
    plt.plot(range(1, dimensions), energy, '.-')
    plt.savefig(LOCATION + 'energy.png', dpi=500)


    return evalues, evectors, m2