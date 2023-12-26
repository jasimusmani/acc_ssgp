'''
Example usage (from parent directory):
`python -m subspace_data.error_computation_gn`

'''

import os
import pickle
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA


def main(data_path, rollout_path):
  with open(rollout_path, "rb") as file:
    rollout_data = pickle.load(file)

  # Load PCA loading matrix and mean scalar
  pca_load = data_path
  with open(os.path.join(pca_load, 'metadata.json'), 'rt') as fp:
    metadata = json.loads(fp.read())
  MODE_NUMBER = metadata["mode_number"]
  with open(os.path.join(pca_load, 'pca_eigen_vectors.pkl'), 'rb') as f:
    W = pickle.load(f)
    W = W[:, :MODE_NUMBER]
  with open(os.path.join(pca_load, 'pca_mean_scalar.pkl'), 'rb') as f:
    MEAN = pickle.load(f)

  c = 0
  for ax_i, (label, rollout_field) in enumerate(
      [(r"$\bf{Ground\ truth}$" + "\nMaterial point method", "ground_truth_rollout"),
       (r"$\bf{Prediction}$" + "\nGraph network", "predicted_rollout")]):
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]], axis=0)

    # Convert 3D soil data to 2D data
    particles_rigid = metadata["particles_rigid"]
    particles_nonrigid_full = W.shape[0]
    particles_all_sub = trajectory.shape[1]
    frames = trajectory.shape[0]
    dimension = trajectory.shape[2]
    Xr2D = np.zeros((trajectory.shape[2]*frames, particles_all_sub-particles_rigid))
    for i in range(frames):
      for j in range(particles_rigid, particles_all_sub):
        for k in range(dimension):
          Xr2D[k+i*dimension][j-particles_rigid] = trajectory[i][j][k]
    #Visualiaztion of the particles (Inverse PCA)
    print(Xr2D.shape)
    print(W.shape)
    X2D = np.matmul(Xr2D[:,:8], np.transpose(W))
    # rec_ker = sklearn.metrics.pairwise.pairwise_kernels(Xr2D, metric='rbf',gamma = 0.04) 
    # inv_mat = np.load("inv_mat_test_1.npy")
    # X2D = np.matmul(rec_ker,inv_mat[:960,:])
    if c == 0:
      trajectory1 = X2D
    else:
      trajectory2 = X2D
    c += 1

  error = np.power(np.subtract(trajectory1, trajectory2), 2) # MSE
  mean_particle = np.mean(error, axis=0)
  return mean_particle


if __name__ == "__main__":
  data_path = 'learning_to_simulate/datasets/Excavation_PCA'
  rollout_path = 'learning_to_simulate/rollouts/Excavation_NRI_8modes_4/rollout_test_8.pkl'
  mean_particle_excav = main(data_path, rollout_path)

  # data_path = 'learning_to_simulate/datasets/Wheel_PCA_data'
  # rollout_path = 'learning_to_simulate/rollouts/wheel_non_zero_diagonal_noise_106/rollout_test_11.pkl'
  mean_particle_wheel = main(data_path, rollout_path)
  print(mean_particle_wheel.shape)
  final_avg_mean = np.mean(mean_particle_wheel)
  print(final_avg_mean)
  # # Fig 1
  plt.figure(figsize=(10, 5))
  ax = plt.gca()
  ax.set_xscale('log')
  ax.set_yscale('log')
  plt.hist([mean_particle_wheel], bins=1000, color=['black'])
  ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
  plt.xlabel("Position MSE", fontsize=15)
  plt.ylabel("#Particles", fontsize=15)
  plt.title('GN Error Distribution Wheel: NRI Reduced (1 MSSG) ', fontsize=15)
  ax.legend(['Wheel'], loc="upper right", fontsize=15)
  # plt.savefig("gn_error_distribution_wheel_no_svd_4.png", dpi=300)

  # Fig 1
  # plt.figure(figsize=(10, 5))
  # ax = plt.gca()
  # ax.set_xscale('log')
  # ax.set_yscale('log')
  # plt.hist([mean_particle_excav, mean_particle_wheel], bins=1000, color=['black', 'firebrick'])
  # ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
  # plt.xlabel("Position MSE", fontsize=15)
  # plt.ylabel("#Particles", fontsize=15)
  # plt.title('GN Error Distribution Wheel', fontsize=15)
  # ax.legend(['Wheel: PCG', 'Wheel: FCG'], loc="upper right", fontsize=15)
  # plt.savefig("gn_error_distribution_wheel_PCG_FCG.png", dpi=300)
