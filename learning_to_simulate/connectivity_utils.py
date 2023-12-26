# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tools to compute the connectivity of the graph."""

import functools
from importlib_metadata import re

import numpy as np
from sklearn import neighbors
import tensorflow.compat.v1 as tf


# added
def _compute_connectivity(positions, radius, add_self_edges):
  """Get the indices of connected edges with radius connectivity.
  Args:
    positions: Positions of nodes in the graph. Shape:
      [num_nodes_in_graph, num_dims].
    radius: Radius of connectivity.
    add_self_edges: Whether to include self edges or not.
  Returns:
    senders indices [num_edges_in_graph]
    receiver indices [num_edges_in_graph]
  """

  # tree = neighbors.KDTree(positions)
  # receivers_list = tree.query_radius(positions, r=radius)
  # num_nodes = len(positions)
  # senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  # receivers = np.concatenate(receivers_list, axis=0)

  # if not add_self_edges:
  #   # Remove self edges.
  #   mask = senders != receivers
  #   senders = senders[mask]
  #   receivers = receivers[mask]
  # np.save("sender", senders)
  # np.save("receiver.npy", receivers)
  # return senders, receivers
#added amin connection

  # num_nodes = len(positions)
  # senders, receivers = [], []
  # for i in range(num_nodes):
  #   for j in range(num_nodes):
  #     senders = np.append(senders, i)
  #     receivers = np.append(receivers, j)

#added: FUNCTION TO GENERATE PARTIAL GRAPH FROM NRI: FULLY CONNECTED RIGID BODIES, NRI CONNECTED MODES

  # num_nodes = len(positions)
  # # print(num_nodes)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #   for j in range(num_nodes):
  #     if num_nodes==23:
  #       if i>=18 and j>=18:
  #         continue 
  #     if num_nodes==28:
  #       if i>=23 and j>=23:
  #         continue 
  #     senders = np.append(senders, i)
  #     # if num_nodes==23:
  #     #   receivers = np.load("recievers_main.npy")
  #     # if num_nodes==28:
  #     #   receivers = np.load("recievers_main_20.npy")
  #     receivers = np.append(receivers, j)
  
  #added:for NRI FULL PARTIAL GRAPH
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #         if i==j:
  #             continue
  #         if 0<=i<(num_nodes-8) and 0<=j<(num_nodes-8):
  #             continue
  #         if i>= (num_nodes-5) and 0<=j<(num_nodes-8):
  #             continue
  #         if 0<=i<(num_nodes-8) and j>=(num_nodes-5):
  #             continue
  #         if i>= (num_nodes-5) and j>=(num_nodes-5):
  #             continue
  #         senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #         receivers = np.append(receivers, j)

# #added:for NRI FULL PARTIAL GRAPH with 4 modes and three rigid body connections
#   num_nodes = len(positions)
#   senders, receivers = [], []
#   senders=  []
#   for i in range(num_nodes):
#       for j in range(num_nodes):
#           if i==j:
#               continue
#           if 0<=i<(num_nodes-11) and 0<=j<(num_nodes-11):
#               continue
#           if i>= (num_nodes-4) and 0<=j<(num_nodes-11):
#               continue
#           if 0<=i<(num_nodes-11) and j>=(num_nodes-4):
#               continue
#           if i>= (num_nodes-4) and j>=(num_nodes-4):
#               continue
#           senders = np.append(senders, i)
#         # if num_nodes==23:
#         #   receivers = np.load("recievers_main.npy")
#         # if num_nodes==28:
#         #   receivers = np.load("recievers_main_20.npy")
#           receivers = np.append(receivers, j)
#added:for NRI FULL PARTIAL GRAPH with 4 modes only
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #         if i==j:
  #             continue
  #         if 0<=i<(num_nodes-8) and 0<=j<(num_nodes-8):
  #             continue
  #         if i>= (num_nodes-4) and 0<=j<(num_nodes-8):
  #             continue
  #         if 0<=i<(num_nodes-8) and j>=(num_nodes-4):
  #             continue
  #         if i>= (num_nodes-4) and j>=(num_nodes-4):
  #             continue
  #         senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #         receivers = np.append(receivers, j)
#added:for NRI FULL PARTIAL GRAPH with 4 modes and 2 rigid bodies above
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #         if i==j:
  #             continue
  #         if 0<=i<(num_nodes-10) and 0<=j<(num_nodes-10):
  #             continue
  #         if i>= (num_nodes-5) and 0<=j<(num_nodes-10):
  #             continue
  #         if 0<=i<(num_nodes-10) and j>=(num_nodes-5):
  #             continue
  #         if i>= (num_nodes-5) and j>=(num_nodes-5):
  #             continue
  #         senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #         receivers = np.append(receivers, j)
  
  # #added:for NRI FULL PARTIAL GRAPH with NRI from 1 sample
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #         if i==j:
  #             continue
  #         if 0<=i<(num_nodes-8) and 0<=j<(num_nodes-8):
  #             continue
  #         # if i>= (num_nodes-4) and 0<=j<(num_nodes-8):
  #         #     continue
  #         if 0<=i<(num_nodes-8) and j>=(num_nodes-4):
  #             continue
  #         if i>= (num_nodes-4) and j>=(num_nodes-4):
  #             continue
  #         senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #         receivers = np.append(receivers, j)



  # #added:for NRI assumption 
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # senders=  []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #         if i==j:
  #             continue
  #         if 0<=i<(num_nodes-8) and 0<=j<(num_nodes-8):
  #             continue
  #         if 0<=i<(num_nodes) and 0<=j<(num_nodes-8):
  #             continue
  #         senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #         receivers = np.append(receivers, j)
  #         # np.save('assump_rec', receivers)
  #         # np.save('assump_send', senders)



# # #added:for NRI graph when NRI output with rigid bodies : NRI_8modes_4 final one
  # num_nodes = len(positions)
  # senders, receivers = [], []
  # for i in range(num_nodes):
  #     for j in range(num_nodes):
  #       if i==j:
  #           continue
  #       if 0<=i<(num_nodes-8) and 0<=j<(num_nodes-8):
  #           continue
  #       if 0<=i<(num_nodes) and 0<=j<(num_nodes-8):
  #           continue
  #       if i>= (num_nodes-4) and j>=(num_nodes-8):
  #             continue
  #       senders = np.append(senders, i)
  #       # if num_nodes==23:
  #       #   receivers = np.load("recievers_main.npy")
  #       # if num_nodes==28:
  #       #   receivers = np.load("recievers_main_20.npy")
  #       receivers = np.append(receivers, j)
# #           # np.save('assump_rec', receivers)
# #           # np.save('assump_send', senders)


  #wheel simulated with the partial graph

  # al = np.load('/home/jasimusmani/Documents/Subspace_Graph_Physics_Main/learning_to_simulate/corrected_diagonal_fb_new.npy')
  al = np.load('./learning_to_simulate/corrected_diagonal_fb_new.npy')
  senders = list()
  receivers = list()
  c = 0
  s = []
  for p in range(24):
      for q in range(24):
          if al[p][q]==1:
              c=c+1
              receivers.append(q)
      s = [p]*c
      c= 0
      senders.extend(s)
  senders = np.stack(senders)
  receivers = np.stack(receivers)
  # print(senders.shape)
  # np.save('wheel_sender',senders)
  # np.save('wheel_reciever',receivers)
  return senders, receivers


def _compute_connectivity_for_batch(
    positions, n_node, radius, add_self_edges):
  """`compute_connectivity` for a batch of graphs.
  Args:
    positions: Positions of nodes in the batch of graphs. Shape:
      [num_nodes_in_batch, num_dims].
    n_node: Number of nodes for each graph in the batch. Shape:
      [num_graphs in batch].
    radius: Radius of connectivity.
    add_self_edges: Whether to include self edges or not.
  Returns:
    senders indices [num_edges_in_batch]
    receiver indices [num_edges_in_batch]
    number of edges per graph [num_graphs_in_batch]
  """

  # TODO(alvarosg): Consider if we want to support batches here or not.
  # Separate the positions corresponding to particles in different graphs.
  positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
  receivers_list = []
  senders_list = []
  n_edge_list = []
  num_nodes_in_previous_graphs = 0

  # Compute connectivity for each graph in the batch.
  for positions_graph_i in positions_per_graph_list:
    senders_graph_i, receivers_graph_i = _compute_connectivity(
        positions_graph_i, radius, add_self_edges)

    num_edges_graph_i = len(senders_graph_i)
    n_edge_list.append(num_edges_graph_i)

    # Because the inputs will be concatenated, we need to add offsets to the
    # sender and receiver indices according to the number of nodes in previous
    # graphs in the same batch.
    # print(num_nodes_in_previous_graphs)
    receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
    senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)
    # print(senders_list)
    num_nodes_graph_i = len(positions_graph_i)

    num_nodes_in_previous_graphs += num_nodes_graph_i
  # Concatenate all of the results.
  senders = np.concatenate(senders_list, axis=0).astype(np.int32)
  receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
  n_edge = np.stack(n_edge_list).astype(np.int32)
  # np.save("n_edge_save", n_edge)
  # np.save("senders",senders)
  # np.save("recievers",receivers)
  return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(
    positions, n_node, radius, add_self_edges=True):
  """`_compute_connectivity_for_batch` wrapped in a pyfunc."""
  partial_fn = functools.partial(
      _compute_connectivity_for_batch, add_self_edges=add_self_edges)
  senders, receivers, n_edge = tf.py_function(
      partial_fn,
      [positions, n_node, radius],
      [tf.int32, tf.int32, tf.int32])
  senders.set_shape([None])
  receivers.set_shape([None])
  n_edge.set_shape(n_node.get_shape())
  return senders, receivers, n_edge