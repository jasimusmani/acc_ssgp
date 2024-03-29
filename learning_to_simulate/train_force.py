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
# pylint: disable=line-too-long
"""Training script for https://arxiv.org/pdf/2002.09405.pdf with modification made for
  Accelerating Graph Networks for Physics based Simulation.

Example usage (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH}`

Evaluate model from checkpoint (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --mode=eval`

Produce rollouts (from parent directory):
`python -m learning_to_simulate.train --data_path={DATA_PATH} --model_path={MODEL_PATH} --output_path={OUTPUT_PATH} --mode=eval_rollout --rigid_body_type wheel`

"""
# pylint: enable=line-too-long
import collections
import functools
import json
import os
import pickle
# import re  # added
import timeit  # added

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tree


from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils

# import learned_simulator
# import noise_utils
# import reading_utils

# flags.DEFINE_enum(
#     'mode', 'train', ['train', 'eval', 'eval_rollout'],
#     help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_string('rigid_body_type', 'excavation', ['excavation', 'wheel'],
                    help='Type of rigid body: excavation or wheel.')

# flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
#                   help='Split to use when running evaluation.')
flags.DEFINE_enum('eval_split', 'valid', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
# flags.DEFINE_string('data_path', './learning_to_simulate/datasets/Wheel_PCA_data', help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')

flags.DEFINE_float('noise_std', 6.7e-6, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
# flags.DEFINE_string('model_path', './learning_to_simulate/models/no_noise',
#                     help=('The path for saving checkpoints of the model. '
#                           'Defaults to a temporary directory.'))

flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')
# added:
# flags.DEFINE_float('con_radius', 0.025, help=('The connectivity radius.'))
# flags.DEFINE_integer('gpu', int(0), help='Visible GPU.')
flags.DEFINE_integer('mssg', int(1), help='Number of message passing steps.')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
FORCE = True  # added

def get_kinematic_mask(particle_types):
  """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
  return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)


def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.

  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # added:
  force = tensor_dict['force']
  force = tf.transpose(force, perm=[1, 0])
  target_force = tf.expand_dims(force[:, -1], 0)
  target_position_force = tf.concat([target_position, target_force], 0) 
  tensor_dict["force"] = force[:, :-1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]

  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  
  # return tensor_dict, target_position  # removed
  return tensor_dict, target_position_force  # added


def prepare_rollout_inputs(context, features):
  """Prepares an input trajectory for rollout."""
  out_dict = {**context}
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tf.transpose(features['position'], [1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  
  # added:
  force = tf.transpose(features['force'], perm=[1, 0])
  target_force = tf.expand_dims(force[:, -1], 0)
  target_position_force = tf.concat([target_position, target_force], 0) 
  out_dict['force'] = force[:, :-1]

  # Remove the target from the input.
  out_dict['position'] = pos[:, :-1]
  # Compute the number of nodes
  out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
  if 'step_context' in features:
    out_dict['step_context'] = features['step_context']
  out_dict['is_trajectory'] = tf.constant([True], tf.bool)
  
  # return out_dict, target_position  # removed
  return out_dict, target_position_force  # added


def batch_concat(dataset, batch_size):
  """We implement batching as concatenating on the leading axis."""

  # We create a dataset of datasets of length batch_size.
  windowed_ds = dataset.window(batch_size)

  # The plan is then to reduce every nested dataset by concatenating. We can
  # do this using tf.data.Dataset.reduce. This requires an initial state, and
  # then incrementally reduces by running through the dataset

  # Get initial state. In this case this will be empty tensors of the
  # correct shape.
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)

  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))


def get_input_fn(data_path, batch_size, mode, split):
  """Gets the learning simulation input function for tf.estimator.Estimator.
  Args:
    data_path: the path to the dataset directory.
    batch_size: the number of graphs in a batch.
    mode: either 'one_step_train', 'one_step' or 'rollout'
    split: either 'train', 'valid' or 'test.

  Returns:
    The input function for the learning simulation model.
  """

  def input_fn():
    """Input function for learning simulation."""

    # Loads the metadata of the dataset.
    metadata = _read_metadata(data_path)

    # Create a tf.data.Dataset from the TFRecord.
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        # reading_utils.parse_serialized_simulation_example, metadata=metadata))  # removed
        reading_utils.parse_serialized_simulation_example, metadata=metadata, FORCE=FORCE))  # added

    if mode.startswith('one_step'):
      # Splits an entire trajectory into chunks of 7 steps.
      # Previous 5 velocities, current velocity and target.
      split_with_window = functools.partial(
          reading_utils.split_trajectory,
          window_length=INPUT_SEQUENCE_LENGTH + 1,
          FORCE=FORCE)  # added
      ds = ds.flat_map(split_with_window)

      # Splits a chunk into input steps and target steps
      ds = ds.map(prepare_inputs)

      # If in train mode, repeat dataset forever and shuffle.
      if mode == 'one_step_train':
        ds = ds.repeat()
        ds = ds.shuffle(512)

      # Custom batching on the leading axis.
      ds = batch_concat(ds, batch_size)

    elif mode == 'rollout':
      # Rollout evaluation only available for batch size 1
      assert batch_size == 1
      ds = ds.map(prepare_rollout_inputs)

    else:
      raise ValueError(f'mode: {mode} not recognized')
    return ds

  return input_fn

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

#added: Jasim
# Rigid body correction using singular value decomposition
def euclidean_transform_3D(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)
    N = A.shape[0]; 

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    centroid_A = centroid_A.reshape(1,3)
    centroid_B = centroid_B.reshape(1,3)
    
    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.matmul(np.transpose(BB) , AA)

    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    asa = np.linalg.det(U@Vt)
    diag = np.array([[1,0,0],[0,1,0],[0,0,asa]])
    R = U @diag@ Vt

    # handle svd sign problem
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R@centroid_A.T + centroid_B.T

    next_position = (np.matmul(R,np.transpose(A))) + np.tile(t,(1,16))
    next_position = np.transpose(next_position)
    next_position = next_position.reshape(16,3)

    return next_position


def my_tensorflow_function(A,B):
  return tf.py_function(euclidean_transform_3D, [A,B], tf.float32)

def rollout(simulator, features, num_steps):
  """Rolls out a trajectory by applying the model in sequence."""
  initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
  # print('shape of ground truth postions', ground_truth_positions.shape)
  # added:
  initial_forces = features['force'][:, 0:INPUT_SEQUENCE_LENGTH]
  ground_truth_forces = features['force'][:, INPUT_SEQUENCE_LENGTH:]
  
  global_context = features.get('step_context')
  def step_fn(step, current_positions, predictions, predictions_force):  # ...added

    if global_context is None:
      global_context_step = None
    else:
      global_context_step = global_context[
          step + INPUT_SEQUENCE_LENGTH - 1][tf.newaxis]

    # next_position = simulator(  # removed
    next_position, next_force = simulator(  # added
        current_positions,
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=global_context_step)

    #added: Jasim
    # Rigid body correction using SVD for Wheel
    
    if FLAGS.rigid_body_type == 'wheel':
      A = current_positions[:16,5:,:]
      A=tf.reshape(A,[16,3])
      B = next_position[:16,:]
      # print('shape of a',A.shape)
      next_position_rigid = my_tensorflow_function(A,B)
      next_position_rigid=tf.reshape(next_position_rigid,[16,3])
      
      # next_position = (np.matmul(R,A.T)) + np.tile(t,(1,16))
      next_position_a = next_position_rigid
      next_position_b = next_position[16:,:]

      #uncomment it for wheel
      next_position = tf.concat([next_position_a, next_position_b], axis=0)
    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = get_kinematic_mask(features['particle_type'])
    next_position_ground_truth = ground_truth_positions[:, step]
    next_position = tf.where(kinematic_mask, next_position_ground_truth,
                             next_position)
    updated_predictions = predictions.write(step, next_position)
    # added:
    next_force = tf.reduce_mean(next_force, 0)
    updated_predictions_force = predictions_force.write(step, next_force)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    next_positions = tf.concat([current_positions[:, 1:],
                                next_position[:, tf.newaxis]], axis=1)

    return (step + 1, next_positions, updated_predictions, updated_predictions_force)  # ...added

  predictions = tf.TensorArray(size=num_steps, dtype=tf.float32)
  predictions_force = tf.TensorArray(size=num_steps, dtype=tf.float32)  # added



  _, _, predictions, predictions_force = tf.while_loop(  # ...added
      cond=lambda step, state, prediction, predictions_force: tf.less(step, num_steps),  # ...added
      body=step_fn,
      loop_vars=(0, initial_positions, predictions, predictions_force),  # ...added
      back_prop=False,
      parallel_iterations=1)

  output_dict = {
      'initial_positions': tf.transpose(initial_positions, [1, 0, 2]),
      'predicted_rollout': predictions.stack(),
      'ground_truth_rollout': tf.transpose(ground_truth_positions, [1, 0, 2]),
      'particle_types': features['particle_type'],
      # added:
      'initial_forces': tf.transpose(initial_forces, [1, 0]),  
      'predicted_forces': predictions_force.stack(),
      'ground_truth_forces': tf.transpose(ground_truth_forces, [1, 0]),
  }

  if global_context is not None:
    output_dict['global_context'] = global_context
  return output_dict


def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)



def _get_simulator(model_kwargs, metadata, acc_noise_std, vel_noise_std):
  """Instantiates the simulator."""
  # Cast statistics to numpy so they are arrays when entering the model.
  cast = lambda v: np.array(v, dtype=np.float32)

  acceleration_stats = Stats(
      cast(metadata['acc_mean']),
      _combine_std(cast(metadata['acc_std']), acc_noise_std))
  velocity_stats = Stats(
      cast(metadata['vel_mean']),
      _combine_std(cast(metadata['vel_std']), vel_noise_std))

  # added:
  force_stats = Stats(
      cast(metadata['rigid_body_force_mean']),
      _combine_std(cast(metadata['rigid_body_force_std']), vel_noise_std))

  normalization_stats = {'acceleration': acceleration_stats,
                         'velocity': velocity_stats,
                         'force': force_stats}  # added

  if 'context_mean' in metadata:
    context_stats = Stats(
        cast(metadata['context_mean']), cast(metadata['context_std']))
    normalization_stats['context'] = context_stats

  simulator = learned_simulator.LearnedSimulator(
      num_dimensions=metadata['dim'],
      connectivity_radius=metadata['default_connectivity_radius'],
      graph_network_kwargs=model_kwargs,
      boundaries=metadata['bounds'],
      num_particle_types=NUM_PARTICLE_TYPES,
      normalization_stats=normalization_stats,
      particle_type_embedding_size=16,
      rigid_body_ref_point=metadata['rigid_body_ref_point'])  # added
  return simulator

# regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
def get_one_step_estimator_fn(data_path,
                              noise_std,
                              message_passing_steps,
                              latent_size=128,
                              hidden_size=128,
                              hidden_layers=2):
  """Gets one step model for training simulation."""
  metadata = _read_metadata(data_path)

  model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)

  def estimator_fn(features, labels, mode):
    # target_next_position = labels  # removed

    # added: If batch size is 1:
    # target_next_position = labels[:-1]
    # target_next_force = labels[-1]
    # added: If batch size is 2:
    target_next_position1 = labels[:features['n_particles_per_example'][0]]
    target_next_force1 = tf.expand_dims(labels[features['n_particles_per_example'][0]], 0)
    target_next_position2 = labels[features['n_particles_per_example'][0]+1:-1]
    target_next_force2 = tf.expand_dims(labels[-1], 0)
    target_next_position = tf.concat([target_next_position1, target_next_position2], 0)
    target_next_force = tf.concat([target_next_force1, target_next_force2], 0)

    simulator = _get_simulator(model_kwargs, metadata,
                               vel_noise_std=6.7e-7,
                               acc_noise_std=6.7e-7)
    # Sample the noise to add to the inputs to the model during training.
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        features['position'], noise_std_last_step=noise_std)
    non_kinematic_mask = tf.logical_not(
        get_kinematic_mask(features['particle_type']))
    noise_mask = tf.cast(
        non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]
    sampled_noise *= noise_mask
    # Get the predictions and target accelerations.
    pred_target = simulator.get_predicted_and_target_normalized_accelerations(
        next_position=target_next_position,
        position_sequence=features['position'],
        position_sequence_noise=sampled_noise,
        next_force=target_next_force,  # added
        n_particles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
        global_context=features.get('step_context'))
    pred_acceleration, pred_force, target_acceleration, target_force = pred_target
  
    # Calculate the loss and mask out loss on kinematic particles
    num_non_kinematic = tf.reduce_sum(
        tf.cast(non_kinematic_mask, tf.float32))

    # ...added:
    loss1 = (pred_acceleration - target_acceleration)**2
    loss1 = tf.where(non_kinematic_mask, loss1, tf.zeros_like(loss1))
    loss1 = tf.reduce_sum(loss1) / tf.reduce_sum(num_non_kinematic)


    # added: If batch size is 1:
    # loss2 = (tf.reduce_mean(pred_force, 0) - target_force)**2  
    # loss2 = tf.reduce_mean(loss2)
    # added: If batch size is 2:
    pred_force1 = tf.expand_dims(tf.reduce_mean(pred_force[:features['n_particles_per_example'][0]], 0), 0)
    pred_force2 = tf.expand_dims(tf.reduce_mean(pred_force[features['n_particles_per_example'][0]:], 0), 0)
    pred_force = tf.concat([pred_force1, pred_force2], 0)
    loss2 = tf.reduce_mean((pred_force - target_force)**2)

    vars   = tf.trainable_variables()
    lossL1 = tf.add_n([tf.losses.absolute_difference(tf.zeros_like(v), v) for v in vars]) * 0.01

    loss = loss1 + loss2 #+ lossL1 # added

    global_step = tf.train.get_global_step()

    # Set learning rate to decay from 1e-4 to 1e-6 exponentially.
    min_lr = 1e-6
    lr = tf.train.exponential_decay(learning_rate=1e-4 - min_lr,
                                    global_step=global_step,
                                    decay_steps=int(5e6),
                                    decay_rate=0.1) + min_lr
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss, global_step)

    # Calculate next position and add some additional eval metrics (only eval).
    # predicted_next_position = simulator(  # removed
    predicted_next_position, predicted_next_force = simulator(  # added
      position_sequence=features['position'],
      n_particles_per_example=features['n_particles_per_example'],
      particle_types=features['particle_type'],
      global_context=features.get('step_context'))

    # ...added:
    predictions = {
      'predicted_next_position_force': tf.concat(
        [predicted_next_position, predicted_next_force], 1)
    }

    eval_metrics_ops = {
      'loss_mse': tf.metrics.mean_squared_error(
        pred_acceleration, target_acceleration),
      'one_step_position_mse': tf.metrics.mean_squared_error(
        predicted_next_position, target_next_position),

      # added: If batch size is 1:
      # 'loss_force_mse': tf.metrics.mean_squared_error(
      #   tf.reduce_mean(pred_force, 0), target_force),
      # added: If batch size is 2:
      'loss_force_mse': tf.metrics.mean_squared_error(pred_force, target_force)
    }



    return tf.estimator.EstimatorSpec(
      mode=mode,
      train_op=train_op,
      loss=loss,
      predictions=predictions,
      eval_metric_ops=eval_metrics_ops,
    )

  return estimator_fn

def get_rollout_estimator_fn(data_path,
                             message_passing_steps,
                             latent_size=128,
                             hidden_size=128,
                             hidden_layers=2):
  """Gets the model function for tf.estimator.Estimator."""
  metadata = _read_metadata(data_path)

  model_kwargs = dict(
      latent_size=latent_size,
      mlp_hidden_size=hidden_size,
      mlp_num_hidden_layers=hidden_layers,
      num_message_passing_steps=message_passing_steps)

  def estimator_fn(features, labels, mode):
    del labels  # Labels to conform to estimator spec.
    message = []
    simulator = _get_simulator(model_kwargs, metadata, acc_noise_std= 6.7e-7,
                               vel_noise_std=6.7e-7)
    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
    rollout_op = rollout(simulator, features, num_steps=num_steps)

    squared_error = (rollout_op['predicted_rollout'] -
                     rollout_op['ground_truth_rollout']) ** 2
    loss1 = tf.reduce_mean(squared_error)  # ...added

    # added:
    squared_error = (rollout_op['predicted_forces'] - rollout_op['ground_truth_forces']) ** 2
    loss2 = tf.reduce_mean(squared_error)

    r1= rollout_op['predicted_rollout']
    r1 = r1[:,:16,:]
    r2= rollout_op['ground_truth_rollout']
    r2=r2[:,:16,:]

    squared_error1 = (r1 - r2)**2
    loss3 = tf.reduce_mean(squared_error1)
    # print(loss3.shape, loss3)
    loss = loss1 + loss2  # added

    eval_ops = {
      'rollout_error_mse': tf.metrics.mean_squared_error(
        rollout_op['predicted_rollout'], rollout_op['ground_truth_rollout']),
      # added:
      'force_error_mse': tf.metrics.mean_squared_error(
        rollout_op['predicted_forces'], rollout_op['ground_truth_forces']),
    }

    # Add a leading axis, since Estimator's predict method insists that all
    # tensors have a shared leading batch axis fo the same dims.
    rollout_op = tree.map_structure(lambda x: x[tf.newaxis], rollout_op)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=None,
        loss=loss,
        predictions=rollout_op,
        eval_metric_ops=eval_ops)

  return estimator_fn


def _read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())


def main(_):
  """Train or evaluates the model."""

  # added:
  # gpus = tf.config.experimental.list_physical_devices('GPU')
  # if gpus:
  #   # Restrict TensorFlow to only use the first GPU
  #   print(gpus)
  #   try:
  #     tf.config.experimental.set_visible_devices(gpus[int(FLAGS.gpu)], 'GPU')
  #   except RuntimeError as e:
  #     # Visible devices must be set at program startup
  #     print(e)


  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 16
  config.inter_op_parallelism_threads = 16
  config.gpu_options.allow_growth = True
  tf.estimator.RunConfig(session_config=config)


  os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Use CPU


  if FLAGS.mode in ['train', 'eval']:
    estimator = tf.estimator.Estimator(
        get_one_step_estimator_fn(FLAGS.data_path, FLAGS.noise_std, FLAGS.mssg),
        model_dir=FLAGS.model_path)

    if FLAGS.mode == 'train':

      # Train all the way through.
      estimator.train(
          input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
            mode='one_step_train', split='train'),
          max_steps=FLAGS.num_steps,
          )  # added

    else:
      # One-step evaluation from checkpoint.
      eval_metrics = estimator.evaluate(input_fn=get_input_fn(
          FLAGS.data_path, FLAGS.batch_size,
          mode='one_step', split=FLAGS.eval_split))
      logging.info('Evaluation metrics:')
      logging.info(eval_metrics)
  elif FLAGS.mode == 'eval_rollout':
    if not FLAGS.output_path:
      raise ValueError('A rollout path must be provided.')
    
    start = timeit.default_timer()  # added

    rollout_estimator = tf.estimator.Estimator(
        get_rollout_estimator_fn(FLAGS.data_path, FLAGS.mssg),
        model_dir=FLAGS.model_path)
    # added:
    hooks = [tf.estimator.ProfilerHook(
        save_steps=1,
        # save_secs=1,
        output_dir=os.path.join(FLAGS.model_path, "tracing_new"),
        show_dataflow=False,
        show_memory=True)]

    # Iterate through rollouts saving them one by one.
    metadata = _read_metadata(FLAGS.data_path)
    rollout_iterator = rollout_estimator.predict(
        input_fn=get_input_fn(FLAGS.data_path, batch_size=1,
            mode='rollout', split=FLAGS.eval_split), 
      )

    for example_index, example_rollout in enumerate(rollout_iterator):
      # if example_index==1:
      #   break
      example_rollout['metadata'] = metadata
      filename = f'rollout_{FLAGS.eval_split}_{example_index}.pkl'
      filename = os.path.join(FLAGS.output_path, filename)
      logging.info('Saving: %s.', filename)
      if not os.path.exists(FLAGS.output_path):
        os.mkdir(FLAGS.output_path)
      with open(filename, 'wb') as file:
        pickle.dump(example_rollout, file)
    
      stop = timeit.default_timer()  # added

      print('Time (with serializing output): ', stop - start)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  app.run(main)
