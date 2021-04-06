from tf_agents.environments import tf_py_environment
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
import datetime

import tensorflow as tf

import testEnv
import log


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    max = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        if episode_return > max: max = episode_return

    avg_return = total_return / num_episodes
    if max == 0.0:
        return avg_return.numpy()[0], max
    return avg_return.numpy()[0], max.numpy()

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

def run(discount = 0.95, size=100, balP = 0.25, pecP = 0.25):
    # How long should training run?
    num_iterations = size*10*10
    # How many initial random steps, before training start, to
    # collect initial data.
    initial_collect_steps = 1000 
    # How many steps should we run each iteration to collect 
    # data from.
    collect_steps_per_iteration = 100
    # How much data should we store for training examples.
    replay_buffer_max_length = 10000

    batch_size = 64  
    learning_rate = 1e-4 
    # How often should the program provide an update.
    log_interval = 250  

    # How many episodes should the program use for each evaluation.
    num_eval_episodes = size
    # How often should an evaluation occur.
    eval_interval = size*10

    train_py_env = testEnv.PhylogenyEnv(False, discount, size,balP, pecP)
    eval_py_env = testEnv.PhylogenyEnv(True, discount, size,balP, pecP)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
      train_env.observation_spec(),
      train_env.action_spec(),
      fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                  train_env.action_spec())

  
    avg, max = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    print(str(avg) + " max: " + str(max))

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length)


    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, 
      sample_batch_size=batch_size, 
      num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return, max = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    print(datetime.datetime.now().time())

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return, max = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1} , Max Return = {2}'.format(step, avg_return, max))
            returns.append(avg_return)

    print(datetime.datetime.now().time())

    out = "["
    for x in returns:
        out += str(x) + ","

    out = out[:-2] + "]"
    print(out)

    text = "Size: "+str(size)+" discount: "+str(discount)+" result: "+out
    log.logging(text)

    return out

def runForAcc(discount = 0.95, size=100, balP = 0.25, pecP = 0.25):
    # How long should training run?
    num_iterations = size*10*10
    print(num_iterations)
    # How many initial random steps, before training start, to
    # collect initial data.
    initial_collect_steps = 1000 
    # How many steps should we run each iteration to collect 
    # data from.
    collect_steps_per_iteration = 100
    # How much data should we store for training examples.
    replay_buffer_max_length = 10000

    batch_size = 64  
    learning_rate = 1e-4 
    # How often should the program provide an update.
    log_interval = 250  

    # How many episodes should the program use for each evaluation.
    num_eval_episodes = size
    # How often should an evaluation occur.
    eval_interval = size*10

    train_py_env = testEnv.PhylogenyEnv(False, discount, size,balP, pecP)
    eval_py_env = testEnv.PhylogenyEnv(True, discount, size,balP, pecP)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
      train_env.observation_spec(),
      train_env.action_spec(),
      fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                  train_env.action_spec())

  
    avg, max = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    print(str(avg) + " max: " + str(max))

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length)


    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, 
      sample_batch_size=batch_size, 
      num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return, max = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    print(datetime.datetime.now().time())

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return, max = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1} , Max Return = {2}'.format(step, avg_return, max))
            returns.append(avg_return)

    print(datetime.datetime.now().time())

    out = "["
    for x in returns:
        out += str(x) + ","

    out = out[:-2] + "]"
    print(out)

    text = "FOR ACCURACY Size: "+str(size)+" discount: "+str(discount) + " balP: " + str(balP) + " pecP: " + str(pecP) +" result: " + out
    log.logging(text)

    return agent.policy