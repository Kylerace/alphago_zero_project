import gym
import gym_go
from gym_go.envs import GoEnv

from bespoke_go import *
from tree_search import *

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.engine.sequential import Sequential
from keras.engine.training import Model
from typing import Union, List, Tuple
import collections
from gym import Space, spaces
from gym.spaces.discrete import Discrete
import sklearn
from sklearn.neighbors import NearestNeighbors
import math
from enum import Enum
import collections


experiences = collections.deque(maxlen = 500_000)
current_model: GoModel = None

def sample_experiences(experiences: collections.deque, samples):
    assert samples > 0
    indices = np.random.choice(len(experiences), min(samples, len(experiences)), replace = False)

    sampled_timesteps = [experiences[idx] for idx in indices]

    return sampled_timesteps

def loss_function(predicted_value, outcome, model_action_probabilities, mcts_action_probabilities):
    val_loss = keras.losses.mean_squared_error(tf.expand_dims(outcome, 0), tf.expand_dims(predicted_value, 0))
    pol_loss = keras.losses.categorical_crossentropy(mcts_action_probabilities, model_action_probabilities)

    return val_loss + pol_loss

#TODOKYLER: put this on another thread and introduce locks to realign the data.
#pass timesteps to the training thread from the inference thread after every game
#pass the new model to the inference thread from the training thread after every x games
#@tf.function
def train_model(model: GoModel, optimizer, timestep_lists: list):

    for timestep in timestep_lists:
        state = timestep[TIMESTEP_STATE_INDEX]
        mcts_action_probabilities = timestep[TIMESTEP_PROBABILITY_DISTRIBUTION_INDEX]
        outcome = timestep[TIMESTEP_OUTCOME_INDEX]

       #with tf.GradientTape() as tape:
            
        #    model_action_probabilities, model_value = model(state, training = True)

        #    loss = loss_function(model_value, outcome, model_action_probabilities, mcts_action_probabilities)

        #gradients = tape.gradient(loss, model.trainable_variables)
        gradients = train_step(model, state, mcts_action_probabilities, outcome)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #print(f"loss: {loss}")

@tf.function
def train_step(model, state, mcts_action_probabilities, outcome):
    with tf.GradientTape() as tape:
            
        model_action_probabilities, model_value = model(state, training = True)

        loss = loss_function(model_value, outcome, model_action_probabilities, mcts_action_probabilities)

    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients
    