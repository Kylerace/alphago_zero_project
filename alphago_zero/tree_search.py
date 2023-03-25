from bespoke_go import *

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.engine.sequential import Sequential
from keras.engine.training import Model
from typing import Union, List, Tuple
import collections
import gym
from gym import Space, spaces
from gym.spaces.discrete import Discrete
import math

import gym_go
from gym_go.envs import GoEnv
import copy

import time
from time import perf_counter

import numpy as np
import random



from keras.layers.convolutional.conv2d import Conv2D
from keras.layers import ReLU
from keras.layers import BatchNormalization
from tensorflow import Tensor
from tensorflow.python.ops.gen_nn_ops import relu
from tensorflow.python.ops.gen_math_ops import tanh
from keras.layers import Dense

class ConvBlock(tf.keras.Model):

    def __init__(self, filters = 256, kernel_size = 3):
        super(ConvBlock, self).__init__()

        self.conv_input = Conv2D(filters = filters, kernel_size = kernel_size, padding = "same", strides=(1, 1))
        self.batch_norm = BatchNormalization()

    def __call__(self, input_tensor: Tensor, training = False):
        a = self.conv_input(input_tensor)
        b = self.batch_norm(a, training = training)
        ret = relu(b)
        return ret

class ResidualBlock(tf.keras.Model):
    
    def __init__(self, filters = 256, kernel_size = 3):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2D(filters = filters, kernel_size = kernel_size, padding = "same", strides=(1, 1))
        self.batch_norm1 = BatchNormalization()

        self.conv2 = Conv2D(filters = filters, kernel_size = kernel_size, padding = "same", strides=(1, 1))
        self.batch_norm2 = BatchNormalization()

    def __call__(self, input_tensor: Tensor, training = False, mask = None):

        a = self.conv1(input_tensor)
        b = self.batch_norm1(a, training = training)
        c = relu(b)
        d = self.conv2(c)
        e = self.batch_norm2(d, training = training)
        add = e + input_tensor
        ret = relu(add)
        return ret

class GoModel(tf.keras.Model):

    #TODOKYLER: figure out saving and loading from a file
    def __init__(self, boardsize = 7, blocks = 20):
        super().__init__()

        blocks -= 1 #this includes the conv input block

        #input_shape = (1?, boardsize, boardsize)

        #TODOKYLER: augment the data by rotating the input randomly (go is symmetrical)
        self.conv_input = ConvBlock(filters = 256, kernel_size = 3)
        self.residual_tower = Sequential([ResidualBlock(filters = 256, kernel_size = 3) for block in range(blocks)], name = f"Residual_Tower_{blocks}_Blocks")

        #POLICY head
        self.policy_conv = Conv2D(filters = 2, kernel_size = 1, padding = "same", strides=(1, 1))
        self.policy_batch_norm = BatchNormalization()
        #relu
        self.policy_connected = Dense(units = (boardsize ** 2), activation=None)
        # -> logit probabilities of every intersection + pass

        #VALUE head

        self.value_conv = Conv2D(filters = 1, kernel_size = 1, padding = "same", strides=(1, 1))
        self.value_batch_norm = BatchNormalization()
        #relu
        self.value_connected = Dense(units = 1, activation = None)
        #tanh nonlinearity

        #self.common = Sequential([layers.Dense(neurons_per_layer, activation=activation) for _ in range(number_of_layers)])

    @tf.function()
    def __call__(self, input_tensor: Tensor, training = False) -> Tuple[Tensor, Tensor]:
        #x1 = tf.convert_to_tensor(input_tensor, dtype = tf.float64)
        #x2 = tf.expand_dims(x1, axis = 0)
        #input_tensor = tf.reshape(x2, (1, 7, 7, 17))
        input_tensor = tf.reshape(tf.expand_dims(tf.convert_to_tensor(input_tensor, dtype = tf.float64), axis = 0), (1, 7, 7, 17))
        a = self.conv_input(input_tensor, training)
        b = self.residual_tower(
            a, 
            training
            )

        pc = self.policy_conv(b)
        pd = self.policy_batch_norm(pc)
        pe = relu(pd)
        s_pe = tf.squeeze(pe)
        r_spe = tf.reshape(s_pe, (1, 98))
        pf = self.policy_connected(r_spe)
        #s_pf = tf.squeeze(pf)


        vc = self.value_conv(b)
        vd = self.value_batch_norm(vc)
        ve = relu(vd)
        r_ve = tf.reshape(ve, (1, 49))
        vf = self.value_connected(r_ve)
        vg = tanh(vf)

        scalar_vg = tf.reshape(vg, ())
        action_probabilities: tf.Tensor = tf.squeeze(tf.nn.softmax(pf))

        return action_probabilities, scalar_vg

        #policy, value
        return ( #POLICY: output is the n*n+1 logit probabilities of each move
            self.policy_connected(
                relu(
                    self.policy_batch_norm(
                        self.policy_conv(
                            b
                        )
                    )
                )
            )
            ,
            tanh( #VALUE: output is a single scalar within the range [-1, 1]
                self.value_connected(
                    relu(
                        self.value_batch_norm(
                            self.value_conv(
                                b
                            )
                        )
                    )
                )
            )
        )

#not sure if this is essentially the same as alphago's coefficient but this is an exponent now, 0.0001 = exploitation, 1+ = exploration
MCT_TEMPERATURE = 1

MCT_BOARDSTATE_INDEX = 0
# 
MCT_CACHED_Q_INDEX = 1
# the value of each node is added to every father node when its calculated. this is used to calculate the quality
MCT_CACHED_VALUES_INDEX = 2
# incremented every time we are visited
MCT_VISITS_INDEX = 3
# created when the network first evaluates the softmax'd action probabilities of this position as a leaf node, never updated
MCT_PRIOR_PROBABILITY_INDEX = 4
#the action used to get to us from parent
MCT_ACTION_INDEX = 5
MCT_TERMINAL_STATE_INDEX = 6
MCT_VALID_MOVES_INDEX = 7
MCT_CHILDREN_INDEX = 8

#i think mean value is just the mean reward of all simulations here and below?
#boardstate, total reward of simulations, total simulations at node & all descendants, child nodes
example_mc_node = ["A", 0, 0, 0, 0, 0, False, None, []]

# for i in range(num_iterations): {selection -> expansion -> evaluation -> backpropagation} -> play

#expand and search the tree search_iterations times. doesnt actually pick anything itself, just returns debug info
def monte_carlo_tree_search(root_node: list, model: tf.keras.Model, search_iterations = 100, c_puct = 0.5, epsilon = 0.75, alpha = 0.1, go_env:GoEnv = None, print_perf = False):
    mcts_start = perf_counter()
    inference_total = 0
    JUST_inference_total = 0

    iterations_completed = 0

    nodes_created = 0
    nodes_selected = 0
    nodes_PUCTed = 0
    nodes_backpropped = 0
    leaf_nodes_found = 0

    max_depth = 0

    current_depth = 0

    for iteration in range(search_iterations):
        current_depth = 1
        selection_path = []

        #SELECTION: find a leaf node according to some tradeoff between exploitation and exploration, and check if its terminal

        parent_node = None
        next_node = root_node
        while True:
            selection_path.append(next_node)

            if len(next_node[MCT_CHILDREN_INDEX]) == 0: 
                leaf_nodes_found += 1

                if next_node[MCT_TERMINAL_STATE_INDEX] == True: #terminal, already explored (i think)
                    print("found a terminal node!")
                #    continue #TODOKYLER: figure out what to do for terminal nodes
                #else:
                
                if go_env != None:
                    i = -1
                    for selected_node in selection_path:
                        i += 1
                        if i == 0:
                            print("______MCTS SELECT START______")
                        else:
                            print("            V      ")

                        print(str2(selected_node[MCT_BOARDSTATE_INDEX]))
                        print("legal moves:", selected_node[MCT_VALID_MOVES_INDEX])

                    print("______MCTS SELECT END______")
                    print(go_env)
                    x=1
                break 

            # Upper Confidence Bound (UCB) selection for already explored nodes
            child_nodes = next_node[MCT_CHILDREN_INDEX]
            length = len(child_nodes)
            nodes_PUCTed += length

            sqrt_sum_visits = math.sqrt(sum([child_nodes[n][MCT_VISITS_INDEX] for n in range(length)]))

            # Polynomial Upper Confidence (Bound for) Trees
            # TODOKYLER: add dirichlet noise

            noise = np.random.dirichlet([alpha] * length)

            def PUCT(node):
                return node[MCT_CACHED_Q_INDEX] + c_puct * node[MCT_PRIOR_PROBABILITY_INDEX] * (
                    sqrt_sum_visits / (node[MCT_VISITS_INDEX] + 1)
                    )

            best_node = None
            best_value = -100000

            index = -1
            for node in child_nodes:
                index += 1
                node_value = node[MCT_CACHED_Q_INDEX] + c_puct * ((1 - epsilon) * node[MCT_PRIOR_PROBABILITY_INDEX] + epsilon * noise[index]) * (
                    sqrt_sum_visits / (node[MCT_VISITS_INDEX] + 1)
                    )
                
                if node_value > best_value:
                    best_value = node_value
                    best_node = node

            parent_node = next_node
            next_node = best_node #max(child_nodes, key = PUCT)

            nodes_selected += 1

            current_depth += 1
            max_depth = max(max_depth, current_depth)


        # EXPANSION: find every legal action from the leaf, and create new nodes from them

        #now next_node is an unexplored leaf node

        current_state = next_node[MCT_BOARDSTATE_INDEX]
        current_valid_moves = next_node[MCT_VALID_MOVES_INDEX]

        #print("now calling the model on the state")

        inference_start = perf_counter()
        #create a custom model that translates the state as we need, or an env wrapper i guess
        #with tf.device("/device:GPU:0"):

        action_probabilities_, leaf_value = model(current_state, False)
        JUST_inference_total += (perf_counter() - inference_start) * 1000

        #print(f"BEFORE {action_probabilities_.device=}")
        with tf.device("/device:CPU:0"):
            action_probabilities_var = tf.Variable(action_probabilities_)

        #print(f"AFTER {action_probabilities_var.device=}")

        action_probabilities = action_probabilities_var.numpy()
        #action_probabilities: tf.Tensor = tf.squeeze(tf.nn.softmax(action_logits))
        leaf_value = leaf_value.numpy()

        inference_total += (perf_counter() - inference_start) * 1000

        #print(model.__call__.pretty_printed_concrete_signatures())

        expanded_nodes: list = next_node[MCT_CHILDREN_INDEX]

        index = -1
        for legal_move in current_valid_moves:
            index += 1
            if legal_move == 0:
                continue


            new_node = create_mcts_node(current_state, action_probabilities[index], current_valid_moves, index)
            expanded_nodes.append(new_node)
            nodes_created += 1

        # BACKPROPAGATION: move back up the tree, updating the statistics of each ancestor node of the leaf

        for node_to_update in reversed(selection_path):
            node_to_update[MCT_VISITS_INDEX] += 1
            node_to_update[MCT_CACHED_VALUES_INDEX] += leaf_value

            node_to_update[MCT_CACHED_Q_INDEX] = node_to_update[MCT_CACHED_VALUES_INDEX] / node_to_update[MCT_VISITS_INDEX]

            nodes_backpropped += 1

        iterations_completed += 1

    mcts_end = (perf_counter() - mcts_start) * 1000
    if print_perf == True:
        print(f"MCTS total: {mcts_end} ms, inference + convert: {inference_total} ms (just inference {JUST_inference_total} ms)")

    return iterations_completed, nodes_created, nodes_selected, nodes_PUCTed, nodes_backpropped, leaf_nodes_found, max_depth

#actually pick the next action from the mcts probability distribution, returning the new root, and the time step list
# of the form: [new state, mcts probability distribution, and the outcome (filled in after the end of the game)]
def mcts_decide(root_node: list, legal_moves, v_resign: float, temperature = MCT_TEMPERATURE):

    # PLAY: we have searched enough, now find the best move from the root node

    best_node = None
    best_nodes = [] #all nodes with an equal value
    best_action_value = -100000000

    t_probs: tf.TensorArray = tf.TensorArray(dtype = tf.float32, size = len(legal_moves))
    t_probs = t_probs.write(0, 0) #just so it gets filled as an array of zeros

    inverse_temperature = 1 / temperature
    total_visit_count = root_node[MCT_VISITS_INDEX] if temperature == 1 else sum([(child_node[MCT_VISITS_INDEX] ** inverse_temperature) for child_node in root_node[MCT_CHILDREN_INDEX]])

    if len(root_node[MCT_CHILDREN_INDEX]) == 0:
        print("no child nodes found! resigning (this might be an error)")
        best_action = len(legal_moves)
        return best_action, root_node, [root_node[MCT_BOARDSTATE_INDEX], t_probs.stack(), 0]

    total_value = 0
    for child_node in root_node[MCT_CHILDREN_INDEX]:

        child_value = (child_node[MCT_VISITS_INDEX] ** inverse_temperature) / (total_visit_count)

        total_value += child_value
        #mcts_action_probabilities[child_node[MCT_ACTION_INDEX]] = child_value
        t_probs = t_probs.write(child_node[MCT_ACTION_INDEX], child_value)

        if child_value == best_action_value:
            best_nodes.append(child_node)

        elif child_value > best_action_value:
            best_action_value = child_value
            best_nodes = [child_node]
        else:
            x = 1

    print(f"{total_value=}")

    if len(best_nodes) == 0:
        raise ValueError("no move selected by mcts_decide()!")
    
    best_node = random.choice(best_nodes)

    best_action = best_node[MCT_ACTION_INDEX]

    if best_action_value < v_resign and root_node[MCT_CACHED_Q_INDEX] < v_resign: #TODOKYLER: using Q here is wrong i think?
        best_action = len(legal_moves) #resign

    #next action, new root node, and time step array used for training
    return best_action, best_node, [best_node[MCT_BOARDSTATE_INDEX], t_probs.stack(), 0]

#MCT_BOARDSTATE_INDEX = 0
# 
#MCT_CACHED_Q_INDEX = 1
# the value of each node is added to every father node when its calculated. this is used to calculate the quality
#MCT_CACHED_VALUES_INDEX = 2
# incremented every time we are visited
#MCT_VISITS_INDEX = 3
# created when the network first evaluates the softmax'd action probabilities of this position as a leaf node, never updated
#MCT_PRIOR_PROBABILITY_INDEX = 4
#the action used to get to us from parent
#MCT_ACTION_INDEX = 5
#MCT_TERMINAL_STATE_INDEX = 6
#MCT_VALID_MOVES_INDEX = 7
#MCT_CHILDREN_INDEX = 8

def create_mcts_node(state, prior_probability, old_valid_moves, action):
    #return state, is_done_now, new_invalid_moves
    #next_state, reward, done, info = board.step(action)
    new_state, is_done_now, new_valid_moves = next_state2(state, action, old_valid_moves)
    
    return [new_state, 0, 0, 0, prior_probability, action, is_done_now, new_valid_moves, []]

def realign_to_opponents_move(current_root_node: list, enemy_action, boardsize):
    if enemy_action == None or enemy_action == boardsize ** 2: #resigned
        return current_root_node
    
    for child_node in current_root_node[MCT_CHILDREN_INDEX]:
        if child_node[MCT_ACTION_INDEX] == enemy_action:
            return child_node

#retur
def dont_repeat_search(starting_time, current_time, *args, **kwargs):
    return False

# decides how much time will be spent on each move
class SearchTimeManager:
    def start(self):
        return
    
    def end(self):
        return
    
    def continue_searching(self):
        return False

def play_move(game: GoEnv, model: tf.keras.Model, root_node: list, timestamps: list, search_iterations = 100, v_resign = 0.05, temperature = MCT_TEMPERATURE):
    
    if not game or not model or not root_node:
        raise ValueError("play_move() doesnt have required arguments!")
    
    iterations_completed, nodes_created, nodes_selected, \
        nodes_PUCTed, nodes_backpropped, leaf_nodes_found, max_depth \
        = monte_carlo_tree_search(root_node, model, search_iterations)
    
    if iterations_completed != search_iterations:
        raise ValueError(f"monte_carlo_tree_search() only completed {iterations_completed} iterations instead of {search_iterations}!")
        

    #return best_node[MCT_ACTION_INDEX], best_node, [best_node[MCT_BOARDSTATE_INDEX], mcts_action_probabilities, 0]
    next_action, new_root_node, timestamp_list = mcts_decide(root_node, game.valid_moves(), v_resign, temperature)

    # i dont know how to deal with the game boards in the tree and the real board getting desync'd. deal with it when it comes up

    timestamps.append(timestamp_list)

    state, reward, done, info = game.step(next_action)

    return state, reward, done, info, new_root_node, timestamp_list
    
def on_game_start():
    pass

TIMESTEP_STATE_INDEX = 0
TIMESTEP_PROBABILITY_DISTRIBUTION_INDEX = 1
TIMESTEP_OUTCOME_INDEX = 2

#train the model
def on_game_end(model: tf.keras.Model, timestamps, outcome):
    # 1 == black won
    # -1 == white won
    multi = -1
    for timestamp in reversed(timestamps):
        multi *= -1
        timestamp[TIMESTEP_OUTCOME_INDEX] = outcome * multi
        #TODOKYLER: make this work for alternating players? how does zero do this in general?


def str_rank(state, root_node):
    board_str = ''

    size = state.shape[1]

    all_action_visit_counts = [0 for _ in range(size ** 2)]
    ranked_actions = [0 for _ in range(size ** 2)]

    action_num = 0
    for node in root_node[MCT_CHILDREN_INDEX]:
        all_action_visit_counts[node[MCT_ACTION_INDEX]] = node[MCT_VISITS_INDEX]

    for action_index in range(len(all_action_visit_counts)):
        action_visits = all_action_visit_counts[action_index]
        action_rank = 1

        for other_action_index in range(len(all_action_visit_counts)):
            if other_action_index == action_index:
                continue

            other_action_visits = all_action_visit_counts[other_action_index]

            if other_action_visits > action_visits:
                action_rank += 1

        ranked_actions[action_index] = action_rank

    board_str += '\t'
    for i in range(size):
        board_str += '{}'.format(i).ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += '{}\t'.format(i)
        for j in range(size):
            if state[0, i, j] == 1: #black
                board_str += '○'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            elif state[1, i, j] == 1: #white
                board_str += '●'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            else:
                if i == 0:
                    if j == 0:
                        board_str += '╔═'
                    elif j == size - 1:
                        board_str += '╗'
                    else:
                        board_str += '╤═'
                elif i == size - 1:
                    if j == 0:
                        board_str += '╚═'
                    elif j == size - 1:
                        board_str += '╝'
                    else:
                        board_str += '╧═'
                else:
                    if j == 0:
                        board_str += '╟─'
                    elif j == size - 1:
                        board_str += '╢'
                    else:
                        board_str += '┼─'
        board_str += '\n'

    #black_area, white_area = areas2(state)

    #ppp = prev_player_passed2(state)
    t = turn2(state)

    board_str += '\tTurn: {}\n'.format('BLACK' if t == 0 else 'WHITE')
    #board_str += '\tBlack Area: {}, White Area: {}\n'.format(int(black_area), int(white_area))
    return board_str