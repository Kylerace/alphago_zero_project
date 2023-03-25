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
import sklearn
from sklearn.neighbors import NearestNeighbors
import math

import gym_go
from gym_go.envs import GoEnv
import copy
from copy import deepcopy

import time
from time import perf_counter

MAX, MIN = 10000, -10000

#each node is of the form: [cached value, board state, [children]]
#leaf nodes have an empty child list

NODE_VALUE_INDEX = 0
NODE_BOARDSTATE_INDEX = 1
NODE_CHILDREN_INDEX = 2

            #maximizer
test_tree = [5, "A", #in a real tree the second index is a deepcopy of the game environment at that point, right now its just an identification string
            [   #minimizer
                [10, "B",
                    #maximizer
                    [
                    [3, "D",
                        [   #minimizer
                            [-4, "H", []], 
                            [1, "I", []], 
                            [2, "J", []]
                        ]
                    ], 
                    [-5, "E",
                        [
                            [8, "K", []], 
                            [-8, "L", []], 
                            [-9, "M", []]
                        ]
                    ]
                    ]
                ], 
                #minimizer
                [3, "C",
                    #maximizer
                    [
                    [2, "F",
                        [   #minimizer
                            [9, "N", []], 
                            [4, "O", []], 
                            [5, "P", []]
                        ]
                    ], 
                    [10, "G",
                        [
                            [-3, "Q", []], 
                            [4, "R", []], 
                            [3, "S", []]
                        ]
                    ]
                    ]
                ]
            ]
        ]

def tabulate(depth, text):
    ret = ""
    for _ in range(depth-1):
        ret = f"{ret}\t"
    print(f"{ret}{text}")

# we get a node containing a value, a board state, and subsequent nodes. 
# when is_maximizing_player is True, return the maximal non-pruned node that is descended from the given node
# when it is False, return the minimal value node that is descended from the given node
def ab_minimax(node: list, depth: int, is_maximizing_player: bool, alpha: float, beta: float, time_budget = 0, agent = None, debug = False):

    #i think only leaf nodes should get expanded, as long as all nodes get expanded at least once before pruning them,
    #so theres no concept of a node that isnt yet adequately searched but should be, other than leaf nodes

    if len(node[NODE_CHILDREN_INDEX]) == 0: #if the node is a leaf, try to find new moves
        node[NODE_CHILDREN_INDEX], time_budget = create_new_ab_nodes_from(node, agent, is_maximizing_player, time_budget, depth, 1.0)

        if len(node[NODE_CHILDREN_INDEX]) == 0:
            val, state = node[NODE_VALUE_INDEX], node[NODE_BOARDSTATE_INDEX]
            if debug == True:
                max_min_string = "MAX:" if is_maximizing_player == True else "MIN:"
                tabulate(depth, f"{max_min_string} checking LEAF node: {node[NODE_BOARDSTATE_INDEX]} ({node[NODE_VALUE_INDEX]}) at depth: {depth}, alpha|beta: {alpha}|{beta}")
            return val, state, val, state
    
    if debug == True:
        max_min_string = "MAX:" if is_maximizing_player == True else "MIN:"
        tabulate(depth, f"{max_min_string} checking node: {node[NODE_BOARDSTATE_INDEX]} ({node[NODE_VALUE_INDEX]}) at depth: {depth}, alpha|beta: {alpha}|{beta}")

    current_node = node[NODE_BOARDSTATE_INDEX]
    next_node = None

    direct_best_node = None    


    if is_maximizing_player:
        best_value = MIN
        direct_best_value = MIN
        
        #TODOKYLER: search possible actions here, passing in the time budget, the board state, and the agent
        # legal actions -> agent normalized quality of each state : action pair -> set search time for each node path
        # -> search new paths recursively until each path runs out of budget 
        # -> when everythings done alpha beta search the new tree for the best move
        for child_node in node[NODE_CHILDREN_INDEX]: 
                
            # we want to find the best node for us that maximizes our position, 
            # each node's value for us is set to the value of the board after the enemy makes their strongest move
            child_value, minimized_node, _, __ = ab_minimax(child_node, depth + 1, False, alpha, beta, debug=debug)

            if child_value > best_value:
                best_value = child_value
                next_node = minimized_node

                direct_best_value = child_node[NODE_VALUE_INDEX]
                direct_best_node = child_node[NODE_BOARDSTATE_INDEX]

            if debug == True and best_value > alpha:
                tabulate(depth, f"\talpha @ {depth} going from {alpha} to {best_value} via {next_node} from {current_node} {node[NODE_VALUE_INDEX]}'s child {child_node[NODE_BOARDSTATE_INDEX]} {child_node[NODE_VALUE_INDEX]}")
            alpha = max(alpha, best_value)

            if beta <= alpha: #they found a move after this gives us a bad outcome if we take this path
                if debug == True:
                    tabulate(depth, f"\t\tmaximizer beta pruning children of {node[NODE_BOARDSTATE_INDEX]} {node[NODE_VALUE_INDEX]} at {child_node[NODE_BOARDSTATE_INDEX]} {child_node[NODE_VALUE_INDEX]} with alpha|beta: {alpha}|{beta}")
                break
            
        return best_value, next_node, direct_best_value, direct_best_node
    
    # if we are the minimizing player, we want to find the move that minimizes the quality of the action taken before this node
    # we are still the same player, we're just trying to find what our enemy will play on their turn

    best_value = MAX
    best_leaf_value = MAX

    for child_node in node[NODE_CHILDREN_INDEX]:
            
        child_value, maximized_node, _, __ = ab_minimax(child_node, depth + 1, True, alpha, beta, debug=debug)

        if child_value < best_value:
            best_value = child_value
            next_node = maximized_node

            direct_best_value = child_node[NODE_VALUE_INDEX]
            direct_best_node = child_node[NODE_BOARDSTATE_INDEX]

        if debug == True and best_value < beta:
            tabulate(depth, f"beta @ {depth} going from {beta} to {best_value} via {next_node} from {current_node} {node[NODE_VALUE_INDEX]}'s child {child_node[NODE_BOARDSTATE_INDEX]} {child_node[NODE_VALUE_INDEX]}")
        beta = min(beta, best_value)

        if beta <= alpha:
            if debug == True:
                tabulate(depth, f"minimizer beta pruning children of {node[NODE_BOARDSTATE_INDEX]} {node[NODE_VALUE_INDEX]} at {child_node[NODE_BOARDSTATE_INDEX]} {child_node[NODE_VALUE_INDEX]} with alpha|beta: {alpha}|{beta}")
            break

    return best_value, next_node, direct_best_value, direct_best_node

def create_new_ab_nodes_from(current_node: list, quality_model, is_maximizing_player: bool, overall_time_budget: float, depth_from_last_actual_move: int, gamma: float):
    #create/get legal move bitboard
    #pass it to quality_model, get quality bitboard for each move (punish illegal moves?)
    #sort moves by quality, recalculate remaining time, allocate it to each move and then recursively call ourselves again
    #with is_maximizing_player flipped, when each call runs out of time  pass up the new nodes
    #then when all time is up, 
    
    return current_node[NODE_CHILDREN_INDEX], overall_time_budget

def create_node(board: GoEnv, action):
    copied_board = deepcopy(board)
    state, reward, done, info = copied_board.step(action)
    return [reward, copied_board, []]