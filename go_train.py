import argparse
from tree_search import *
from go_wrapper import *
from self_play import *
from play_functions import *
from train import *

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

#tf.debugging.set_log_device_placement(True)
print(tf.config.list_logical_devices('GPU'))

x=1

test_agent = GoModel(boardsize = 7, blocks = 10)

bad_agent = GoModel(boardsize = 7, blocks = 1)

#probabilities, value 
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
test_agent.compile(
    optimizer = optimizer,
    #loss = [keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics = ["accuracy"]
    )

value_loss = keras.losses.MeanSquaredError()
policy_loss = keras.losses.CategoricalCrossentropy()

value_metric = keras.metrics.MeanSquaredError()
policy_metric = keras.metrics.CategoricalAccuracy()

#create a model that finds the quality of every legal action in bitboard form
#search the future space of legal moves with the alpha beta tree search, alloting the available search time with 
#the relative amount the model likes each option over the others, and picking the max quality action when the time is up

# Initialize environment
go_env: GoEnv = gym.make('gym_go:go-v0', size=7, komi=0)
go_env = GoWrapper(go_env)

go_env.reset()

#if the game gets to this many turns it ends
GAME_TURN_LIMIT = 300

self_play_episodes = 100


evaluation_games = 20

#print(test_agent.summary())

can_ever_render = True

render_now = can_ever_render

episodes = 10000

# the model that is used for self play
alpha_model = None

max_experiences = 100_000
experiences = collections.deque(maxlen = max_experiences)#tf.queue.RandomShuffleQueue()
max_experiences_per_training_run = 500

turns_before_low_temp = 30
high_temperature = 1
low_temperature = 0.05
temperature = high_temperature

c_puct = 0.3
alpha = 0.1
epsilon = 0.75



player_A = bad_agent
player_A_name = "untrainable agent"
player_A_wins = 0

player_B = test_agent
player_B_name = "training agent"
player_B_wins = 0

black_wins = 0
white_wins = 0
total_games = 0

old_black_player = None
old_white_player = None

black_player = player_A
white_player = player_B

player_A_side = "b"
player_B_side = "w"

# Game loop
for game_number in range(1, 10001):

    print(f"______________NOW STARTING GAME {game_number}______________")

    temperature = high_temperature

    timestep_list = []

    black_root = None
    white_root = None

    black_action = None
    white_action = None

    done = False
    turn_counter = 0

    def print_turn(turn_counter):
        print(f"TURN NUMBER: {turn_counter}, {player_A_name} ({player_A_side}): {player_A_wins}, {player_B_name} ({player_B_side}): {player_B_wins} (b {black_wins}|w {white_wins}|total {total_games})")

    while not done:
        
        if turn_counter > turns_before_low_temp:
            temperature = low_temperature

        state, reward, done, info, black_root, timestep_list, black_action = play_agent(go_env, black_player, black_root, white_action, timestep_list, temperature = temperature, c_puct = c_puct, epsilon = epsilon, alpha = alpha)
        go_env.render(mode="terminal", wait_for_input = False)
        turn_counter += 1
        print_turn(turn_counter)
        #time.sleep(0.5)

        #valid_moves = go_env.valid_moves()

        #print(f"legal moves: {valid_moves} len: {len(valid_moves)}")

        if go_env.game_ended() or turn_counter > GAME_TURN_LIMIT:
            break
        
        state, reward, done, info, white_root, timestep_list, white_action = play_agent(go_env, white_player, white_root, black_action, timestep_list, temperature = temperature, c_puct = c_puct, epsilon = epsilon, alpha = alpha)#play_player_input(go_env)
        go_env.render(mode="terminal", wait_for_input = False)
        turn_counter += 1
        print_turn(turn_counter)
        #time.sleep(0.5)
        x = 1

        if go_env.game_ended() or turn_counter > GAME_TURN_LIMIT:
            break

    if go_env.game_ended() == False:
        if turn_counter <= GAME_TURN_LIMIT:
            raise ValueError(f"game hasnt ended and turns only got to {turn_counter} but we broke out of the game loop on game number {game_number}!")
        print(f"ending game {game_number} because it exceeds {GAME_TURN_LIMIT} turns")

    last_player = go_env.unwrapped.turn() #0b 1w
    winner = go_env.winning() #winner from BLACKS perspective
    #winner = 0 if the game isnt ended or if black score - (white score + komi) == 0 
    #winner = 1 if black score - (white score + komi) > 0
    #winner = -1 if black score - (white score + komi) < 0
    #The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

    black_delta = 0
    white_delta = 0

    if winner == 1:
        black_delta = 1
        black_wins += 1
    if winner == -1:
        white_delta = 1
        white_wins += 1
    total_games += 1

    if black_player == player_A:
        player_A_wins += black_delta
        player_B_wins += white_delta

        player_A_side = "w"
        player_B_side = "b"

        black_player = player_B
        white_player = player_A
    else:
        player_A_wins += white_delta
        player_B_wins += black_delta

        player_A_side = "b"
        player_B_side = "w"

        black_player = player_A
        white_player = player_B

    corrected_winner = winner

    #flip winner if last_player == 1 (white)
    if last_player == 1:
        corrected_winner *= -1

    outcome = corrected_winner#this is 1 for the winning side, -1 for the losing side, and 0 for ties

    for timestep in reversed(timestep_list):
        timestep[TIMESTEP_OUTCOME_INDEX] = outcome
        outcome = outcome * -1

    for timestep in timestep_list:
        experiences.append(timestep)

    training_start = perf_counter()

    sampled_experiences = sample_experiences(experiences, max_experiences_per_training_run)
    print(f"now training with {len(sampled_experiences)} timesteps!")

    train_model(test_agent, optimizer, sampled_experiences)

    training_end = (perf_counter() - training_start) * 1000

    print(f"training took {training_end} ms")

    go_env.reset()
    turn_counter = 0


while not done:
        
    state, reward, done, info, black_root, timestep_list, black_action = play_player_input(go_env, black_player, black_root, white_action, timestep_list)
    go_env.render(mode="terminal", wait_for_input = False)
    turn_counter += 1
    print(f"TURN NUMBER: {turn_counter} (black {black_wins} | white {white_wins} | total {total_games})")
    #time.sleep(0.5)

    #valid_moves = go_env.valid_moves()

    #print(f"legal moves: {valid_moves} len: {len(valid_moves)}")

    if go_env.game_ended() or turn_counter > GAME_TURN_LIMIT:
        break
    
    state, reward, done, info, white_root, timestep_list, white_action = play_agent(go_env, white_player, white_root, black_action, timestep_list)#play_player_input(go_env)
    go_env.render(mode="terminal", wait_for_input = False)
    turn_counter += 1
    print(f"TURN NUMBER: {turn_counter} (black {black_wins} | white {white_wins} | total {total_games})")
    #time.sleep(0.5)
    x = 1

    if go_env.game_ended() or turn_counter > GAME_TURN_LIMIT:
        break