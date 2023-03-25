from tree_search import *
from bespoke_go import *
from play_functions import *

import gym
import gym_go
from gym_go.envs import GoEnv

def self_play(go_env: GoEnv, alpha_model: GoModel, episodes = 100, turn_limit = 300, turns_before_low_temp = 30, high_temperature = 1, low_temperature = 0.05, c_puct = 0.3, alpha = 0.1, epsilon = 0.75, print_game = True, print_stats = True, ):
    experiences = []
    black_player = alpha_model
    white_player = alpha_model

    black_wins = 0
    white_wins = 0

    total_games = 0
    # Game loop
    for game_number in range(1, episodes + 1):

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
            print(f"TURN NUMBER: {turn_counter}, (b {black_wins}|w {white_wins}|total {total_games})")

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

            if go_env.game_ended() or turn_counter > turn_limit:
                break
            
            state, reward, done, info, white_root, timestep_list, white_action = play_agent(go_env, white_player, white_root, black_action, timestep_list, temperature = temperature, c_puct = c_puct, epsilon = epsilon, alpha = alpha)#play_player_input(go_env)
            go_env.render(mode="terminal", wait_for_input = False)
            turn_counter += 1
            print_turn(turn_counter)
            #time.sleep(0.5)
            x = 1

            if go_env.game_ended() or turn_counter > turn_limit:
                break

        if go_env.game_ended() == False:
            if turn_counter <= turn_limit:
                raise ValueError(f"game hasnt ended and turns only got to {turn_counter} but we broke out of the game loop on game number {game_number}!")
            print(f"ending game {game_number} because it exceeds {turn_limit} turns")

        last_player = go_env.unwrapped.turn() #0b 1w
        winner = go_env.winning() #winner from BLACKS perspective
        #winner = 0 if the game isnt ended or if black score - (white score + komi) == 0 
        #winner = 1 if black score - (white score + komi) > 0
        #winner = -1 if black score - (white score + komi) < 0
        #The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.

        if winner == 1:
            black_wins += 1
        if winner == -1:
            white_wins += 1
        total_games += 1

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
            
    return experiences