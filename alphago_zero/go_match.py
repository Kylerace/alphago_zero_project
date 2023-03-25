import argparse
from tree_search import *
from go_wrapper import *

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

class GameOverState(Enum):
    won = 1,
    lost = 2,
    tie = 3
    other = 4

class Player:
    def __init__(self):
        pass

    def start_game(self, go_env: GoEnv, player):
        pass

    def end_game(self, result: GameOverState):
        pass

    def play_move(self, go_env: GoEnv):
        pass

#this isnt used for training, just match compares
class HumanPlayer(Player):

    def play_move(self, go_env: GoWrapper):
        state, reward, done, info = None, None, None, None
        while True:
            action = go_env.render(mode="human") #grabbed from the on_mouse_press() pyglet window event, not sanitized
            try:
                state, reward, done, info = go_env.step(action)
            except:
                continue
            break
        return state, reward, done, info
    
    def start_game(self, _, player):
        if player == 1:
            print("starting game as black")
        elif player == 0:
            print("starting game as white")

    def end_game(self, result: GameOverState):
        if result == GameOverState.won:
            print("you won the game!")
        elif result == GameOverState.lost:
            print("you lost the game!")
        elif result == GameOverState.tie:
            print("it was a tie!")
        elif result == GameOverState.other:
            print("something weird happened")

class ModelPlayer(Player):
    model = None

    def __init__(self, model):
        self.model = model
        super().__init__()

    def start_game(self, go_env: GoWrapper, player):
        pass

    def end_game(self, result: GameOverState):
        pass

    def play_move(self, go_env: GoWrapper):
        pass

class RandomPlayer(Player):
    def play_move(go_env: GoWrapper):
        action = go_env.uniform_random_action()
        return go_env.step(action)

class MatchArgs:
    boardsize = 7
    komi=0
    players = None
    time_controls = None

    def __init__(self, boardsize=7, komi=0, players = (HumanPlayer(), RandomPlayer()), time_controls = (None, None)):
        self.boardsize = boardsize
        self.komi = komi
        self.players = players
        self.time_controls = time_controls

def match(args:MatchArgs = MatchArgs()):

    # Initialize environment
    go_env: GoEnv = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi)
    go_env = GoWrapper(go_env)

    go_env.reset()

    can_ever_render = True

    black_player:Player = args.players[0]
    white_player:Player = args.players[1]

    black_player.start_game(go_env, 1)
    white_player.start_game(go_env, 0)

    # Game loop
    done = False
    while not done:
        state, reward, done, info = black_player.play_move(go_env)

        if go_env.game_ended():
            break

        state, reward, done, info = white_player.play_move(go_env)

    black_won = go_env.winner()

    if black_won == 0: #invalid
        black_player.end_game(GameOverState.other)
        white_player.end_game(GameOverState.other)
    elif black_won == 1:
        black_player.end_game(GameOverState.won)
        white_player.end_game(GameOverState.lost)
    elif black_won == -1:
        black_player.end_game(GameOverState.lost)
        white_player.end_game(GameOverState.won)

    return black_won

class TournamentArgs:
    match_args:MatchArgs
    best_of = 5

    def __init__(self, match_args = MatchArgs(), best_of = 5):
        self.match_args = match_args
        self.best_of = best_of


def tournament(args: TournamentArgs()):
    pass

if __name__ == "__main__":
    import model_save
    second_player = RandomPlayer()
    boardsize = 7

    current_alpha = model_save.get_current_alpha()
    if current_alpha != None:
        second_player = current_alpha
        boardsize = model_save.alpha_board_size()

    match(MatchArgs(boardsize = boardsize, komi = 0, players = (HumanPlayer(), second_player), time_controls = (None, None)))