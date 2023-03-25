import gym
import numpy as np
import gym_go
from gym_go.envs import GoEnv
from bespoke_go import *

GO_OBSERVATION_PLANES = 17

class GoWrapper(gym.Wrapper):
    size = 7
    turn = 1
    color = 0 #0 = black, 1 = white
    state_ = None
    current_valid_moves = None

    def __init__(self, env: GoEnv):
        super().__init__(env)

        self.size = env.size
        size = self.size

        self.state_ = np.zeros((NUM_CHANNELS, size, size), dtype=np.float32)
        self.current_valid_moves = compute_valid_moves2(self.state_, turn2(self.state_), None)

    def step(self, action):
        #return np.copy(self.state_), self.reward(), self.done, self.info()
        old_env_state_ = np.copy(self.env.state_)
        observation, reward, done, info = self.env.step(action)
        next_color = info['turn']
        #BLACK = 0
        #WHITE = 1
        if next_color == self.color: #nothing moved
            return np.copy(self.state_), reward, done, info
    
        if next_color == 0:
            self.turn += 1

        self.color = next_color

        #           axis: 0, 1, 2
        #state_.shape = (17, 7, 7)
        #evalutes from inside -> out
        #insert 2 new state planes into indices 0 & 1 of axis 0, new shape is (19, 7, 7)
        #oldest black and oldest white moves move from indices 14 & 15 to 16 & 17
        #delete oldest black and oldest white last move planes from indices 16 & 17 of axis 0, new shape is (17, 7, 7)
        #set index 17 (old color) of axis 0 to new color
        old_state_ = self.state_

        after_insert = np.insert(
                            old_state_, #into
                            0, #index
                            observation[0:2], #values (0, 1) 
                            axis = 0
                        )

        after_delete = np.delete(
                        after_insert,
                        (16, 17),
                        axis = 0
                    )

        new_state_ = np.copy(after_delete)
        new_state_[-1] = np.ones((self.size,self.size)) * self.color

        new_action = action
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            assert 0 <= action[0] < self.size
            assert 0 <= action[1] < self.size
            new_action = self.size * action[0] + action[1]
        elif action is None:
            new_action = self.size ** 2

        test_state, test_done, test_new_current_valid_moves = next_state2(old_state_, new_action, self.current_valid_moves)
        self.current_valid_moves = test_new_current_valid_moves

        self.state_ = new_state_

        assert np.array_equal(test_state, new_state_)

        return new_state_, reward, done, info
    
    def reset(self):
        super().reset()
        self.state_ = np.zeros((NUM_CHANNELS, self.size, self.size))
        self.turn = 1
        self.color = 0
        self.current_valid_moves = compute_valid_moves2(self.state_, turn2(self.state_), None)