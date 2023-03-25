from go_wrapper import *
import gym_go
from gym_go.envs import GoEnv
from tree_search import *
import numpy as np

def play_player_input(go_env: GoWrapper, timestep_list = []):
    state, reward, done, info = None, None, None, None
    while True:
        action = go_env.render(mode="human") #grabbed from the on_mouse_press() pyglet window event, not sanitized
        #try:
        old_state = np.copy(state)

        state, reward, done, info = go_env.step(action)
        if np.array_equal(state, old_state):
            print("nothing changed!")
            continue
        
        null_timestep = [None, None, None] #gets filtered out
        timestep_list.append(null_timestep)
        #except Exception as e:
        #    print(e)
        #    x = 1
        #    continue
        break
    return state, reward, done, info

def play_random(go_env: GoEnv):
    action = go_env.uniform_random_action()

    return go_env.step(action), action


def play_agent(go_env: GoEnv, model: GoModel, input_root_node = None, enemy_action = None, timestep_list = [], temperature = MCT_TEMPERATURE, c_puct = 0.7, epsilon = 0.75, alpha = 0.1):

    state = go_env.state_
    boardsize = state.shape[1]

    root_node = input_root_node
    if input_root_node == None:
        root_node = [state, 0, 0, 0, 1, 49, False, compute_valid_moves2(state, turn2(state), None), []]
    else:
        root_node = realign_to_opponents_move(root_node, enemy_action, boardsize)

    bespoke_valid_moves = root_node[MCT_VALID_MOVES_INDEX]
    #new_bespoke_valid_moves = compute_valid_moves2()
    real_valid_moves = np.delete(go_env.valid_moves(), -1)

    assert np.array_equal(bespoke_valid_moves, real_valid_moves)

    reward, done, info = None, None, None
    our_action = None
    while True:        
        action = None
        if False:
            logits, value = model(state, False)
            action = tf.reshape(tf.random.categorical(logits, 1), []).numpy()
            x = 1
        else:
            total_start = perf_counter()
            iterations_completed, nodes_created, nodes_selected, \
            nodes_PUCTed, nodes_backpropped, leaf_nodes_found, max_depth \
                = monte_carlo_tree_search(root_node, model, search_iterations = 10, c_puct = c_puct, epsilon = epsilon, alpha = alpha, print_perf = True)
            
            our_action, root_node, timestep = mcts_decide(root_node, root_node[MCT_VALID_MOVES_INDEX], -0.95, temperature = temperature)

            timestep_list.append(timestep)
            total_end = (perf_counter() - total_start) * 1000
            print(f"total move time: {total_end} ms")

        if go_env.valid_moves()[our_action] == 0:
            player_string = "black player" if go_env.turn() == 0 else "white player"
            print(f"{player_string} (model) tried to play {our_action} when thats not a legal move!")
            continue

        state, reward, done, info = go_env.step(our_action)

        break

    return state, reward, done, info, root_node, timestep_list, our_action