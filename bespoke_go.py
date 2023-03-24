import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

#NEW BESPOKE GO STUFF BEGIN

TURN_BLACK = 0 # old 0
TURN_WHITE = 1 # old 1

CHANNEL_BLACK = 0
CHANNEL_WHITE = 1

TURN_CHANNEL = -1
NUM_CHANNELS = 17

def init_state2(size):

    return np.zeros((NUM_CHANNELS, size, size))

def turn2(state):
    """
    :param state:
    :return: Whose turn it is (TURN_BLACK/TURN_WHITE)
    """
    return int(np.max(state[TURN_CHANNEL]))

def prev_player_passed2(state):
    #if black has moved yet (first plane isnt all zero), and the non moving players latest board is identical to their t-1 board,
    # then the previous player has passed 
    shape = state.shape[1:]
    player_is_white = turn2(state)
    cond1 = np.array_equal(state[0], np.zeros(shape, dtype=np.float32))
    cond2 = np.array_equal(
        state[
            player_is_white # 0b, 1w
        ],
        state[
            player_is_white + 3 # 3b, 4w
        ]
    )
    return cond1 and cond2

    #return np.max(state[govars.PASS_CHNL] == 1) == 1

group_struct = np.array([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]])

surround_struct = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])

neighbor_deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

def compute_valid_moves2(state, player, ko_protect=None):
    """
    Updates valid moves in the OPPONENT's perspective
    also only output moves on the board. passing is not checked or outputted
    1.) Opponent cannot move at a location
        i.) If it's occupied
        i.) If it's protected by ko
    2.) Opponent can move at a location
        i.) If they can kill
    3.) Opponent cannot move at a location
        i.) If it's adjacent to one of their groups with only one liberty and
            not adjacent to other groups with more than one liberty and is completely surrounded
        ii.) If it's surrounded by opponent pieces and all of those corresponding groups
            move more than one liberty
    """

    # All pieces and empty spaces
    all_pieces = np.sum(state[[CHANNEL_BLACK, CHANNEL_WHITE]], axis=0)
    empties = 1 - all_pieces

    # Setup invalid and valid arrays
    possible_invalid_array = np.zeros(state.shape[1:])
    definite_valids_array = np.zeros(state.shape[1:])

    # Get all groups
    all_own_groups, num_own_groups = measurements.label(state[player])
    all_opp_groups, num_opp_groups = measurements.label(state[1 - player])
    expanded_own_groups = np.zeros((num_own_groups, *state.shape[1:])) #(x, 7, 7) -> num_own_groups planes of shape (7,7) holding 0's
    expanded_opp_groups = np.zeros((num_opp_groups, *state.shape[1:]))

    # Expand the groups such that each group is in its own channel
    for i in range(num_own_groups):
        expanded_own_groups[i] = all_own_groups == (i + 1)

    for i in range(num_opp_groups):
        expanded_opp_groups[i] = all_opp_groups == (i + 1)

    # Get all liberties in the expanded form
    all_own_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_own_groups, surround_struct[np.newaxis])
    all_opp_liberties = empties[np.newaxis] * ndimage.binary_dilation(expanded_opp_groups, surround_struct[np.newaxis])

    own_liberty_counts = np.sum(all_own_liberties, axis=(1, 2))
    opp_liberty_counts = np.sum(all_opp_liberties, axis=(1, 2))

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    # Definite valids are on single liberties of own groups, multi-liberties of opponent groups
    # or you are not surrounded
    possible_invalid_array += np.sum(all_own_liberties[own_liberty_counts > 1], axis=0) #own_liberty_counts > 1
    possible_invalid_array += np.sum(all_opp_liberties[opp_liberty_counts == 1], axis=0) #opp_liberty_counts == 1

    definite_valids_array += np.sum(all_own_liberties[own_liberty_counts == 1], axis=0) #own_liberty_counts == 1
    definite_valids_array += np.sum(all_opp_liberties[opp_liberty_counts > 1], axis=0) #opp_liberty_counts > 1

    # All invalid moves are occupied spaces + (possible invalids minus the definite valids and it's surrounded)
    surrounded = ndimage.convolve(all_pieces, surround_struct, mode='constant', cval=1) == 4
    invalid_moves = all_pieces + possible_invalid_array * (definite_valids_array == 0) * surrounded

    # Ko-protection
    if ko_protect is not None:
        invalid_moves[ko_protect[0], ko_protect[1]] = 1
    normalized_invalid_moves = invalid_moves > 0
    valid_moves = (1 - normalized_invalid_moves).flatten()
    return valid_moves

def adj_data2(state, action2d, player):
    neighbors = neighbor_deltas + action2d
    valid = (neighbors >= 0) & (neighbors < state.shape[1])
    valid = np.prod(valid, axis=1)
    neighbors = neighbors[np.nonzero(valid)] #2d indices of every adjacent spot to the move (that isnt out of bounds)

    opp_pieces = state[1 - player] #every enemy piece
    surrounded = (opp_pieces[neighbors[:, 0], neighbors[:, 1]] > 0).all() #check if theres an opponent piece on every neighbor

    return neighbors, surrounded

def update_pieces2(state, adj_locs, player):
    opponent = 1 - player
    killed_groups = []

    all_pieces = np.sum(state[[TURN_BLACK, TURN_WHITE]], axis=0)
    empties = 1 - all_pieces

    all_opp_groups, _ = ndimage.measurements.label(state[opponent])

    # Go through opponent groups
    all_adj_labels = all_opp_groups[adj_locs[:, 0], adj_locs[:, 1]]
    all_adj_labels = np.unique(all_adj_labels)
    for opp_group_idx in all_adj_labels[np.nonzero(all_adj_labels)]:
        opp_group = all_opp_groups == opp_group_idx
        liberties = empties * ndimage.binary_dilation(opp_group)
        if np.sum(liberties) <= 0:
            # Killed group
            opp_group_locs = np.argwhere(opp_group)
            state[opponent, opp_group_locs[:, 0], opp_group_locs[:, 1]] = 0
            killed_groups.append(opp_group_locs)

    return killed_groups

def set_turn2(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[TURN_CHANNEL] = 1 - state[TURN_CHANNEL]

#old state = [b, w, ...]
#old shape = ()
#returns the new state, whether the game is completed, 
def next_state2(state, action1d, valid_moves):
    state = np.copy(state)

    board_shape = state.shape[1:] #(size, size)
    pass_idx = np.prod(board_shape) # size**2
    passed = action1d == pass_idx
    action2d = action1d // board_shape[0], action1d % board_shape[1]

    player = turn2(state) #0b, 1w
    previously_passed = prev_player_passed2(state)
    ko_protect = None

    is_done_now = False

    if passed:
        old_state = state
        after_insert = np.insert(state, 0, state[0:2], axis=0)
        after_delete = np.delete(after_insert, (16, 17), axis = 0)
        state = after_delete
        
        if previously_passed:
            is_done_now = True
    else:
        # Assert move is valid
        assert valid_moves[action1d] == 1, ("In move", action1d, action2d)

        #invalid_moves = compute_invalid_moves2(state, player, ko_protect)
        #if invalid_moves[action2d[0], action2d[1]] == True:
        #    print(f"next_state2() thinks {action1d}->{action2d} is invalid")
        #    new_invalid_moves = compute_invalid_moves2(state, player, ko_protect)

        old_state = state
        after_insert = np.insert(state, 0, state[0:2], axis=0)
        after_delete = np.delete(after_insert, (16, 17), axis = 0)
        state = after_delete

        #add the piece
        state[player, action2d[0], action2d[1]] = 1

        # Get adjacent location and check whether the piece will be surrounded by opponent's piece
        adj_locs, surrounded = adj_data2(state, action2d, player)

        # Update pieces
        killed_groups = update_pieces2(state, adj_locs, player)

        # If only killed one group, and that one group was one piece, and piece set is surrounded,
        # activate ko protection
        if len(killed_groups) == 1 and surrounded:
            killed_group = killed_groups[0]
            if len(killed_group) == 1:
                ko_protect = killed_group[0]

    new_valid_moves = compute_valid_moves2(state, player, ko_protect)

    old_turn = np.copy(state)[TURN_CHANNEL]

    set_turn2(state)

    assert not np.array_equal(state[TURN_CHANNEL], old_turn)

    return state, is_done_now, new_valid_moves
        

def areas2(state):
    '''
    Return black area, white area
    '''

    all_pieces = np.sum(state[[CHANNEL_BLACK, CHANNEL_WHITE]], axis=0)
    empties = 1 - all_pieces

    empty_labels, num_empty_areas = ndimage.measurements.label(empties)

    black_area, white_area = np.sum(state[CHANNEL_BLACK]), np.sum(state[CHANNEL_WHITE])
    for label in range(1, num_empty_areas + 1):
        empty_area = empty_labels == label
        neighbors = ndimage.binary_dilation(empty_area)
        black_claim = False
        white_claim = False
        if (state[CHANNEL_BLACK] * neighbors > 0).any():
            black_claim = True
        if (state[CHANNEL_WHITE] * neighbors > 0).any():
            white_claim = True
        if black_claim and not white_claim:
            black_area += np.sum(empty_area)
        elif white_claim and not black_claim:
            white_area += np.sum(empty_area)

    return black_area, white_area

def str2(state):
    board_str = ''

    size = state.shape[1]
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
