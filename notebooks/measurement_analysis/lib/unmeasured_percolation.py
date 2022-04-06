import numpy as np
def fill_out_block(block, verbose=False):
    """
    This function fills out all the unmeasured sites of a 2x2 block. Scan over the whole trajectory to fill out a trajectory
    Args:
        block: The 2x2 block of the matrix to be filled out. 
    Returns:
        block: The filled out 2x2 block
    """
    if np.count_nonzero(block) == 4:
        # if the whole block is full of measurements
        if verbose:
            print('full')
        return block
    elif np.allclose(block[0, :], np.array([1, 1])) or np.allclose(block[0, :], -np.array([1, 1])):
        # if the top row is [1, 1] or [-1, -1], make the bottom row the same
        if verbose:
            print('top->bottom')
        block[1, :] = block[0, :]
        return block
    elif np.allclose(block[1, :], np.array([1, 1])) or np.allclose(block[1, :], -np.array([1, 1])):
        # if the bottom row is [1, 1] or [-1, -1], make the top row the same
        if verbose:
            print('bottom->top')
        block[0, :] = block[1, :]
        return block
    elif np.abs(block[0, 0])==1 and block[0, 0] == -block[1, 1]:
        # if the main diagonal is [1, -1], or [-1, 1], make the off diagonal [-1, 1] or [1, -1] resp.
        if verbose:
            print('main -> off')
        block[0, 1] = -block[0, 0]
        block[1, 0] = -block[1, 1]
        return block
    elif np.abs(block[0, 1])==1 and block[0, 1] == -block[1, 0]:
        # if the off diagonal is [1, -1], or [-1, 1], make the main diagonal [-1, 1] or [1, -1] resp.
        if verbose:
            print('off -> main')
        block[0, 0] = -block[0, 1]
        block[1, 1] = -block[1, 0]
        return block
    elif np.abs(block[0, 0])==1 and block[0, 0] == block[1, 1]:
        # if the main diagonal is [1, 1] or [-1, -1], and either the top or bottom of the off diagonal is set, set the bottom or top of the off diagonal, resp.
        if verbose:
            print('main + 1 -> off')
        if np.abs(block[0, 1]) == 1:
            block[1, 0] = block[0, 1]
            return block
        elif np.abs(block[1, 0]) == 1:
            block[0, 1] = block[1, 0]
            return block
        else:
            return block
    elif np.abs(block[0, 1])==1 and block[0, 1] == block[1, 0]:
        if verbose:
            print('off + 1 -> main')
        # if the off diagonal is [1, 1] or [-1, -1], and either the top or bottom of the main diagonal is set, set the bottom or top of the main diagonal, resp.
        if np.abs(block[0, 0]) == 1:
            block[1, 1] = block[0, 0]
            return block
        elif np.abs(block[1, 1]) == 1:
            block[0, 0] = block[1, 1]
            return block
        else:
            return block
    elif np.allclose(block[:, 0], np.array([1, -1])) or np.allclose(block[:, 0], np.array([-1, 1])):
        if verbose:
            print('left -> right, opp')
        # if the left column is [1, -1], or [-1, 1], set the right column to the opposite. 
        block[:, 1] = -block[:, 0]
        return block
    elif np.allclose(block[:, 1], np.array([1, -1])) or np.allclose(block[:, 1], np.array([-1, 1])):
        # if the right column is [1, -1], or [-1, 1], set the left column to the opposite. 
        if verbose:
            print('right -> left, opp')
        block[:, 0] = -block[:, 1]
        return block
    else:
        if verbose:
            print('nothing')
        return block

def fill_out(traj, verbose=False):
    """
    Run through every block in traj, and fill out unmeasured sharp sites
    """
    for i in range(traj.shape[0]-1):
        # each row
        for j in range(i%2, traj.shape[1]-1, 2):
            # each pair of columns, odd even on odd rows, even odd on even rows
            traj[i:i+2, j:j+2] = fill_out_block(traj[i:i+2, j:j+2], verbose)
            if verbose:
                print(i, j)
            #if i == 1:
            #    return traj

    return traj

def iterated_fill_out(traj, verbose=False):
    """
    Keep filling out unmeasured sites until a frame stops changing. 
    """
    i = 0
    new_traj = fill_out(np.copy(traj))
    while not np.allclose(traj, new_traj):
        traj = new_traj
        new_traj = fill_out(np.copy(traj), verbose)
        i = i+1
        if verbose:
            print(np.linalg.norm(traj-new_traj))
    return traj
