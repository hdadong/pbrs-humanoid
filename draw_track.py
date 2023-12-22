import numpy as np
import os
from matplotlib import pylab # type: ignore
import matplotlib.pyplot as plt

def draw_picture(
    timestep: int,
    num_episode: int,
    pred_state: np.ndarray,
    true_state: np.ndarray,
    save_replay_path: str = './',
    name: str = 'reward',
) -> None:
    """Draw a curve of the predicted value and the ground true value.

    Args:
        timestep (int): current step.
        num_episode (int): number of episodes.
        pred_state (list): predicted state.
        true_state (list): true state.
        save_replay_path (str): save replay path.
        name (str): name of the curve.
    """
    target1 = list(pred_state)
    target2 = list(true_state)
    input1 = np.arange(0, np.array(pred_state).shape[0], 1)
    input2 = np.arange(0, np.array(pred_state).shape[0], 1)

    pylab.plot(input1, target1, 'r-', label='true_tracking')
    pylab.plot(input2, target2, 'b-', label='true_tracked')
    #input_min = min(np.min(pred_state),np.min(true_state))
    #input_max = max(np.max(pred_state),np.max(true_state))

    pylab.xlabel('Step')
    pylab.ylabel(name)
    pylab.xticks(np.arange(0, np.array(pred_state).shape[0], 50))  # Set the axis numbers
    if name == 'reward':
        pylab.yticks(np.arange(0, 3, 0.2))
    else:
        pylab.yticks(np.arange(-0.34, 0.46, 0.1))
        pylab.yticks(np.arange(-3.14, 3.14, 0.4))

    pylab.legend(
        loc=3,
        borderaxespad=2.0,
        bbox_to_anchor=(0.7, 0.7),
    )  # Sets the position of that box for what each line is
    pylab.grid()  # draw grid
    pylab.savefig(
        os.path.join(
            save_replay_path,
            str(name) + str(timestep) + '_' + str(num_episode) + '.png',
        ),
        dpi=200,
    )  # save as picture
    pylab.close()
    
true_position = np.load('/home/bigeast/pbrs-humanoid/true_position.npy')
leg_position = np.load('/home/bigeast/pbrs-humanoid/true_position_tracking.npy')
for i in range(0, leg_position.shape[1]):
    if i in [4, 9]:
        t = 2
    else:
        t = i
    pred_state_idx = leg_position[:100,i]
    true_state_idx = true_position[50:150,t]
    draw_picture(
                    timestep=0,
                    num_episode=0,
                    pred_state=pred_state_idx,
                    true_state=true_state_idx,
                    save_replay_path='./',
                    name='obs_'+str(i),
                )
