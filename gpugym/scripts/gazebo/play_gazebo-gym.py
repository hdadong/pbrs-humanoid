# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch

from gpugym.scripts.gazebo.simple_env import SimpleTask

# from gpugym.scripts.gazebo.task_regi import get_args, export_policy, export_critic

# from gpugym.scripts.gazebo.config import HumanoidCfg, HumanoidCfgPPO

# make global task registry
#task_registry = TaskRegistry()

#task_registry.register("pbrs:humanoid_bruce", SimpleTask, HumanoidCfg(), HumanoidCfgPPO())




def play(args):

    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False #True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.init_state.reset_ratio = 0.8

    # prepare environment
    env_gym, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # prepare environment
    env_gazebo = SimpleTask(env_cfg)


    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env_gym, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env_gym.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env_gym.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env_gym.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    play_log = []
    env_gym.max_episode_length = 50
    
    
    phase = torch.zeros(
        1, 1, dtype=torch.float,
        device='cuda:0', requires_grad=False)
    dt = env_cfg.control.decimation * env_cfg.sim.dt
    
    env_gym.reset_idx(torch.tensor([0], device='cuda:0'))
    env_gym.initial_step()
    obs_gym = env_gym.get_observations()
    obs_gazebo = env_gazebo.get_observations()
    for i in range(1000000000):       
        # obs = torch.zeros(
        #     1, 38, dtype=torch.float,
        #     device='cuda:0', requires_grad=False)
        # obs[:,0] = xxx                               # [1] Base height
        # obs[:,1:4] = xxx                             # [3] Base linear velocity
        # obs[:,4:7] = xxx                             # [3] Base angular velocity
        # obs[:,7:10] = xxx                            # [3] Projected gravity
        #print('obs[:,7:10]', obs[:,7:10])
        
        obs_gym[:,10:13] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float,device='cuda:0', requires_grad=False)  # [3] Velocity commands
        #obs[:,13] = smooth_sqr_wave(phase)             # [1] Contact schedule
        #obs[:,14] = torch.sin(2*torch.pi*phase)        # [1] Phase variable
        #obs[:,15] = torch.cos(2*torch.pi*phase)        # [1] Phase variable
        # obs[:,15:25] = xxx                           # [10] Joint states
        # obs[:,25:35] = xxx                           # [10] Joint velocities
        # obs[:,35:37] = xxx                           # [2] Contact states 
        actions_gym = policy(obs_gym.detach())
        actions_gazebo = policy(obs_gazebo.detach())
        

        # zero_action = torch.zeros_like(actions_gym)
        # zero_action[0][0] = 1.25
        # zero_action[0][1] = 0.7
        # zero_action[0][2] = 0.5
        # zero_action[0][3] = -0.2
        # zero_action[0][4] = 0.1
        # zero_action[0][5] = 1.25
        # zero_action[0][6] = 0.7
        # zero_action[0][7] = 0.5
        # zero_action[0][8] = -0.2
        # zero_action[0][9] = 0.1
        obs_gym, _, rews, dones, infos   = env_gym.step(actions_gym.detach())
        if (i>100): 
            obs_gazebo  = env_gazebo.step(actions_gazebo.detach(), obs_gym)
        else:  
            env_gazebo.gs.simulator.step_simulation()
            
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env_gym.gym.write_viewer_image_to_file(env_gym.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env_gym.dt
            env_gym.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            play_log.append(
                [i*env_gym.dt]
                + obs_gym[robot_index, :].cpu().numpy().tolist()
                + actions_gym[robot_index, :].detach().cpu().numpy().tolist()
                + env_gym.root_states[0, :].detach().cpu().numpy().tolist()
                + env_gym.contact_forces[robot_index, env_gym.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env_gym.contact_forces[robot_index, env_gym.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env_gym.torques[robot_index, :].detach().cpu().numpy().tolist()
            )
        elif i==stop_state_log:
            np.savetxt('./play_log.csv', play_log, delimiter=',')
            # logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env_gym.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()


#  python gpugym/scripts/gazebo/play_gazebo-gym.py --task=pbrs:humanoid_bruce_stand
#  python gpugym/scripts/gazebo/play_gazebo-gym.py --task=pbrs:humanoid_bruce_stand --load_run Dec22_09-08-23_ICRA2023 --checkpoint 2000
# python gpugym/scripts/gazebo/play_gazebo-gym.py --task=pbrs:humanoid_bruce_stand --load_run Dec21_23-28-54_ICRA2023 

# python gpugym/scripts/gazebo/play_gazebo-gym.py --task=pbrs:humanoid_bruce_stand --load_run   Dec25_22-38-35_ICRA2023 --checkpoint 42550 42400-42550 good

#Loading model from: /home/bigeast/pbrs-humanoid/logs/PBRS_HumanoidLocomotion/Dec25_22-38-35_ICRA2023/model_42400.pt
if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
