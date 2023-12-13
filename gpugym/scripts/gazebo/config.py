"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

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

import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)
                
class FixedRobotCfg(BaseConfig):
    class env:
        num_envs = 1096
        num_observations = 7
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 1
        env_spacing = 4.  # not used with heightfields/trimeshes
        root_height = 2.
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain:
        mesh_type = 'none'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    # class commands:
    #     curriculum = False
    #     max_curriculum = 1.
    #     num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    #     resampling_time = 10. # time before command are changed[s]
    #     heading_command = True # if true: compute ang vel command from heading error
    #     class ranges:
    #         lin_vel_x = [-1.0, 1.0]  # min max [m/s]
    #         lin_vel_y = [-1.0, 1.0]   # min max [m/s]
    #         ang_vel_yaw = [-1, 1]    # min max [rad/s]
    #         heading = [-3.14, 3.14]

    class init_state:

        reset_mode = "reset_to_basic" 
        # default setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below

        # * target state when action = 0, also reset positions for basic mode
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}



    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0}  # [N*m/rad]
        damping = {'joint_a': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        actuated_joints_mask = []  # for each dof: 1 if actuated, 0 if passive
        # Empty implies no chance in the _compute_torques step

    class asset:
        file = ""
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        disable_actions = False
        disable_motors = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True # fix the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales: # reward coefficients, weightings of reward
            termination = -0.0
            torques = -0.00001
            dof_vel = -0.
            collision = -1.
            action_rate = -0.01
            dof_pos_limits = -1.

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off

    class normalization:
        clip_observations = 1000.
        clip_actions = 1000.
        class obs_scales:
            dof_pos = 1.
            dof_vel = 1.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            noise = 0.1  # implement as needed, also in your robot class

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class FixedRobotCfgPPO(BaseConfig):
    seed = 2
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'fixed_robot'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # (n_robots in Rudin 2021 paper - batch_size = n_steps * n_robots)
        num_observations = 235
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:

        # * target state when action = 0, also reset positions for basic mode
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        reset_mode = "reset_to_basic" 
        # reset setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below

        # * root defaults
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.5, 0.75],  # z
                          [0., 0.],  # roll
                          [0., 0.],  # pitch
                          [0., 0.]]  # yaw
        root_vel_range = [[-0.1, 0.1],  # x
                          [-0.1, 0.1],  # y
                          [-0.1, 0.1],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        exp_avg_decay = None

    class asset:
        file = ""
        keypoints = []
        end_effectors = []
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        disable_actions = False
        disable_motors = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = .0
            tracking_lin_vel = .0
            tracking_ang_vel = 0.
            lin_vel_z = 0
            ang_vel_xy = 0.
            orientation = 0.
            torques = 0.
            dof_vel = 0.
            base_height = 0.
            feet_air_time = 0.
            collision = 0.
            feet_stumble = 0.0
            action_rate = 0.
            action_rate2 = 0.
            stand_still = 0.
            dof_pos_limits = 0.
            
            feat_alternate_leading = 0.
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 2
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # adam optimizer options
        learning_rate = 1.e-4 # 5.e-4
        weight_decay = 0


    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'legged_robot'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

                
                
                
import torch



class HumanoidCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 38
        num_actions = 10
        episode_length_s = 5

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        ang_vel_command = True

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [0, 5.0]        # min max [m/s]
            lin_vel_y = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-0.0005, 0.0005]     # min max [rad/s]
            heading = [0., 0.]

            # PLAY COMMAND RANGES #
            # lin_vel_x = [3., 3.]    # min max [m/s]
            # lin_vel_y = [-0., 0.]     # min max [m/s]
            # ang_vel_yaw = [2, 2]      # min max [rad/s]
            # heading = [0, 0]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 0.48]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [0.48, 0.48],
            [-torch.pi/100, torch.pi/100],
            [-torch.pi/100, torch.pi/100],
            [-torch.pi/100, torch.pi/100]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.05, .05],
            [-.05, .05],
            [-.05, .05],
            [-.05, .05],
            [-.05, .05],
            [-.05, .05]
        ]

        # default_joint_angles = {
        #     'left_hip_yaw': 0.,
        #     'left_hip_abad': 0.,
        #     'left_hip_pitch': -0.2,
        #     'left_knee': 0.25,  # 0.6
        #     'left_ankle': 0.0,
        #     'right_hip_yaw': 0.,
        #     'right_hip_abad': 0.,
        #     'right_hip_pitch': -0.2,
        #     'right_knee': 0.25,  # 0.6
        #     'right_ankle': 0.0,
        # }
        default_joint_angles = {
            'hip_yaw_l': -2.6794896523796297e-08,
            'hip_yaw_r': -2.6794896523796297e-08,
            'hip_roll_l': -0.0022512623568848866,
            'hip_roll_r': 0.0022512623568848866,
            'hip_pitch_l': 0.0042626806266173,
            'hip_pitch_r': 0.0042626806266173,
            'knee_pitch_l': -0.0061344649420423,
            'knee_pitch_r': -0.0061344649420423,
            'ankle_pitch_l': 0.42187178431542494,
            'ankle_pitch_r': 0.42187178431542494,
            
        }

        # dof_pos_range = {
        #     'left_hip_yaw': [-0.1, 0.1],
        #     'left_hip_abad': [-0.2, 0.2],
        #     'left_hip_pitch': [-0.2, 0.2],
        #     'left_knee': [0.6, 0.7],
        #     'left_ankle': [-0.3, 0.3],
        #     'right_hip_yaw': [-0.1, 0.1],
        #     'right_hip_abad': [-0.2, 0.2],
        #     'right_hip_pitch': [-0.2, 0.2],
        #     'right_knee': [0.6, 0.7],
        #     'right_ankle': [-0.3, 0.3],
        # }
        # dof_pos_range = {
        #     'hip_yaw_l': [-0.2, 0.2],
        #     'hip_yaw_r': [-0.2, 0.2],
        #     'hip_roll_l': [-0.2, 0.2],
        #     'hip_roll_r': [-0.2, 0.2],
        #     'hip_pitch_l': [-1.0, 1.0],
        #     'hip_pitch_r': [-1.0, 1.0],
        #     'knee_pitch_l': [-1.0, 1.0],
        #     'knee_pitch_r': [-1.0, 1.0],
        #     'ankle_pitch_l': [-1.0, 1.0],
        #     'ankle_pitch_r': [-1.0, 1.0],
        # }
        dof_pos_range = {
            'hip_yaw_l': [-0.3, 0.3],
            'hip_yaw_r': [-0.3, 0.3],
            'hip_roll_l': [-0.15, 0.25],
            'hip_roll_r': [-0.25, 0.15],
            'hip_pitch_l': [0.00, 0.7],
            'hip_pitch_r': [0.00, 0.7],
            'knee_pitch_l': [-1.0, 0.25],
            'knee_pitch_r': [-1.0, 0.25],
            'ankle_pitch_l': [-1.0, 1.0],
            'ankle_pitch_r': [-1.0, 1.0],
        }
        # dof_vel_range = {
        #     'left_hip_yaw': [-0.1, 0.1],
        #     'left_hip_abad': [-0.1, 0.1],
        #     'left_hip_pitch': [-0.1, 0.1],
        #     'left_knee': [-0.1, 0.1],
        #     'left_ankle': [-0.1, 0.1],
        #     'right_hip_yaw': [-0.1, 0.1],
        #     'right_hip_abad': [-0.1, 0.1],
        #     'right_hip_pitch': [-0.1, 0.1],
        #     'right_knee': [-0.1, 0.1],
        #     'right_ankle': [-0.1, 0.1],
        # }
        dof_vel_range = {
            'hip_yaw_l': [-2.00, 3.00],
            'hip_yaw_r': [-3.00, 2.00],
            'hip_roll_l': [-1.30, 1.30],
            'hip_roll_r': [-1.30, 1.30],
            'hip_pitch_l': [-4.50, 4.00],
            'hip_pitch_r': [-4.50, 4.00],
            'knee_pitch_l': [-9.00, 9.00],
            'knee_pitch_r': [-9.00, 9.00],
            'ankle_pitch_l': [-50.00, 30.00],
            'ankle_pitch_r': [-50.00, 30.00],
        }
    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        # stiffness = {
        #     'left_hip_yaw': 30.,
        #     'left_hip_abad': 30.,
        #     'left_hip_pitch': 30.,
        #     'left_knee': 30.,
        #     'left_ankle': 30.,
        #     'right_hip_yaw': 30.,
        #     'right_hip_abad': 30.,
        #     'right_hip_pitch': 30.,
        #     'right_knee': 30.,
        #     'right_ankle': 30.,
        # }
        stiffness = {
            'hip_yaw_l': 1.,
            'hip_yaw_r': 1.,
            'hip_roll_l': 1.,
            'hip_roll_r': 1.,
            'hip_pitch_l': 2.,
            'hip_pitch_r': 2.,
            'knee_pitch_l': 2.,
            'knee_pitch_r': 2.,
            'ankle_pitch_l': 0.2,
            'ankle_pitch_r': 0.2,
        }
        damping = {
            'hip_yaw_l': 0.01,
            'hip_yaw_r': 0.01,
            'hip_roll_l': 0.01,
            'hip_roll_r': 0.01,
            'hip_pitch_l': 0.02,
            'hip_pitch_r': 0.02,
            'knee_pitch_l': 0.02,
            'knee_pitch_r': 0.02,
            'ankle_pitch_l': 0.003,
            'ankle_pitch_r':0.003,
        }

        action_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.05

    class asset(LeggedRobotCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}'\
        #     '/resources/robots/mit_humanoid/mit_humanoid_fixed_arms.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}'\
            '/resources/robots/bruce/bruce.urdf'
        keypoints = ["base_link"]
        end_effectors = ['ankle_pitch_link_l', 'ankle_pitch_link_r']
        foot_name = 'ankle_pitch'
        # terminate_after_contacts_on = [
        #     'base',
        #     'left_upper_leg',
        #     'left_lower_leg',
        #     'right_upper_leg',
        #     'right_lower_leg',
        #     'left_upper_arm',
        #     'right_upper_arm',
        #     'left_lower_arm',
        #     'right_lower_arm',
        #     'left_hand',
        #     'right_hand',
        # ]
        terminate_after_contacts_on = [
            'base_link',
            'hip_yaw_link_l',
            'hip_roll_link_l',
            'hip_pitch_link_l',
            'knee_pitch_link_l',
            'hip_yaw_link_r',
            'hip_roll_link_r',
            'hip_pitch_link_r',
            'knee_pitch_link_r',
            'shoulder_pitch_link_l',
            'shoulder_roll_link_l',
            'elbow_pitch_link_l',
            'shoulder_pitch_link_r',
            'shoulder_roll_link_r',
            'elbow_pitch_link_r',
        ]
        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3  

    class rewards(LeggedRobotCfg.rewards):
        # ! "Incorrect" specification of height
        # base_height_target = 0.7
        base_height_target = 0.52
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5




        class scales(LeggedRobotCfg.rewards.scales):
            # * "True" rewards * #
            action_rate = -1.e-3
            action_rate2 = -1.e-4
            tracking_lin_vel = 10.
            tracking_ang_vel = 5.
            torques = -1e-4
            dof_pos_limits = -10
            torque_limits = -1e-2
            termination = -100

            # * Shaping rewards * #
            # Sweep values: [0.5, 2.5, 10, 25., 50.]
            # Default: 5.0
            # orientation = 5.0

            # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
            # Default: 2.0
            # base_height = 2.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            # joint_regularization = 1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            ori_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            baseHeight_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            jointReg_pb = 1.0

            feet_air_time = 10.0
            feat_alternate_leading = 10.0
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            base_z = 1./0.6565

        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05
            dof_pos = 0.005
            dof_vel = 0.001
            lin_vel = 0.00001
            ang_vel = 0.00005
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]

        class physx:
            max_depenetration_velocity = 10.0


class HumanoidCfgPPO(LeggedRobotCfgPPO):
    do_wandb = True
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24
        max_iterations = 100000
        run_name = 'ICRA2023'
        experiment_name = 'PBRS_HumanoidLocomotion'
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'
