"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO


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
