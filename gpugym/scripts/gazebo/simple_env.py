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

import sys
import numpy as np
import torch

import time
import Settings.BRUCE_data as RDS
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
from Settings.BRUCE_macros import *
from Library.BRUCE_GYM.GAZEBO_INTERFACE import Manager as gazint









#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "September 6, 2023"
__version__   = "0.0.3"
__status__    = "Production"

"""
Script for communication with Gazebo
"""

import time
import Settings.BRUCE_data as RDS
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
from Settings.BRUCE_macros import *
from Library.BRUCE_GYM.GAZEBO_INTERFACE import Manager as gazint

PI   = 3.1415926
PI_2 = PI / 2
PI_3 = PI / 3
PI_4 = PI / 4

class GazeboSimulator:
    def __init__(self):
        # robot info
        self.num_legs = 2
        self.num_joints_per_leg = 5
        self.num_arms = 2
        self.num_joints_per_arms = 3
        self.num_joints = self.num_legs * self.num_joints_per_leg + self.num_arms * self.num_joints_per_arms
        self.num_contact_sensors = 4
        # self.leg_p_gains = [265, 150,  80,  80,    30]
        # self.leg_i_gains = [  0,   0,   0,   0,     0]
        # self.leg_d_gains = [ 1., 2.3, 0.8, 0.8, 0.003]
        
        self.leg_p_gains = [265, 150,  80,  80,    250]
        self.leg_i_gains = [  0,   0,   0,   0,     0]
        self.leg_d_gains = [ 1., 2.3, 0.8,  1.0, 0.025]
        
        # self.leg_p_gains = [35, 35,  35,  35,    0.2]
        # self.leg_i_gains = [  0,   0,   0,   0,     0]
        # self.leg_d_gains = [ 1., 1,  1, 1, 0.003]

        self.arm_p_gains = [ 1.6,  1.6,  1.6]
        self.arm_i_gains = [   0,    0,    0]
        self.arm_d_gains = [0.03, 0.03, 0.03]

        self.p_gains = self.leg_p_gains * 2 + self.arm_p_gains * 2  # the joint order matches the robot's sdf file
        self.i_gains = self.leg_i_gains * 2 + self.arm_i_gains * 2
        self.d_gains = self.leg_d_gains * 2 + self.arm_d_gains * 2
        
        # simulator info
        self.simulator = None
        self.simulation_frequency = 1000  # Hz
        self.simulation_modes = {'torque': 0, 'position': 2}
        self.simulation_mode = self.simulation_modes['position']
        
        
    def initialize_simulator(self):
        self.simulator = gazint.GazeboInterface(robot_name='bruce', num_joints=self.num_joints, num_contact_sensors=self.num_contact_sensors)
        self.simulator.set_step_size(1. / self.simulation_frequency)
        self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_all_position_pid_gains(self.p_gains, self.i_gains, self.d_gains)

        # arm pose
        ar1, ar2, ar3 = -0.7,  1.3,  2.0
        al1, al2, al3 =  0.7, -1.3, -2.0

        # leg pose
        bpr = np.array([0.04, -0.07, -0.38])  # right foot position  in body frame
        bpl = np.array([0.04, +0.07, -0.38])  # left  foot position  in body frame
        bxr = np.array([1., 0., 0.])          # right foot direction in body frame
        bxl = np.array([1., 0., 0.])          # left  foot direction in body frame
        lr1, lr2, lr3, lr4, lr5 = kin.legIK_foot(bpr, bxr, +1.)
        ll1, ll2, ll3, ll4, ll5 = kin.legIK_foot(bpl, bxl, -1.)
        # self.initial_pose = [lr1+PI_2, lr2-PI_2, lr3, lr4, lr5,
        #                 ll1+PI_2, ll2-PI_2, ll3, ll4, ll5,
        #                 ar1, ar2, ar3,
        #                 al1, al2, al3]
        PI   = 3.1415926
        PI_2 = PI / 2

        self.initial_pose = [-1.5707963267948966+PI_2, 1.5861961113210856-PI_2, 0.5969977704117716, -0.9557276982676578, 0.3587299278558862,
                        -1.5707963267948966+PI_2, 1.5553965422687077-PI_2,  0.5969977704117716, -0.9557276982676578, 0.3587299278558862,
                        ar1, ar2, ar3,
                        al1, al2, al3]
        # self.initial_pose = [-1.57+PI_2, 1.577-PI_2, 0.59815, -1.021412, 0.412,
        #                 -1.564+PI_2, 1.55-PI_2,  0.59, -1.055, 0.433,
        #                 ar1, ar2, ar3,
        #                 al1, al2, al3]
        # self.initial_pose = [-1.5+PI_2, 1.5-PI_2, 0.57, -0.6, 0.0,
        #                 -1.5+PI_2, 1.5-PI_2,  0.65, -0.8, 0.0,
        #                 ar1, ar2, ar3,
        #                 al1, al2, al3]
        print("self.initial_pose", self.initial_pose)
        self.simulator.reset_simulation(initial_pose=self.initial_pose)
        
        print('Gazebo Initialization Completed!')


    def write_position(self, leg_positions, arm_positions):
        """
        Send goal positions to the simulator.
        """
        goal_position = [leg_positions[0], leg_positions[1], leg_positions[2], leg_positions[3], leg_positions[4],
                         leg_positions[5], leg_positions[6], leg_positions[7], leg_positions[8], leg_positions[9],
                         arm_positions[0], arm_positions[1], arm_positions[2],
                         arm_positions[3], arm_positions[4], arm_positions[5]]
        if self.simulation_mode != self.simulation_modes['position']:
            self.simulation_mode = self.simulation_modes['position']
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_position(goal_position)

    def write_torque(self, leg_torques, arm_torques):
        """
        Send goal torques to the simulator.
        """
        goal_torque = [leg_torques[0], leg_torques[1], leg_torques[2], leg_torques[3], leg_torques[4],
                       leg_torques[5], leg_torques[6], leg_torques[7], leg_torques[8], leg_torques[9],
                       arm_torques[0], arm_torques[1], arm_torques[2],
                       arm_torques[3], arm_torques[4], arm_torques[5]]
        if self.simulation_mode != self.simulation_modes['torque']:
            self.simulation_mode = self.simulation_modes['torque']
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_torque(goal_torque)

    def get_arm_goal_torques(self, arm_positions, arm_velocities):
        """
        Calculate arm goal torques.
        """
        arm_goal_torque = np.zeros(6)
        for i in range(6):
            arm_goal_torque[i] = self.arm_p_gains[i % self.num_joints_per_arms] * (arm_positions[i] - self.q_arm[i]) + self.arm_d_gains[i % self.num_joints_per_arms] * (arm_velocities[i] - self.dq_arm[i])
        return arm_goal_torque
        
    def update_sensor_info(self):
        """
        Get sensor info and write it to shared memory.
        """
        # get joint states
        q = self.simulator.get_current_position()
        dq = self.simulator.get_current_velocity()
        
        self.q_leg  = np.array([q[0], q[1], q[2], q[3], q[4],
                                q[5], q[6], q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]
        self.dq_arm = dq[10:16]

        leg_data = {'joint_positions':  self.q_leg,
                    'joint_velocities': self.dq_leg}
        arm_data = {'joint_positions':  self.q_arm,
                    'joint_velocities': self.dq_arm}
        MM.LEG_STATE.set(leg_data)
        MM.ARM_STATE.set(arm_data)
        
        # get imu states
        self.accel = self.simulator.get_imu_acceleration()
        self.omega = self.simulator.get_imu_angular_rate()
        self.foot_contacts = self.simulator.get_foot_contacts()
        
        sense_data = {'imu_acceleration': self.accel,
                      'imu_ang_rate':     self.omega,
                      'foot_contacts':    self.foot_contacts}
        MM.SENSE_STATE.set(sense_data)

        
    def get_observation(self):
        """
        get observations write it to shared memory.
        """
        # get joint states
        q = self.simulator.get_current_position()
        dq = self.simulator.get_current_velocity()
        
        # TODO: fix joint order, leg joint states [10] and velocities [10]
        self.q_leg  = np.array([q[0], q[1], q[2], q[3], q[4],
                                q[5], q[6], q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]

        # self.dq_leg[4] = max(-1, min(1, self.dq_leg[4]))
        # self.dq_leg[9] = max(-1, min(1, self.dq_leg[9]))


        self.dq_arm = dq[10:16]

        # get body states
        self.accel = self.simulator.get_imu_acceleration()
        self.omega = self.simulator.get_imu_angular_rate()
        self.foot_contacts = self.simulator.get_foot_contacts()
        contact_states = np.array([
            1 if self.foot_contacts[0] or self.foot_contacts[1] else 0,
            1 if self.foot_contacts[2] or self.foot_contacts[3] else 0,
        ])

        R_wb = self.simulator.get_body_rot_mat()
        w_bb = self.omega
        p_wb = self.simulator.get_body_position()
        v_wb = self.simulator.get_body_velocity()
        a_wb = R_wb @ self.accel
        v_bb = R_wb.T @ v_wb
        proj_grav = R_wb.T @ np.array([0., 0., -1])

        obs = {
            "body_pos": p_wb,
            "body_linear_vel": v_bb,
            "body_angular_vel": w_bb,
            "proj_gravity": proj_grav,
            "leg_states": self.q_leg,
            "leg_velocities": self.dq_leg,
            "arm_states": self.q_arm,
            "arm_velocities": self.dq_arm,
            "contact_states": contact_states,   # [2] [RF, LF]
        }

        return obs
    
    def calculate_robot_model(self):
        """
        Calculate kinematics & dynamics and write it to shared memory.
        """
        r1, r2, r3, r4, r5 = self.q_leg[0], self.q_leg[1], self.q_leg[2], self.q_leg[3], self.q_leg[4]
        l1, l2, l3, l4, l5 = self.q_leg[5], self.q_leg[6], self.q_leg[7], self.q_leg[8], self.q_leg[9]
        dr1, dr2, dr3, dr4, dr5 = self.dq_leg[0], self.dq_leg[1], self.dq_leg[2], self.dq_leg[3], self.dq_leg[4]
        dl1, dl2, dl3, dl4, dl5 = self.dq_leg[5], self.dq_leg[6], self.dq_leg[7], self.dq_leg[8], self.dq_leg[9]

        R_wb = self.simulator.get_body_rot_mat()
        w_bb = self.omega
        p_wb = self.simulator.get_body_position()
        v_wb = self.simulator.get_body_velocity()
        a_wb = R_wb @ self.accel
        v_bb = R_wb.T @ v_wb
        yaw_angle = np.arctan2(R_wb[1, 0], R_wb[0, 0])

        # compute leg forward kinematics
        p_bt_r, v_bt_r, Jv_bt_r, dJv_bt_r, \
        p_bh_r, v_bh_r, Jv_bh_r, dJv_bh_r, \
        p_ba_r, v_ba_r, Jv_ba_r, dJv_ba_r, \
        p_bf_r, v_bf_r,  R_bf_r,  Jw_bf_r, dJw_bf_r, \
        p_bt_l, v_bt_l, Jv_bt_l, dJv_bt_l, \
        p_bh_l, v_bh_l, Jv_bh_l, dJv_bh_l, \
        p_ba_l, v_ba_l, Jv_ba_l, dJv_ba_l, \
        p_bf_l, v_bf_l,  R_bf_l,  Jw_bf_l, dJw_bf_l = kin.legFK(r1, r2, r3, r4, r5,
                                                                l1, l2, l3, l4, l5,
                                                                dr1, dr2, dr3, dr4, dr5,
                                                                dl1, dl2, dl3, dl4, dl5)

        # compute robot forward kinematics
        p_wt_r, v_wt_r, Jv_wt_r, dJvdq_wt_r, \
        p_wh_r, v_wh_r, Jv_wh_r, dJvdq_wh_r, \
        p_wa_r, v_wa_r, Jv_wa_r, dJvdq_wa_r, \
        p_wf_r, v_wf_r,  \
        R_wf_r, w_ff_r, Jw_ff_r, dJwdq_ff_r, \
        p_wt_l, v_wt_l, Jv_wt_l, dJvdq_wt_l, \
        p_wh_l, v_wh_l, Jv_wh_l, dJvdq_wh_l, \
        p_wa_l, v_wa_l, Jv_wa_l, dJvdq_wa_l, \
        p_wf_l, v_wf_l,  \
        R_wf_l, w_ff_l, Jw_ff_l, dJwdq_ff_l = kin.robotFK(R_wb, p_wb, w_bb, v_bb,
                                                          p_bt_r, Jv_bt_r, dJv_bt_r,
                                                          p_bh_r, Jv_bh_r, dJv_bh_r,
                                                          p_ba_r, Jv_ba_r, dJv_ba_r, R_bf_r, Jw_bf_r, dJw_bf_r,
                                                          p_bt_l, Jv_bt_l, dJv_bt_l,
                                                          p_bh_l, Jv_bh_l, dJv_bh_l,
                                                          p_ba_l, Jv_ba_l, dJv_ba_l, R_bf_l, Jw_bf_l, dJw_bf_l,
                                                          dr1, dr2, dr3, dr4, dr5,
                                                          dl1, dl2, dl3, dl4, dl5)

        # calculate robot dynamics
        H, CG, AG, dAGdq, p_wg, v_wg, k_wg = dyn.robotID(R_wb, p_wb, w_bb, v_bb,
                                                         r1, r2, r3, r4, r5,
                                                         l1, l2, l3, l4, l5,
                                                         dr1, dr2, dr3, dr4, dr5,
                                                         dl1, dl2, dl3, dl4, dl5)

        # save as estimation data
        estimation_data = {}
        estimation_data['time_stamp']        = np.array([self.simulator.get_current_time()])
        estimation_data['body_position']     = p_wb
        estimation_data['body_velocity']     = v_wb
        estimation_data['body_acceleration'] = a_wb
        estimation_data['body_rot_matrix']   = R_wb
        estimation_data['body_ang_rate']     = w_bb
        estimation_data['body_yaw_ang']      = np.array([yaw_angle])
        estimation_data['com_position']      = p_wg
        estimation_data['com_velocity']      = v_wg
        estimation_data['ang_momentum']      = k_wg
        estimation_data['H_matrix']          = H
        estimation_data['CG_vector']         = CG
        estimation_data['AG_matrix']         = AG
        estimation_data['dAGdq_vector']      = dAGdq
        estimation_data['foot_contacts']     = self.foot_contacts

        estimation_data['right_foot_rot_matrix'] = R_wf_r
        estimation_data['right_foot_ang_rate']   = w_ff_r
        estimation_data['right_foot_Jw']         = Jw_ff_r
        estimation_data['right_foot_dJwdq']      = dJwdq_ff_r
        estimation_data['right_foot_position']   = p_wf_r
        estimation_data['right_foot_velocity']   = v_wf_r
        estimation_data['right_toe_position']    = p_wt_r
        estimation_data['right_toe_velocity']    = v_wt_r
        estimation_data['right_toe_Jv']          = Jv_wt_r
        estimation_data['right_toe_dJvdq']       = dJvdq_wt_r
        estimation_data['right_heel_position']   = p_wh_r
        estimation_data['right_heel_velocity']   = v_wh_r
        estimation_data['right_heel_Jv']         = Jv_wh_r
        estimation_data['right_heel_dJvdq']      = dJvdq_wh_r
        estimation_data['right_ankle_position']  = p_wa_r
        estimation_data['right_ankle_velocity']  = v_wa_r
        estimation_data['right_ankle_Jv']        = Jv_wa_r
        estimation_data['right_ankle_dJvdq']     = dJvdq_wa_r

        estimation_data['left_foot_rot_matrix']  = R_wf_l
        estimation_data['left_foot_ang_rate']    = w_ff_l
        estimation_data['left_foot_Jw']          = Jw_ff_l
        estimation_data['left_foot_dJwdq']       = dJwdq_ff_l
        estimation_data['left_foot_position']    = p_wf_l
        estimation_data['left_foot_velocity']    = v_wf_l
        estimation_data['left_toe_position']     = p_wt_l
        estimation_data['left_toe_velocity']     = v_wt_l
        estimation_data['left_toe_Jv']           = Jv_wt_l
        estimation_data['left_toe_dJvdq']        = dJvdq_wt_l
        estimation_data['left_heel_position']    = p_wh_l
        estimation_data['left_heel_velocity']    = v_wh_l
        estimation_data['left_heel_Jv']          = Jv_wh_l
        estimation_data['left_heel_dJvdq']       = dJvdq_wh_l
        estimation_data['left_ankle_position']   = p_wa_l
        estimation_data['left_ankle_velocity']   = v_wa_l
        estimation_data['left_ankle_Jv']         = Jv_wa_l
        estimation_data['left_ankle_dJvdq']      = dJvdq_wa_l

        MM.ESTIMATOR_STATE.set(estimation_data)







# Base class for RL tasks
class SimpleTask():

    def __init__(self, env_cfg):
        self.sim_params =   None
        self.physics_engine = None
        self.sim_device = 'cuda:0'
        self.headless = True

        self.dt = env_cfg.control.decimation * env_cfg.sim.dt
        self.decimation = env_cfg.control.decimation
        self.device = 'cuda:0'
        self.cfg = env_cfg
        self.graphics_device_id = -1

        self.num_envs = 1
        self.num_obs = 36
        self.num_privileged_obs = 36
        self.num_actions = 16

        self.torque_limits = 1000
        self.action_scale = env_cfg.control.action_scale
        self.num_step = 0

        
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float,
                                   device=self.device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self.num_dof = 16
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float,
                                           device=self.device,
                                           requires_grad=False)
        #self.dof_names = ['hip_yaw_l', 'hip_roll_l', 'hip_pitch_l', 'knee_pitch_l', 'ankle_pitch_l', 'hip_yaw_r', 'hip_roll_r', 'hip_pitch_r', 'knee_pitch_r', 'ankle_pitch_r']
        self.dof_names =  ['hip_yaw_l', 'hip_roll_l', 'hip_pitch_l', 'knee_pitch_l', 'ankle_pitch_l', 'hip_yaw_r', 'hip_roll_r', 'hip_pitch_r', 'knee_pitch_r', 'ankle_pitch_r', 'shoulder_pitch_l', 'shoulder_roll_l', 'elbow_pitch_l', 'shoulder_pitch_r', 'shoulder_roll_r', 'elbow_pitch_r']

        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            print('self.cfg.control.stiffness', self.cfg.control.stiffness)
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}
        self.phase = torch.zeros(1, 1, dtype=torch.float,device=self.device, requires_grad=False)
        
        # create envs, sim and viewer


        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        self.command_torque = []

        self.gazebo_position = []
        self.gym_position = []
        self.desire_position = []

        self.gazebo_velocity = []
        self.gym_velocity = []
        
        
        # When restart this thread, reset the shared memory (so the robot is in idle)
        MM.init()
        MM.connect()

        # BRUCE SETUP
        self.Bruce = RDS.BRUCE()

        self.gs = GazeboSimulator()
        self.gs.initialize_simulator()


        # for i in range(2):
        #     #print(i)
        #     self.gs.update_sensor_info()
        #     self.gs.calculate_robot_model()
        #     self.gs.write_position( self.gs.initial_pose[:10], self.gs.initial_pose[10:16])
        #     self.gs.simulator.step_simulation()
        print("self.gs.initial_pose[:10], self.gs.initial_pose[10:16]", self.gs.initial_pose[:10], self.gs.initial_pose[10:16])
        MM.THREAD_STATE.set({'simulation': np.array([1.0])}, opt='only')  # thread is running
        
        
        self.leg_position = np.load('leg_position.npy')
        self.leg_torque = np.load('leg_torque.npy')
        self.true_position = np.load('true_position.npy')

    def smooth_sqr_wave(self ):
        eps = 0.2
        phase_freq = 1.
        p = 2.*torch.pi*self.phase * phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + eps**2.)) + 1./2.

    def get_observations(self):
        
        obs = torch.zeros(
            1, 46, dtype=torch.float,
            device=self.device, requires_grad=False)

        gs_observation = self.gs.get_observation()

        leg_position_now = gs_observation['leg_states']
        HIP_YAW_R     = leg_position_now[0]
        HIP_ROLL_R= leg_position_now[1]
        HIP_PITCH_R = leg_position_now[2]
        KNEE_PITCH_R= leg_position_now[3]
        ANKLE_PITCH_R= leg_position_now[4]
        HIP_YAW_L   = leg_position_now[5]
        HIP_ROLL_L  = leg_position_now[6]
        HIP_PITCH_L = leg_position_now[7]
        KNEE_PITCH_L= leg_position_now[8]
        ANKLE_PITCH_L= leg_position_now[9]

        joint_positions = torch.tensor(
            [[HIP_YAW_L, 
                HIP_ROLL_L, 
                HIP_PITCH_L,
                KNEE_PITCH_L,
                ANKLE_PITCH_L,
                HIP_YAW_R,
                HIP_ROLL_R,
                HIP_PITCH_R,
                KNEE_PITCH_R,
                ANKLE_PITCH_R
                ]]
        ,dtype=torch.float,device=self.device, requires_grad=False)
        
        
        
        leg_velocity_now = gs_observation['leg_velocities']
        HIP_YAW_R     = leg_velocity_now[0] 
        HIP_ROLL_R= leg_velocity_now[1] 
        HIP_PITCH_R = leg_velocity_now[2]
        KNEE_PITCH_R= leg_velocity_now[3]
        ANKLE_PITCH_R= leg_velocity_now[4]
        HIP_YAW_L   = leg_velocity_now[5]
        HIP_ROLL_L  = leg_velocity_now[6]
        HIP_PITCH_L = leg_velocity_now[7]
        KNEE_PITCH_L= leg_velocity_now[8]
        ANKLE_PITCH_L= leg_velocity_now[9]


        joint_velocities =torch.tensor(
            [[HIP_YAW_L, 
                HIP_ROLL_L, 
                HIP_PITCH_L, 
                KNEE_PITCH_L,
                ANKLE_PITCH_L,
                HIP_YAW_R,
                HIP_ROLL_R,
                HIP_PITCH_R,
                KNEE_PITCH_R,
                ANKLE_PITCH_R
                ]]
        ,dtype=torch.float,device=self.device, requires_grad=False)
        

        obs[:,0] = torch.tensor(gs_observation['body_pos'][2],dtype=torch.float,device=self.device, requires_grad=False)                               # [1] Base height
        obs[:,1:4] =  torch.tensor(gs_observation['body_linear_vel'],dtype=torch.float,device=self.device, requires_grad=False)                             # [3] Base linear velocity
        obs[:,4:7] = torch.tensor(gs_observation['body_angular_vel'],dtype=torch.float,device=self.device, requires_grad=False)                                   # [3] Base angular velocity
        obs[:,7:10] = torch.tensor(gs_observation['proj_gravity'], dtype=torch.float,device=self.device, requires_grad=False)                       # [3] Projected gravity
        obs[:,10:13] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float,device=self.device, requires_grad=False)  # [3] Velocity commands
        obs[:,13] = self.smooth_sqr_wave()             # [1] Contact schedule
        obs[:,14] = torch.sin(2*torch.pi*self.phase)        # [1] Phase variable
        obs[:,15] = torch.cos(2*torch.pi*self.phase)        # [1] Phase variable
        obs[:,16:26] = joint_positions                                  # [10] Joint states
        obs[:,26:36] = joint_velocities                          # [10] Joint velocities
        #obs[:,36:38] = torch.tensor([gs_observation['contact_states'][1], gs_observation['contact_states'][0]],dtype=torch.float,device=self.device, requires_grad=False)                                  # [2] Contact states
        #print("obs", obs)
        obs[:,36:46] = torch.zeros(10, dtype=torch.float,device=self.device, requires_grad=False)

        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        return obs
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""

        return None

    def reset(self):
        """ Reset all robots"""
        self.num_step = 0
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.phase = torch.zeros(1, 1, dtype=torch.float,device=self.device, requires_grad=False)
        #obs = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # TODO
        return None, None



    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller

        #print("1", self.p_gains)
        # print("2", actions)
        # print("3",self.default_dof_pos)
        # print("4", self.dof_pos)
        # print("5", self.d_gains)
        # print("6", self.dof_vel)
        action_16 = torch.zeros(16, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        PI   = 3.1415926
        PI_2 = PI / 2
        default_pos = [-1.5707963267948966+PI_2,
        1.5553965422687077-PI_2,
        0.5969977704117716,
        -0.9557276982676578,
        0.3587299278558862,
        -1.5707963267948966+PI_2,
        1.5861961113210856-PI_2,
        0.5969977704117716,
        -0.9557276982676578 ,
        0.3587299278558862,
        0.7,
        -1.3,
        -2.0,
        -0.7,
        1.3,
        2.0,
        ]
        action_16[:, :10] = actions
        action_16[:,10:16] =  torch.tensor(default_pos[10:16], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        torques = 0.085*self.p_gains*(action_16 * self.action_scale \
                                - self.dof_pos) \
                - 1*self.d_gains*self.dof_vel
        # torques = torch.zeros_like(torques)
        # torques[:, 3] = 50
        # torques[:, 2] = 0

        # self.command_torque.append(torques.cpu().numpy().reshape(-1, 10))

        # arr = (np.array(self.command_torque).reshape(-1, 10))
        # #print(arr.shape)
        # max_value = np.amax(arr, axis=0)
        # min_value = np.amin(arr, axis=0)

        # print("每个维度的最大值：", max_value)
        # print("每个维度的最小值：", min_value)


        # 每个维度的最大值： [ 2.8139427   9.753119   11.796463   12.044629    1.1745793   3.6568465
        #   2.8735065  11.509403    7.886794    0.93507105]
        # 每个维度的最小值： [ -7.9476767  -3.7922683 -12.406027   -6.624006   -2.1163614  -2.0046391
        #   -6.110575  -12.334394   -3.077526   -1.5054433]


        # 每个维度的最大值： [ 61.11411   100.90909   228.29492   153.30461     6.6565685  53.77648
        #   25.2868    156.03282   172.16446    14.149808 ]
        # 每个维度的最小值： [ -36.825253  -34.866634  -49.319305  -74.413635   -8.695641  -50.94762
        #  -235.49443   -93.87101   -92.51104   -12.250632]


        #print('torques', torques)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, actions, obs_gym):
        # TODO
        #print("action", actions)
        if self.Bruce.thread_error():
            self.Bruce.stop_threading()


        for deci in range(self.decimation):
            self.num_step+=1

            self.gs.update_sensor_info()
            self.gs.calculate_robot_model()
        
            leg_command = MM.LEG_COMMAND.get()
            arm_command = MM.ARM_COMMAND.get()
            

            gs_observation = self.gs.get_observation()
        
        
            leg_position_now = gs_observation['leg_states']
            arm_position_now = gs_observation['arm_states']

            HIP_YAW_R     = leg_position_now[0] 
            HIP_ROLL_R= leg_position_now[1] 
            HIP_PITCH_R = leg_position_now[2]
            KNEE_PITCH_R= leg_position_now[3]
            ANKLE_PITCH_R= leg_position_now[4]
            HIP_YAW_L   = leg_position_now[5] 
            HIP_ROLL_L  = leg_position_now[6]
            HIP_PITCH_L = leg_position_now[7]
            KNEE_PITCH_L= leg_position_now[8]
            ANKLE_PITCH_L= leg_position_now[9]

            SHOULDER_PITCH_R = arm_position_now[0]
            SHOULDER_ROLL_R = arm_position_now[1]
            ELBOW_YAW_R = arm_position_now[2]
            SHOULDER_PITCH_L = arm_position_now[3]
            SHOULDER_ROLL_L = arm_position_now[4]
            ELBOW_YAW_L = arm_position_now[5]

            joint_positions = torch.tensor(
                [[HIP_YAW_L, 
                    HIP_ROLL_L, 
                    HIP_PITCH_L,
                    KNEE_PITCH_L,
                    ANKLE_PITCH_L,
                    HIP_YAW_R,
                    HIP_ROLL_R,
                    HIP_PITCH_R,
                    KNEE_PITCH_R,
                    ANKLE_PITCH_R,
                    SHOULDER_PITCH_L,
                    SHOULDER_ROLL_L,
                    ELBOW_YAW_L,
                    SHOULDER_PITCH_R,
                    SHOULDER_ROLL_R,
                    ELBOW_YAW_R,
                    ]]
            ,dtype=torch.float,device=self.device, requires_grad=False)
            
            
            
            leg_velocity_now = gs_observation['leg_velocities']
            arm_velocity_now = gs_observation['arm_states']

            HIP_YAW_R     = leg_velocity_now[0] 
            HIP_ROLL_R= leg_velocity_now[1] 
            HIP_PITCH_R = leg_velocity_now[2]
            KNEE_PITCH_R= leg_velocity_now[3]
            ANKLE_PITCH_R= leg_velocity_now[4]
            HIP_YAW_L   = leg_velocity_now[5]
            HIP_ROLL_L  = leg_velocity_now[6]
            HIP_PITCH_L = leg_velocity_now[7]
            KNEE_PITCH_L= leg_velocity_now[8]
            ANKLE_PITCH_L= leg_velocity_now[9]

            SHOULDER_PITCH_R = arm_velocity_now[0]
            SHOULDER_ROLL_R = arm_velocity_now[1]
            ELBOW_YAW_R = arm_velocity_now[2]
            SHOULDER_PITCH_L = arm_velocity_now[3]
            SHOULDER_ROLL_L = arm_velocity_now[4]
            ELBOW_YAW_L = arm_velocity_now[5]
            
            joint_velocities =torch.tensor(
                [[HIP_YAW_L, 
                    HIP_ROLL_L, 
                    HIP_PITCH_L, 
                    KNEE_PITCH_L,
                    ANKLE_PITCH_L,
                    HIP_YAW_R,
                    HIP_ROLL_R,
                    HIP_PITCH_R,
                    KNEE_PITCH_R,
                    ANKLE_PITCH_R,
                    SHOULDER_PITCH_L,
                    SHOULDER_ROLL_L,
                    ELBOW_YAW_L,
                    SHOULDER_PITCH_R,
                    SHOULDER_ROLL_R,
                    ELBOW_YAW_R,
                    ]]
            ,dtype=torch.float,device=self.device, requires_grad=False)
            

            self.dof_pos = joint_positions
            self.dof_vel = joint_velocities
            #self.torques = self._compute_torques(actions).view(self.torques.shape)
            # #print(actions+self.default_dof_pos)

            torques_gz = self.torques.cpu().numpy()        
            torques_gz = torques_gz[0]
            # print("self.leg_torque",self.leg_torque.shape)
            # torques_gz = self.leg_torque[self.num_step%10000]
            leg_goal_torques = np.array([
                                    torques_gz[5],
                                    torques_gz[6],
                                    torques_gz[7],
                                    torques_gz[8],
                                    torques_gz[9],
                                    torques_gz[0],
                                    torques_gz[1],
                                    torques_gz[2],
                                    torques_gz[3],
                                    torques_gz[4],
                                        ])
            arm_goal_torques = np.array([
                                    torques_gz[13],
                                    torques_gz[14],
                                    torques_gz[15],
                                    torques_gz[10],
                                    torques_gz[11],
                                    torques_gz[12],
                                        ])
            position = (actions).cpu().numpy()[0]
            #position = np.zeros(10)
            #position = (self.leg_position[self.num_step%10000])
            #position = (self.true_position[self.num_step%10000])
            #if position.has_n
            #print(position)
            
            goal_positions = np.array([
                                    position[5],
                                    position[6],
                                    position[7],
                                    position[8],
                                    position[9],
                                    position[0],
                                    position[1],
                                    position[2],
                                    position[3],
                                    position[4],
                                        ])

            default_pos = [
            -1.5707963267948966+PI_2,
            1.5861961113210856-PI_2,
            0.5969977704117716,
            -0.9557276982676578 ,
            0.3587299278558862,                
            -1.5707963267948966+PI_2,
            1.5553965422687077-PI_2,
            0.5969977704117716,
            -0.9557276982676578,
            0.3587299278558862,
            -0.7,
            1.3,
            2.0,
            0.7,
            -1.3,
            -2.0,
            ]
            #action_16[:,10:16] =  torch.tensor(default_pos[10:16], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            #self.gs.write_torque(leg_goal_torques, arm_goal_torques)

            self.gs.write_position(goal_positions, default_pos[10:16])
            self.gs.simulator.step_simulation()
            time.sleep(0.000)  # delay if needed
            
        
        obs = torch.zeros(
            1, 46, dtype=torch.float,
            device=self.device, requires_grad=False)




        obs[:,0] = torch.tensor(gs_observation['body_pos'][2],dtype=torch.float,device=self.device, requires_grad=False)                               # [1] Base height
        obs[:,1:4] = torch.tensor(gs_observation['body_linear_vel'],dtype=torch.float,device=self.device, requires_grad=False)                             # [3] Base linear velocity
        obs[:,4:7] = torch.tensor(gs_observation['body_angular_vel'],dtype=torch.float,device=self.device, requires_grad=False)                                   # [3] Base angular velocity
        obs[:,7:10] = torch.tensor(gs_observation['proj_gravity'], dtype=torch.float,device=self.device, requires_grad=False)                       # [3] Projected gravity

        # obs[:,0] = obs_gym[:,0]#torch.tensor(gs_observation['body_pos'][2] *  (1./0.6565),dtype=torch.float,device=self.device, requires_grad=False)                               # [1] Base height
        # obs[:,1:4] = obs_gym[:,1:4] #torch.tensor(gs_observation['body_linear_vel'],dtype=torch.float,device=self.device, requires_grad=False)                             # [3] Base linear velocity
        # obs[:,4:7] = obs_gym[:,4:7]#torch.tensor(gs_observation['body_angular_vel'],dtype=torch.float,device=self.device, requires_grad=False)                                   # [3] Base angular velocity
        # obs[:,7:10] = obs_gym[:,7:10]#torch.tensor(gs_observation['proj_gravity'], dtype=torch.float,device=self.device, requires_grad=False)                       # [3] Projected gravity
        
        obs[:,10:13] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float,device=self.device, requires_grad=False)  # [3] Velocity commands
        obs[:,13] = self.smooth_sqr_wave()             # [1] Contact schedule
        obs[:,14] = torch.sin(2*torch.pi*self.phase)        # [1] Phase variable
        obs[:,15] = torch.cos(2*torch.pi*self.phase)        # [1] Phase variable
        obs[:,16:26] = joint_positions[0,:10]                                  # [10] Joint states
        obs[:,26:36] = joint_velocities[0,:10]                          # [10] Joint velocities
        obs[:,36:46] = actions.clone()
        
        # obs[:,16:26] = obs_gym[:,16:26]
        # obs[:,16:26] = obs_gym[:,16:26]
        #obs[:,13:16] = obs_gym[:,13:16]
        #obs[:,36:38] = torch.tensor([gs_observation['contact_states'][1], gs_observation['contact_states'][0]],dtype=torch.float,device=self.device, requires_grad=False)     #obs_gym[:,36:38]                              # [2] Contact states
        #print("obs", obs)
        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        self.gazebo_position.append(joint_positions[0,:10].cpu().numpy())
        self.gym_position.append(obs_gym[0,16:26].cpu().numpy())
        self.desire_position.append((actions[0]).cpu().numpy())
        
        self.gazebo_velocity.append(joint_velocities[0,:10].cpu().numpy())
        self.gym_velocity.append(obs_gym[0,26:36].cpu().numpy())
        
        if len(self.gazebo_position) == 200:

            gazebo_position_arr = np.zeros((len(self.gazebo_position),10))
            gym_position_arr = np.zeros((len(self.gym_position),10))
            desire_position_arr = np.zeros((len(self.desire_position),10))

            gazebo_velocity_arr = np.zeros((len(self.gazebo_velocity),10))
            gym_velocity_arr = np.zeros((len(self.gym_velocity),10))
            
            for i in range(len(self.gazebo_position)):
                gazebo_position_arr[i] = self.gazebo_position[i]
                gym_position_arr[i] = self.gym_position[i]
                desire_position_arr[i] = self.desire_position[i]

                gazebo_velocity_arr[i] = self.gazebo_velocity[i]
                gym_velocity_arr[i] = self.gym_velocity[i]
                
            np.save('gazebo_position.npy', gazebo_position_arr)
            np.save('gym_position.npy', gym_position_arr)
            np.save('desire_position.npy', desire_position_arr)
            
            np.save('gazebo_velocity.npy', gazebo_velocity_arr)
            np.save('gym_velocity.npy', gym_velocity_arr)
            print("save!")
        return obs











