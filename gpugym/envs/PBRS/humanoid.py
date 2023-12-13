"""
Environment file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import LeggedRobot


class Humanoid(LeggedRobot):

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.

    def compute_observations(self):
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        in_contact = torch.gt(
            self.contact_forces[:, self.end_eff_ids, 2], 0).int()
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        self.commands[:, 0:2] = torch.where(
            torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.5,
            0., self.commands[:, 0:2].double()).float()
        self.commands[:, 2:3] = torch.where(
            torch.abs(self.commands[:, 2:3]) < 0.5,
            0., self.commands[:, 2:3].double()).float()
        self.obs_buf = torch.cat((
            base_z,                                 # [1] Base height
            self.base_lin_vel,                      # [3] Base linear velocity
            self.base_ang_vel,                      # [3] Base angular velocity
            self.projected_gravity,                 # [3] Projected gravity
            self.commands[:, 0:3],                  # [3] Velocity commands
            self.smooth_sqr_wave(self.phase),       # [1] Contact schedule
            torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
            torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
            self.dof_pos,                           # [10] Joint states
            self.dof_vel,                           # [10] Joint velocities
            in_contact,                             # [2] Contact states
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0] = noise_scales.base_z * self.obs_scales.base_z
        noise_vec[1:4] = noise_scales.lin_vel
        noise_vec[4:7] = noise_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity
        noise_vec[10:16] = 0.   # commands
        noise_vec[16:26] = noise_scales.dof_pos
        noise_vec[26:36] = noise_scales.dof_vel
        noise_vec[36:38] = noise_scales.in_contact  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements \
                * noise_level \
                * self.obs_scales.height_measurements
        noise_vec = noise_vec * noise_level
        return noise_vec

    def _custom_reset(self, env_ids):
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)

    def _post_physics_step_callback(self):
        self.phase = torch.fmod(self.phase + self.dt, 1.0)
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt) == 0) \
            .nonzero(as_tuple=False).flatten()
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)
            if (self.cfg.domain_rand.push_robots and
                (self.common_step_counter
                % self.cfg.domain_rand.push_interval == 0)):
                self._push_robots()

    def _push_robots(self):
        # Randomly pushes the robots.
        # Emulates an impulse by setting a randomized base velocity.
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:8] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 1), device=self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        #print("self.termination_contact_indices", self.termination_contact_indices)
        #print("self.contact_forces", self.contact_forces[0])
        self.reset_buf = torch.any((term_contact > 1.), dim=1)
        #print("1 term_contact", torch.any((term_contact > 1.), dim=1))
        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        #print("2 self.base_lin_vel", torch.any(torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1))
        self.reset_buf |= torch.any(
           torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)
        #print("3 self.base_ang_vel", torch.norm(self.base_ang_vel, dim=-1, keepdim=True))
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        #print("4 torch.abs(self.projected_gravity[:, 0:1])",  torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1))
        self.reset_buf |= torch.any(
           torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        #print("5 torch.abs(self.projected_gravity[:, 1:2])", torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1))
        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)
        #print("6 self.base_pos[:, 2:3]",torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)    )
        # # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        #print("7 self.time_out_buf", self.time_out_buf)
        self.reset_buf |= self.time_out_buf

# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_feat_alternate_leading(self):
        # Reward long steps with anti-symmetric pattern+
        #print("contact_forces", self.contact_forces[:, self.feet_indices, 2])
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 

        left_foot_contact = contact_filt[:, 0]
        right_foot_contact = contact_filt[:, 1]

        
        # New code for alternate leading behavior
        left_joint_pos = self.dof_pos[:, 2]  # Assuming index 2 is left hip joint position
        right_joint_pos = self.dof_pos[:, 7]  # Assuming index 7 is right hip joint position

        # Check if left leg is leading and in contact, and then right leg takes the lead
        left_leading_and_contact =  torch.logical_and((left_joint_pos > right_joint_pos) , left_foot_contact)
        right_leading_and_contact = torch.logical_and((left_joint_pos < right_joint_pos) , right_foot_contact)
        
        left_over_right = torch.logical_and(left_leading_and_contact, self.feet_ahead[:,1])
        right_over_left = torch.logical_and(right_leading_and_contact, self.feet_ahead[:,0])
        
        
        #alternate_leading_reward = torch.where(left_over_right, 1.0, 0.0)
        #alternate_leading_reward2 = torch.where(right_over_left, 1.0, 0.0)

        # false, falses
        self.feet_ahead[:,0] = torch.where(self.feet_ahead[:,0]==self.feet_ahead[:,1], left_leading_and_contact, self.feet_ahead[:,0])
        self.feet_ahead[:,1] = torch.where(self.feet_ahead[:,0]==self.feet_ahead[:,1], right_leading_and_contact, self.feet_ahead[:,1])
        
        
        self.feet_ahead[:,0] = torch.where(left_over_right, True, self.feet_ahead[:,0])
        self.feet_ahead[:,1] = torch.where(left_over_right, False, self.feet_ahead[:,1])
        
        self.feet_ahead[:,0] = torch.where(right_over_left, False, self.feet_ahead[:,0])
        self.feet_ahead[:,1] = torch.where(right_over_left, True, self.feet_ahead[:,1])
        


        #anti_symmetric_pattern = torch.logical_xor(self.feet_ahead[:,0], self.feet_ahead[:,1])

        # Calculate reward for anti-symmetric steps
        #print(self.feet_air_time.shape, first_contact.shape, anti_symmetric_pattern.shape, anti_symmetric_pattern.unsqueeze(1).repeat(1,2).shape)
        rew_anti_symmetric = self.feet_lead_time[:,1]  * left_over_right * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        rew_anti_symmetric += self.feet_lead_time[:,0]  * right_over_left * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        
        self.feet_lead_time += self.dt
        self.feet_lead_time[:,0] *= self.feet_ahead[:,0]
        self.feet_lead_time[:,1] *= self.feet_ahead[:,1]


        # Combine the rewards
        total_reward = rew_anti_symmetric

        
        
        
        
        
        
        return total_reward


    def _reward_feet_air_time(self):
        # Reward long steps with anti-symmetric pattern+
        #print("contact_forces", self.contact_forces[:, self.feet_indices, 2])
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt

        # Identify anti-symmetric step pattern
        # Assuming feet_indices[0] and feet_indices[1] correspond to left and right foot respectively
        #print(contact_filt.shape)
        left_foot_contact = contact_filt[:, 0]
        right_foot_contact = contact_filt[:, 1]
        anti_symmetric_pattern = torch.logical_xor(left_foot_contact, right_foot_contact)

        # Calculate reward for anti-symmetric steps
        #print(self.feet_air_time.shape, first_contact.shape, anti_symmetric_pattern.shape, anti_symmetric_pattern.unsqueeze(1).repeat(1,2).shape)
        rew_anti_symmetric = torch.sum(self.feet_air_time * first_contact * anti_symmetric_pattern.unsqueeze(1).repeat(1,2), dim=1)
        rew_anti_symmetric *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # No reward for zero command

        self.feet_air_time *= ~contact_filt
        #print("rew_anti_symmetric", rew_anti_symmetric)
        
        
        return rew_anti_symmetric

    

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2])*2/torch.pi)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = self.root_states[:, 2].unsqueeze(1)
        #print("base_height", base_height)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        # Reward tracking upright orientation
        error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_dof_vel(self):
        # Reward zero dof velocities
        dof_vel_scaled = self.dof_vel/self.cfg.normalization.obs_scales.dof_vel
        return torch.sum(self.sqrdexp(dof_vel_scaled), dim=-1)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        
        error += self.sqrdexp(
            (self.dof_pos[:, 1] - self.dof_pos[:, 6])
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        # error += self.sqrdexp(
        #     #(self.dof_pos[:, 2]-0.625 + self.dof_pos[:, 7]-0.625)
        #     (self.dof_vel[:, 2] + self.dof_vel[:, 7])
        #     / self.cfg.normalization.obs_scales.dof_pos)
        return error/4

    def _reward_ankle_regularization(self):
        # Ankle joint regularization around 0
        error = 0
        error += self.sqrdexp(
            (self.dof_pos[:, 4]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 9]) / self.cfg.normalization.obs_scales.dof_pos)
        return error

    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step

# ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.
