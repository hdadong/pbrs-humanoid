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




# Base class for RL tasks
class SimpleTask():

    def __init__(self, env_cfg):
        self.sim_params =   None
        self.physics_engine = None
        self.sim_device = 'cuda:0'
        self.headless = True

        self.dt = env_cfg.control.decimation * env_cfg.sim.dt
        
        self.device = 'cuda:0'

        self.graphics_device_id = -1

        self.num_envs = 1
        self.num_obs = 38
        self.num_privileged_obs = 38
        self.num_actions = 10

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

    def smooth_sqr_wave(self ):
        eps = 0.2
        phase_freq = 1.
        p = 2.*torch.pi*self.phase * phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + eps**2.)) + 1./2.

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        return None

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.phase = torch.zeros(1, 1, dtype=torch.float,device=self.device, requires_grad=False)
        obs = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # TODO
        return obs, obs
    
    def step(self, actions):
        # TODO
        obs = torch.zeros(
            1, 38, dtype=torch.float,
            device=self.device, requires_grad=False)
        
        # obs[:,0] = xxx                               # [1] Base height
        # obs[:,1:4] = xxx                             # [3] Base linear velocity
        # obs[:,4:7] = xxx                             # [3] Base angular velocity
        # obs[:,7:10] = xxx                            # [3] Projected gravity
        obs[:,10:13] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float,device=self.device, requires_grad=False)  # [3] Velocity commands
        obs[:,13] = self.smooth_sqr_wave()             # [1] Contact schedule
        obs[:,14] = torch.sin(2*torch.pi*self.phase)        # [1] Phase variable
        obs[:,15] = torch.cos(2*torch.pi*self.phase)        # [1] Phase variable
        # obs[:,15:25] = xxx                           # [10] Joint states
        # obs[:,25:35] = xxx                           # [10] Joint velocities
        # obs[:,35:37] = xxx                           # [2] Contact states
        
        self.phase = torch.fmod(self.phase + self.dt, 1.0)

        return obs

