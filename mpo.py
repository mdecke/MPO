import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

parser = argparse.ArgumentParser(description="MPO experiment setup")
parser.add_argument("--task", type=str, default='Pendulum-v1', )
parser.add_argument("--n_envs", type=int, default=1, help="number of parallel envs")
parser.add_argument('--env_interactions', type=int, default=1_000_000, help="number of total steps across envs")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Buffer:
    def __init__(self,
                 buffer_sz:int, n_envs:int,
                 observation_dim:Tuple[int,...] | int,
                 action_dim:Tuple[int,...] | int,
                 cfg:Optional[Dict],
                 device:str="cpu",
                 action_dtype:torch.dtype=torch.float32):
        
        self.N = buffer_sz
        self.envs = n_envs
        self.obs_dim = observation_dim
        self.act_dim = action_dim
        self.buffer_cfg = cfg

        #Set up correct shape for arrays
        if isinstance(self.obs_dim, int):
            obs_shape = (self.obs_dim,)
        else:
            obs_shape = tuple(self.obs_dim)
        if isinstance(self.act_dim, int):
            act_shape = (self.act_dim,)
        else:
            act_shape = tuple(self.act_dim)
        
        self.device = device

        # Sequence of Arrays (SoA) --> each variable is stored in a (N by #_envs) tensor
        self.obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=device)
        self.next_obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.empty((self.N, self.envs, *act_shape), dtype=action_dtype, device=device)
        self.rewards = torch.empty((self.N, self.envs, 1), dtype=torch.float32, device=device)
        self.truncation = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=device)
        self.termination = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=device)
        self.infos = torch.empty((self.N, self.envs), dtype=torch.bool, device=device)
        
        self.env_steps = 0
        self.filled_lines = 0

    def add_sample(self,
                   obs:torch.Tensor,
                   actions:torch.Tensor,
                   next_obs:torch.Tensor,
                   rewards:torch.Tensor,
                   truncation:torch.Tensor,
                   termination:torch.Tensor)->None:
        
        #circular indexing
        index = self.env_steps % self.N 

        self.obs[index].copy_(obs)
        self.actions[index].copy_(actions)
        self.rewards[index].copy_(rewards.view(self.envs, 1))
        self.next_obs[index].copy_(next_obs)
        self.truncation[index].copy_(truncation.view(self.envs, 1))
        self.termination[index].copy_(termination.view(self.envs, 1))
        
        #check how many lines are full after new sample
        self.env_steps+=1
        self.filled_lines = min(self.env_steps,self.N)

    def sample(self, batch_size:int, n_step_horizon:int=1):
        
        max_start = self.filled_lines - n_step_horizon
        
        # Safety check: ensure we have enough samples for n-step prediction
        if self.filled_lines - n_step_horizon < 0 : 
            raise ValueError("not enough samples")
        
        start_time = torch.randint(0, max_start, (batch_size,), device=self.device, generator=self.generator)
        sampled_envs  = torch.randint(0, self.envs,  (batch_size,), device=self.device, generator=self.generator)

        offset= torch.arange(n_step_horizon, device=self.device) #-> [0,1,2,...,n_step_horizon]       
        time_window = (start_time[:, None] + offset[None, :]) % self.N  #-> tensor[[start_time[0], start_time[0]+1, start_time[0]+2,...],start_time[1]] wrapped around N
        env_window = sampled_envs[:, None].expand(batch_size, n_step_horizon) #-> convert sampled envs array to 2D array with lines same env id

        obs_hist = self.obs[time_window,env_window,:]
        act_hist = self.actions[time_window,env_window,:]
        next_obs_hist = self.next_obs[time_window,env_window,:]
        r_hist = self.rewards[time_window,env_window,:]
        trunc_hist = self.truncation[time_window,env_window,:]
        term_hist = self.termination[time_window,env_window,:]

        batch = {"obs": obs_hist,
                 "acts": act_hist,
                 "next_obs": next_obs_hist,
                 "r": r_hist,
                 "term": term_hist,
                 "trunc": trunc_hist}

        return batch



def main():
    env = gym.make(args.task)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    data_buffer = Buffer(buffer_sz=100000,
                         n_envs=args.n_envs,
                         observation_dim=obs_dim,
                         action_dim=act_dim,
                         cfg=None) #TODO: buffer only needs buffer cfg, not explicit

    training_steps_per_env = np.ceil(args.env_interactions/args.n_envs).astype(int)
    progress_bar = tqdm(range(training_steps_per_env), unit="step")
    
    obs, _ = env.reset()

    td_horizon = 4
    cumulative_reward = 0.0
    episode_lengths = 0
    for step in progress_bar:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action = env.action_space.sample()
            action_t = torch.tensor(obs, dtype=torch.float32, device=device)
            next_obs, reward, terminated, truncated, info = env.step(action_t.cpu().numpy())

            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).view(args.n_envs, -1)
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device).view(args.n_envs, -1)
            terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=device).view(args.n_envs, -1)
            truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=device).view(args.n_envs, -1)
            cumulative_reward += reward_tensor
            episode_lengths += 1

            data_buffer.add_sample(obs_t, action_t, next_obs_tensor, reward_tensor, truncated_tensor, terminated_tensor)

            obs = next_obs

            if step > td_horizon:
                print(data_buffer.sample(td_horizon))

            if terminated or truncated:
                episode_lengths = 0

    env.close()
        
    


if __name__ == "__main__":
    main()