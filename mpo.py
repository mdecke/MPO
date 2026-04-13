import os
import copy
import yaml
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

parser = argparse.ArgumentParser(description="MPO experiment setup")
parser.add_argument("--task", type=str, default='Pendulum-v1', )
parser.add_argument("--n_envs", type=int, default=1, help="number of parallel envs")
parser.add_argument('--env_interactions', type=int, default=1_000_000, help="number of total steps across envs")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIVATION_FCTS = {
    'relu' : nn.ReLU,
    'elu' : nn.ELU,
    'tanh' : nn.Tanh,
}

def get_activation(name:str='relu') -> nn.Module:
    name_lower = str.lower(name)
    act_fcts = ACTIVATION_FCTS.keys()
    if name_lower not in act_fcts:
        raise KeyError(f'{name_lower} is not a valid activation function')
    act_f = ACTIVATION_FCTS[name_lower]
    return act_f()

def init_model_weights(model:nn.Module, mean:float=0.0, std:float=0.1, seed:int=42) -> None:
    if seed is not None:
        torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)

def load_config(config_dict_path:str, args) -> Dict:
    with open(config_dict_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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

    def sample(self,
               batch_size:int,
               n_step_horizon:int=1)->Dict[str,torch.Tensor]:
        
        max_start = self.filled_lines - n_step_horizon
        
        # Safety check: ensure we have enough samples for n-step prediction
        if self.filled_lines - n_step_horizon < 0 : 
            raise ValueError("not enough samples")
        
        start_time = torch.randint(0, max_start, (batch_size,), device=self.device)
        sampled_envs  = torch.randint(0, self.envs,  (batch_size,), device=self.device)

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

#Policy is stochastic following Normal dist --> N(mu, sigma), with mu and sigma outputs of a MLP.
class Actor(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 action_limit:float,
                 hidden_dims:List[int],
                 lr:float,
                 activation_fct:str,
                 seed:int=42):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.action_limit = action_limit

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        self.mu_head = nn.Linear(prev_dim, self.output_dim)
        self.log_sigma_head = nn.Linear(prev_dim, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input:torch.Tensor, n_samples:int=1) -> torch.Tensor:
        logits = self.net(input)
        mu = self.mu_head(logits)
        log_sigma = self.log_sigma_head(logits)
        if self.training:
            sigma = torch.exp(log_sigma) + 1e-6 #ensure std is non zero
            mu_expanded = mu.unsqueeze(1).expand(-1,n_samples,-1) #shape = (bch_sz,n_samples,act_dim)
            sigma_expanded = sigma.unsqueeze(1).expand(-1,n_samples,-1) #shape = (bch_sz,n_samples,act_dim)
            normal_dist = dist.Normal(mu_expanded,sigma_expanded)
            actions = normal_dist.rsample() #reparametrization trick conserves gradients
        else:
            actions = mu
        bounded_actions = torch.tanh(actions) * self.action_limit
        return bounded_actions

class Critic(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dims:List[int],
                 lr:float,
                 actv_fct:str,
                 output_dim:int=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        layers = []
        prev_dim = self.input_dim


        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(actv_fct))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim)) # Q(s,a): R_s x R_a --> R
        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        model_input = torch.cat((state,action), dim=-1)
        return self.net(model_input)
    
class MPO_Agent():
    def __init__(self,cfg:Dict):
        self.cfg = cfg

        self.obs_dim = self.cfg.get('env',{}).get('obs_dim')
        self.act_dim = self.cfg.get('env',{}).get('act_dim')
        self.act_lim = self.cfg.get('env',{}).get('lim', 1.0)
        self.n_envs = self.cfg.get('env',{}).get('n_envs')

        self.gamma = self.cfg.get('agent', {}).get('params',{}).get('gamma', 0.99)
        self.tau = self.cfg.get('agent', {}).get('params',{}).get('tau', 0.95)
        self.policy_samples = self.cfg.get('agent', {}).get('params',{}).get('policy_samples', 1)
        self.e_step_epsilon = self.cfg.get('agent', {}).get('params',{}).get('e_step_epsilon', 1)
        self.n_temp_dual_steps = self.cfg.get('agent', {}).get('params',{}).get('n_temp_dual_steps', 200)

        self.policy_layers = self.cfg.get('agent',{}).get('policy',{}).get('hidden_layers',[])
        self.policy_lr = self.cfg.get('agent',{}).get('policy',{}).get('lr', 0.001)
        self.policy_actv_fct = self.cfg.get('agent',{}).get('policy',{}).get('act_fct', 'relu')
        self.policy_gradient_clipping = self.cfg.get('agent',{}).get('policy',{}).get('gradient_clip', None)
        
        self.critic_layers = self.cfg.get('agent',{}).get('critic',{}).get('hidden_layers',[])
        self.critic_lr = self.cfg.get('agent',{}).get('critic',{}).get('lr', 0.001)
        self.critic_actv_fct = self.cfg.get('agent',{}).get('critic',{}).get('act_fct', 'relu')
        self.critic_gradient_clipping = self.cfg.get('agent',{}).get('critic',{}).get('gradient_clip', None)

        self.interactions = self.cfg.get('training',{}).get('max_interactions', 1_000_000)
        self.training_steps = torch.ceil(torch.tensor(self.interactions/self.n_envs)).int()
        self.warm_up_steps = self.cfg.get('training',{}).get('warm_up', 1_000) #training has started, ramp up LR over these nb of steps -> stable grads
        self.learning_starts = self.cfg.get('training',{}).get('learning_starts', 10_000) #don't take any gradient for these first n steps

        self.buffer_sz = self.cfg.get('buffer', {}).get('buffer_sz', 1_000_000)
        self.batch_sz = self.cfg.get('buffer', {}).get('batch_sz', 256)
        self.td_horizon = self.cfg.get('buffer', {}).get('td_horizon', 1)

        self._init_buffer()
        self._init_models()

        self.evaluted_q = torch.empty((self.batch_sz, self.policy_samples), dtype=torch.float32, device=device)

        self.critic_loss = []
        self.policy_loss = []
        self.mean_q_value = []

    def _init_buffer(self) -> None:
        self.buffer = Buffer(self.buffer_sz,
                             self.n_envs,
                             self.obs_dim,
                             self.act_dim
                             )
        print("[INFO]: Memory class initialized")

    def _init_models(self) -> None:
        self.policy = Actor(input_dim=self.obs_dim,
                            output_dim=self.act_dim,
                            action_limit=self.act_lim,  # Assuming symmetric action bounds
                            hidden_dims=self.policy_layers,
                            lr=self.policy_lr,
                            activation_fct=self.policy_actv_fct)
        self.critic = Critic(input_dim=self.obs_dim+self.act_dim,
                             hidden_dims=self.critic_layers,
                             lr=self.critic_lr,
                             activation_fct=self.critic_actv_fct)
        
        init_model_weights(self.policy)
        init_model_weights(self.critic)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_critic = copy.deepcopy(self.critic)

        #Targets only updated via polyak interpolation, no need to track grads
        for p in self.target_policy.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

    def update_critic(self,
                      batch_data:Dict[str,torch.Tensor]) -> None:
        next_action = self.target_policy.forward(input=batch_data["next_obs"][:,-1,:])
        q_target = self.target_critic.forward(state=batch_data['next_obs'][:,-1,:], action=next_action)

        #temporal diff
        y = q_target
        for k in range(self.td_horizon-1,-1,-1):
            termination = batch_data['term'][:,k,:]
            reward =batch_data['r'][:,k,:]
            y = reward + self.gamma * (~termination) * y
        
        q_value = self.critic.forward(state=batch_data['obs'][:,-1,:], action=batch_data['acts'][:,-1,:])
        self.mean_q_value.append(q_value.mean().item())

        critic_loss = F.mse_loss(q_value, y)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        
        if self.critic_gradient_clipping:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_gradient_clipping)
        
        self.critic.optimizer.step()

    def solve_temp_dual(self, q_samples:torch.Tensor, epsilon:float, n_dual_steps:int=200) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sz, n_samples = q_samples.shape
        log_eta = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
        dual_optimizer = optim.Adam([log_eta], lr=1e-2)

        q_values = q_samples.detach()

        for _ in range(n_dual_steps):
            dual_optimizer.zero_grad()
            eta = log_eta.exp()
            dual_loss = eta * epsilon + eta * (torch.logsumexp(q_values / eta, dim=-1) - torch.log(n_samples)).mean()
            dual_loss.backward()
            dual_optimizer.step()
        
        eta_star = log_eta.exp().detach()

        weights = torch.softmax(q_values/eta_star, dim=-1)
        return eta_star, weights

    def e_step(self,
               batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch_data['obs'][:,-1,:] #shape = (batch_sz, 1, obs_dim)
        for i,state in zip(range(self.batch_sz),obs):
            sampled_actions = self.target_policy.forward(state.unsqueeze(0),self.policy_samples).squeeze(0) #shape = (1, n_samples, act_dim) 
            for j in range(self.policy_samples):
                self.evaluted_q[i,j] = self.critic.forward(state=state,action=sampled_actions[j,:])
        
        eta, weights = self.solve_temp_dual(self.evaluted_q, self.e_step_epsilon, self.n_temp_dual_steps)
        
        return sampled_actions, weights, eta
        



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


    agent = MPO_Agent()
    
    obs, _ = env.reset()

    td_horizon = 4
    cumulative_reward = 0.0
    episode_lengths = 0
    for step in progress_bar:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action = env.action_space.sample()
            action_t = torch.tensor(action, dtype=torch.float32, device=device)
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