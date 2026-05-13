import os
import copy
import json
import yaml
import math
import random
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

import time
from datetime import datetime

parser = argparse.ArgumentParser(description="MPO experiment setup")
parser.add_argument("--task", type=str, default='Pendulum-v1', )
parser.add_argument('--run_name', type=str, default="train_logs", help='name of csv file with training data')
parser.add_argument("--n_envs", type=int, default=1, help="number of parallel envs")
parser.add_argument('--max_interactions', type=int, default=100000, help="number of total steps across envs")
parser.add_argument('--save_checkpoint_rate', type=int, default=500, help="rate of env interactions at which the models are saved")
parser.add_argument('--save_buffer', action='store_true', help='flag to store replay buffer')
parser.add_argument('--seed', type=int, default=1, help='global random seed (overrides config)')
args = parser.parse_args()

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

def init_model_weights(model:nn.Module, mean:float=0.0, std:float=0.1, nonlinearity:str='relu') -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def quantile_huber_loss(td_errors: torch.Tensor, tau: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    td_errors : (B, N_prime, N)  — pairwise differences y[b,n'] - z_pred[b,n]
    tau       : (B, N)           — quantile levels of z_pred
    """
    abs_err = td_errors.abs()
    huber = torch.where(abs_err <= kappa,
                        0.5 * td_errors.pow(2),
                        kappa * (abs_err - 0.5 * kappa))
    # asymmetric quantile weight: |tau - 1(delta < 0)|
    tau_w = (tau.unsqueeze(1) - (td_errors.detach() < 0).float()).abs()  # (B, N_prime, N)
    return (tau_w * huber/kappa).sum(dim=2).mean(dim=1).mean()

def load_config(config_dict_path:str, args) -> Dict:
    with open(config_dict_path, 'r') as file:
        config = yaml.safe_load(file)
    
    cli_to_yaml = {
        'task':             ('environment', 'task'),
        'n_envs':           ('environment', 'n_envs'),
        'max_interactions': ('training', 'max_interactions'),
        'save_checkpoint_rate': ('training', 'save_checkpoint_rate'),
        'save_buffer': ('buffer', 'save_buffer'),
        'seed':        ('environment', 'seed'),
    }

    for arg_name, yaml_path in cli_to_yaml.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            # walk into nested dict, create intermediate dicts if missing
            d = config
            for key in yaml_path[:-1]:
                d = d.setdefault(key, {})
            d[yaml_path[-1]] = value

    return config


class Buffer:
    def __init__(self,
                 cfg:Dict,
                 action_dtype:torch.dtype=torch.float32):
        self.cfg = cfg

        self.N = self.cfg.get('buffer', {}).get('buffer_size', 1_000_000)

        self.envs = self.cfg.get('environment', {}).get('n_envs', 1)
        self.obs_dim = self.cfg.get('environment', {}).get('obs_dim', 1)
        self.act_dim = self.cfg.get('environment', {}).get('act_dim', 1)
        self.device = self.cfg.get('environment', {}).get('device', 'cpu')
        
        #Set up correct shape for arrays
        if isinstance(self.obs_dim, int):
            obs_shape = (self.obs_dim,)
        else:
            obs_shape = tuple(self.obs_dim)
        if isinstance(self.act_dim, int):
            act_shape = (self.act_dim,)
        else:
            act_shape = tuple(self.act_dim)

        # Sequence of Arrays (SoA) --> each variable is stored in a (N by #_envs) tensor
        self.obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((self.N, self.envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((self.N, self.envs, *act_shape), dtype=action_dtype, device=self.device)
        self.raw_actions = torch.empty((self.N, self.envs, *act_shape), dtype=action_dtype, device=self.device)
        self.old_policy_log_probs = torch.empty((self.N, self.envs,1), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((self.N, self.envs, 1), dtype=torch.float32, device=self.device)
        self.truncation = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=self.device)
        self.termination = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=self.device)
        self.infos = torch.empty((self.N, self.envs), dtype=torch.bool, device=self.device)
        
        self.env_steps = 0
        self.filled_lines = 0

    def add_sample(self,
                   obs:torch.Tensor,
                   actions:torch.Tensor,
                   raw_actions:torch.Tensor,
                   log_probs_mu:torch.Tensor,
                   next_obs:torch.Tensor,
                   rewards:torch.Tensor,
                   truncation:torch.Tensor,
                   termination:torch.Tensor)->None:
        
        #circular indexing
        index = self.env_steps % self.N 
        self.obs[index].copy_(obs)
        self.actions[index].copy_(actions)
        self.raw_actions[index].copy_(raw_actions)
        self.old_policy_log_probs[index].copy_(log_probs_mu)
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
        raw_act_hist = self.raw_actions[time_window,env_window,:]
        old_policy_log_probs_hist = self.old_policy_log_probs[time_window, env_window,:]
        next_obs_hist = self.next_obs[time_window,env_window,:]
        r_hist = self.rewards[time_window,env_window,:]
        trunc_hist = self.truncation[time_window,env_window,:]
        term_hist = self.termination[time_window,env_window,:]

        batch = {"obs": obs_hist,
                 "acts": act_hist,
                 "raw_acts": raw_act_hist,
                 "log_probs": old_policy_log_probs_hist,
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
                 action_limit:List[float],
                 hidden_dims:List[int],
                 lr:float,
                 activation_fct:str,
                 layer_norm:bool=False,
                 seed:int=42):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.action_limit = action_limit

        # Action rescaling.
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_limit[1] - action_limit[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_limit[1] + action_limit[0]) / 2.0,
                dtype=torch.float32,
            ),
        )

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        self.mu_head = nn.Linear(prev_dim, self.output_dim)
        self.log_sigma_head = nn.Linear(prev_dim, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input: torch.Tensor) -> dist.Normal:
        logits = self.net(input)
        mu = self.mu_head(logits)
        log_sigma = self.log_sigma_head(logits)
        sigma = torch.exp(log_sigma) + 1e-6
        return dist.Normal(mu, sigma)

    def get_action(self, obs: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.forward(obs)
        mean = distribution.mean
        raw_acts = distribution.rsample((n_samples,)).permute(1, 0, 2)  # shape: (batch, n_samples, act_dim)
        scaled_acts = torch.tanh(raw_acts)
        actions = scaled_acts * self.action_scale + self.action_bias
        # log_probs = distribution.log_prob(raw_acts)
        expanded_dist = dist.Normal(distribution.loc.unsqueeze(1), distribution.scale.unsqueeze(1))
        log_probs = expanded_dist.log_prob(raw_acts)
        log_probs -= torch.log(self.action_scale * (1 - scaled_acts.pow(2)) + 1e-6)
        log_probs = log_probs.sum(2, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return actions, log_probs, mean, raw_acts
    
    def get_log_probs(self, obs:torch.Tensor, raw_action:torch.Tensor) -> torch.Tensor:
        distribution = self.forward(obs)
        scaled_acts = torch.tanh(raw_action)
        log_probs = distribution.log_prob(raw_action)
        log_probs -= torch.log(self.action_scale * (1 - scaled_acts.pow(2)) + 1e-6)
        return log_probs.sum(-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self,
                 input_dim:int,
                 base_layers:List[int],
                 head_layers:List[int],
                 hidden_dim:int,
                 embedding_dim: int,
                 lr:float,
                 activation_fct:str,
                 layer_norm:bool=False,
                 output_dim:int=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        layers = []
        prev_dim = self.input_dim

        for layer_dim in base_layers+ [hidden_dim]:#[128,64,64,hidden_dim]
            layers.append(nn.Linear(prev_dim, layer_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(layer_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = layer_dim

        self.psi = nn.Sequential(*layers) #base network psi:SxA -> R^hidden_dim

        self.cos_weight = nn.Parameter(torch.empty(embedding_dim,hidden_dim))
        nn.init.xavier_uniform_(self.cos_weight)
        self.cos_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer('cos_idx', torch.arange(embedding_dim).float()) # index of cosine basis function

        layers = []
        prev_dim = hidden_dim

        for layer_dim in head_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(layer_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = layer_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.f_head = nn.Sequential(*layers) # output head f:hidden_dim -> output_dim

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, state:torch.Tensor, action:torch.Tensor, tau:torch.Tensor) -> torch.Tensor:
        sa = torch.cat((state,action), dim=-1)
        psi_sa = self.psi(sa)                                              # (B, H)
        cos = torch.cos(math.pi * tau.unsqueeze(-1) * self.cos_idx)                 # (B, N, E)
        phi = F.relu(cos @ self.cos_weight + self.cos_bias)                      # (B, N, H)
        h = psi_sa.unsqueeze(1) * phi                                     # (B, N, H)
        return self.f_head(h).squeeze(-1)            
        
    
class MPO_Agent():
    def __init__(self,cfg:Dict):
        self.cfg = cfg

        self.obs_dim = self.cfg.get('environment',{}).get('obs_dim')
        self.act_dim = self.cfg.get('environment',{}).get('act_dim')
        self.act_lim = self.cfg.get('environment',{}).get('act_lim', 1.0)
        self.n_envs = self.cfg.get('environment',{}).get('n_envs')
        self.reward_scale = self.cfg.get('environment',{}).get('reward_scale', 1.0)
        self.device = self.cfg.get('environment',{}).get('device', 'cpu')
        self.dt = self.cfg.get('environment',{}).get('dt', 0.1)

        self.gamma = self.cfg.get('agent', {}).get('params',{}).get('gamma', 0.99)
        self.tau = self.cfg.get('agent', {}).get('params',{}).get('tau', 0.95)
        self.policy_samples = self.cfg.get('agent', {}).get('params',{}).get('policy_samples', 1)
        self.e_step_epsilon = self.cfg.get('agent', {}).get('params',{}).get('e_step_epsilon', 1)
        self.n_temp_dual_steps = self.cfg.get('agent', {}).get('params',{}).get('n_temp_dual_steps', 200)
        self.m_step_epsilon_mu = self.cfg.get('agent', {}).get('params',{}).get('m_step_epsilon_mu', 0.1)
        self.m_step_epsilon_sigma = self.cfg.get('agent', {}).get('params',{}).get('m_step_epsilon_sigma', 1e-4)
        self.n_kl_dual_steps = self.cfg.get('agent', {}).get('params',{}).get('n_kl_dual_steps', 100)

        self.policy_layers = self.cfg.get('agent',{}).get('policy',{}).get('hidden_layers',[])
        self.policy_lr = self.cfg.get('agent',{}).get('policy',{}).get('lr', 0.001)
        self.policy_actv_fct = self.cfg.get('agent',{}).get('policy',{}).get('act_fct', 'relu')
        self.policy_layer_norm = self.cfg.get('agent',{}).get('policy',{}).get('layer_norm', False)
        self.policy_gradient_clipping = self.cfg.get('agent',{}).get('policy',{}).get('gradient_clip', None)

        self.critic_psi_layers = self.cfg.get('agent',{}).get('critic',{}).get('psi_layers',[])
        self.critic_f_layers = self.cfg.get('agent',{}).get('critic',{}).get('f_layers', [])
        self.critic_lr = self.cfg.get('agent',{}).get('critic',{}).get('lr', 0.001)
        self.critic_actv_fct = self.cfg.get('agent',{}).get('critic',{}).get('act_fct', 'relu')
        self.critic_layer_norm = self.cfg.get('agent',{}).get('critic',{}).get('layer_norm', False)
        self.critic_gradient_clipping = self.cfg.get('agent',{}).get('critic',{}).get('gradient_clip', None)
        self.critic_embedding_dim = self.cfg.get('agent',{}).get('critic',{}).get('embedding_dim', None)
        self.critic_hidden_dim = self.cfg.get('agent',{}).get('critic',{}).get('hidden_dim', None)
        self.critic_kappa = self.cfg.get('agent',{}).get('critic',{}).get('kappa', 1.0)
        self.critic_n_quantiles = self.cfg.get('agent',{}).get('critic',{}).get('n_quantiles', 1)
        self.critic_risk_type = self.cfg.get('agent',{}).get('critic',{}).get('risk', {}).get('beta_type','neutral')
        self.critic_risk_param = self.cfg.get('agent',{}).get('critic',{}).get('risk', {}).get('beta_param',None)

        
        self.interactions = self.cfg.get('training',{}).get('max_interactions', 1_000_000)
        self.training_steps = torch.ceil(torch.tensor(self.interactions/self.n_envs)).int()
        self.warm_up_steps = self.cfg.get('training',{}).get('warm_up', 1_000) #training has started, ramp up LR over these nb of steps -> stable grads
        self.learning_starts = self.cfg.get('training',{}).get('learning_starts', 10_000) #don't take any gradient for these first n steps
        self.save_checkpoint_rate = self.cfg.get('training',{}).get('save_checkpoint_rate', 500)
        self.utd_ratio = self.cfg.get('training',{}).get('utd_ratio', 1)
        self.policy_learning_starts = self.cfg.get('training',{}).get('policy_learning_starts', 0)

        self.buffer_sz = self.cfg.get('buffer', {}).get('buffer_size', 1_000_000)
        self.batch_sz = self.cfg.get('buffer', {}).get('batch_size', 256)
        self.td_horizon = self.cfg.get('buffer', {}).get('td_horizon', 1)
        self.save_buffer = self.cfg.get('buffer',{}).get('save_buffer', False) 

        self._init_buffer()
        self._init_models()

        #dual pb init:
        self.log_eta = torch.tensor(1.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_temp_optimizer = optim.Adam([self.log_eta], lr=1e-2)

        # self.log_alpha_mu = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_mu = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False)
        # self.dual_kl_mu_optimizer = optim.Adam([self.log_alpha_mu], lr=1e-2)

        # self.log_alpha_sigma = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_sigma = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False)
        # self.dual_kl_sigma_optimizer = optim.Adam([self.log_alpha_sigma], lr=1e-2)

        self.critic_loss = []
        self.policy_loss = []
        self.mean_q_value = []

    def _init_buffer(self) -> None:
        self.buffer = Buffer(self.cfg)

        print("[INFO]: Memory class initialized")

    def _init_models(self) -> None:
        #--- Policy ----
        self.policy = Actor(input_dim=self.obs_dim,
                            output_dim=self.act_dim,
                            action_limit=self.act_lim,  # Assuming symmetric action bounds
                            hidden_dims=self.policy_layers,
                            lr=self.policy_lr,
                            activation_fct=self.policy_actv_fct,
                            layer_norm=self.policy_layer_norm).to(self.device)
        init_model_weights(self.policy,self.policy_actv_fct)

        self.target_policy = copy.deepcopy(self.policy)
        #Targets only updated via polyak interpolation, no need to track grads
        for p in self.target_policy.parameters():
            p.requires_grad = False
        
        #--- Critic IQN ---
        self.q_function = Critic(self.obs_dim + self.act_dim,
                                 self.critic_psi_layers,
                                 self.critic_f_layers,
                                 self.critic_hidden_dim,
                                 self.critic_embedding_dim,
                                 self.critic_lr,
                                 self.critic_actv_fct,
                                 self.critic_layer_norm).to(self.device)
        init_model_weights(self.q_function, self.critic_actv_fct)
        
        self.target_q = copy.deepcopy(self.q_function)
        for p in self.target_q.parameters():
            p.requires_grad = False
        
        
        print("[INFO]: Models initialized")

    def _train(self) -> None:
        self.policy.train()
        self.target_policy.train()
        self.q_function.train()
        self.target_q.train()
        
    
    def _eval(self) -> None:
        self.policy.eval()
        self.target_policy.eval()
        self.q_function.eval()
        self.target_q.eval()
        

    def update_critic(self,
                      batch_data:Dict[str,torch.Tensor]) -> None:
        obs = batch_data['obs'][:, -1, :]
        acts = batch_data['acts'][:, -1, :]
        next_obs = batch_data['next_obs'][:, -1, :]

        with torch.no_grad():
            next_action, _, _, _ = self.target_policy.get_action(obs=next_obs)
            next_action = next_action.squeeze(1)

            # Sample target quantile levels and get distributional targets
            tau_prime = torch.rand(self.batch_sz, self.critic_n_quantiles, device=self.device)
            z_target = self.target_q(next_obs, next_action, tau_prime)  # (B, N_prime)

            
            y = z_target
            for k in range(self.td_horizon - 1, -1, -1):
                termination = batch_data['term'][:, k, :]  # (B, 1)
                reward = batch_data['r'][:, k, :]     # (B, 1)
                y = reward * self.dt + (self.gamma ** self.dt) * (~termination) * y  # (B, N_prime)

        # Sample current quantile levels and predict
        tau = torch.rand(self.batch_sz, self.critic_n_quantiles, device=self.device)
        z_pred = self.q_function(obs, acts, tau)  # (B, N)

        # Pairwise TD errors: (B, N_prime, N)
        td_errors = y.unsqueeze(2) - z_pred.unsqueeze(1)
        critic_loss = quantile_huber_loss(td_errors, tau, self.critic_kappa)

        self.q_function.optimizer.zero_grad()
        critic_loss.backward()
        if self.critic_gradient_clipping:
            nn.utils.clip_grad_norm_(self.q_function.parameters(), self.critic_gradient_clipping)
        self.q_function.optimizer.step()

        self.critic_loss.append(critic_loss.item())

        with torch.no_grad():
            tau_diag = torch.rand(self.batch_sz, self.critic_n_quantiles, device=self.device)
            self.mean_q_value.append(
                self.q_function(obs, acts, tau_diag).mean(dim=-1).mean().item()
            )

    def solve_temp_dual(self, q_samples:torch.Tensor, epsilon:float, n_dual_steps:int=200) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sz, n_samples = q_samples.shape
        q_values = q_samples.detach()

        with torch.enable_grad():
            for _ in range(n_dual_steps):
                self.dual_temp_optimizer.zero_grad()
                eta = self.log_eta.exp()
                dual_loss = eta * epsilon + eta * (torch.logsumexp(q_values / eta, dim=-1) - torch.log(torch.tensor(n_samples, device=self.device))).mean()
                dual_loss.backward()
                self.dual_temp_optimizer.step()
                self.log_eta.data.clamp_(-4.0, 4.0)  # keep eta in [~0.02, ~55], prevents q/eta overflow

        eta_star = self.log_eta.exp().detach()

        weights = torch.softmax(q_values/eta_star, dim=-1)
        return eta_star, weights

    def solve_kl_dual(self, kl_value: torch.Tensor, epsilon: float, n_dual_steps: int = 30) -> torch.Tensor:
        log_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        dual_optimizer = optim.Adam([log_alpha], lr=1e-2)
        kl = kl_value.detach()

        with torch.enable_grad():
            for _ in range(n_dual_steps):
                dual_optimizer.zero_grad()
                alpha = log_alpha.exp()
                dual_loss = alpha * (epsilon - kl)
                dual_loss.backward()
                dual_optimizer.step()

        return log_alpha.exp().detach()
    
    def _apply_risk_distortion(self, tau:torch.Tensor) -> torch.Tensor:
        if self.critic_risk_type == 'neutral':
            return tau
        elif self.critic_risk_type == 'cvar':
            # CVaR(eta): sample tau ~ U([0, eta]) -> shift mass to lower tail
            return self.critic_risk_param * tau
        elif self.critic_risk_type == 'wang':
            # Wang(eta): Phi(Phi^-1(tau) + eta)
            normal = dist.Normal(0.0, 1.0)
            return normal.cdf(normal.icdf(tau.clamp(1e-6, 1 - 1e-6)) + self.critic_risk_param)
        else:
            raise ValueError(f"Unknown risk distortion: {self.critic_risk_type}") 

    def e_step(self,
               batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch_data['obs'][:, -1, :]  # (B, obs_dim)

        bounded_actions, _, _, raw_actions = self.target_policy.get_action(obs, n_samples=self.policy_samples)

        obs_exp = obs.unsqueeze(1).expand(-1, self.policy_samples, -1).reshape(-1, obs.shape[-1])
        acts_flat = bounded_actions.reshape(-1, bounded_actions.shape[-1])

        #Q ~ expected value over quantile samples from the distributional critic
        # tau_eval = torch.rand(obs_exp.shape[0], self.critic_n_quatiles, device=self.device)
        tau_eval = self._apply_risk_distortion(torch.rand(obs_exp.shape[0],
                                                          self.critic_n_quantiles,
                                                          device=self.device))
        z_vals = self.target_q(obs_exp, acts_flat, tau_eval) # (B*K, N_tau)
        q_vals = z_vals.mean(dim=-1).reshape(self.batch_sz, self.policy_samples)

        eta, weights = self.solve_temp_dual(q_vals, self.e_step_epsilon, self.n_temp_dual_steps)
        return raw_actions, weights, eta

    def m_step(self,
               obs: torch.Tensor,
               sampled_actions: torch.Tensor,
               weights: torch.Tensor) -> None:
        # Weighted NLL (supervised fit to E-step distribution)
        curr_d = self.policy.forward(obs)
        log_probs = dist.Normal(curr_d.loc.unsqueeze(1), curr_d.scale.unsqueeze(1)).log_prob(sampled_actions)  # (batch_sz, policy_samples, act_dim)
        nll = -(weights.detach() * log_probs.sum(-1)).sum(-1).mean()

        # Decoupled KL constraints
        old_d = self.target_policy.forward(obs)
        mu_old, sigma_old = old_d.loc.detach(), old_d.scale.detach()

        # D_KL^μ: sg on sigma_theta — gradients flow only through mu_theta
        kl_mu = dist.kl_divergence(
            dist.Normal(curr_d.loc, curr_d.scale.detach()),
            dist.Normal(mu_old, curr_d.scale.detach())
        ).sum(-1).mean()

        # D_KL^Σ: sg on mu_theta — gradients flow only through sigma_theta
        kl_sigma = dist.kl_divergence(
            dist.Normal(curr_d.loc.detach(), curr_d.scale),
            dist.Normal(curr_d.loc.detach(), sigma_old)
        ).sum(-1).mean()

        # alpha_mu = self.solve_kl_dual(kl_mu, self.m_step_epsilon_mu, self.n_kl_dual_steps)
        # alpha_sigma = self.solve_kl_dual(kl_sigma, self.m_step_epsilon_sigma, self.n_kl_dual_steps)

        # No loop needed
        self.alpha_mu = torch.clamp(kl_mu.detach() - self.m_step_epsilon_mu, min=0.1)
        self.alpha_sigma = torch.clamp(kl_sigma.detach() - self.m_step_epsilon_sigma, min=0.1)

        policy_loss = (nll
                       + self.alpha_mu    * (kl_mu    - self.m_step_epsilon_mu)
                       + self.alpha_sigma * (kl_sigma - self.m_step_epsilon_sigma))

        self.policy.optimizer.zero_grad()
        policy_loss.backward()

        if self.policy_gradient_clipping:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_gradient_clipping)

        self.policy.optimizer.step()
        self.policy_loss.append(policy_loss.item())

    def _update_targets(self, critic_only:bool=False) -> None:
        if not critic_only:
            for p, p_tgt in zip(self.policy.parameters(), self.target_policy.parameters()):
                p_tgt.data.lerp_(p.data, 1 - self.tau)

        for p, p_tgt in zip(self.q_function.parameters(), self.target_q.parameters()):
            p_tgt.data.lerp_(p.data, 1 - self.tau)

    def update(self, critic_only:bool=False) -> None:
        batch = self.buffer.sample(self.batch_sz, self.td_horizon)
        # t_0_critic = time.perf_counter()
        self.update_critic(batch)
        # print(f"critic update duration: {time.perf_counter() - t_0_critic}")
        
        if not critic_only:
            obs = batch['obs'][:, -1, :]
            with torch.no_grad():
                # t_0_e_step = time.perf_counter()
                sampled_actions, weights, _ = self.e_step(batch)
                # print(f"e-step duration: {time.perf_counter() - t_0_e_step}")
                
            # t_0_m_step = time.perf_counter()
            self.m_step(obs, sampled_actions, weights)
            # print(f"m-step duration: {time.perf_counter() - t_0_m_step}")
        self._update_targets(critic_only)
        

    def train_agent(self, envs: gym.Env, run_tag: str, log_dir: str = "train_logs") -> None:
        current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_folder = os.path.join(log_dir, f"{run_tag}_{current_date}")
        checkpoints_folder = os.path.join(experiment_folder, "checkpoints")
        os.makedirs(checkpoints_folder, exist_ok=True)

        with open(os.path.join(experiment_folder, "hyperparams.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)

        obs, _ = envs.reset(seed=self.cfg.get('environment', {}).get('seed', 42))

        episode_returns = torch.zeros(self.n_envs, device=self.device)
        episode_lengths = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)

        log_rows = []
        critic_only = True
        next_checkpoint_at = self.save_checkpoint_rate

        progress_bar = tqdm(range(self.training_steps))
        for step in progress_bar:
            self._eval()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action_t, old_log_probs_t, _, raw_actions_t = self.policy.get_action(obs_t)
                action_t = action_t.squeeze(1)

            next_obs, reward, terminated, truncated, _ = envs.step(action_t.cpu().numpy())

            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
            truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

            episode_returns += reward_t

            self.buffer.add_sample(obs=obs_t,
                                   actions=action_t,
                                   raw_actions=raw_actions_t,
                                   log_probs_mu=old_log_probs_t,
                                   next_obs=next_obs_t,
                                   rewards=reward_t * self.reward_scale,
                                   truncation=truncated_t, 
                                   termination=terminated_t)
            obs = next_obs
            episode_lengths += 1

            done = terminated_t | truncated_t
            if done.any():
                mean_return = episode_returns[done].mean().item()
                mean_length = episode_lengths[done].float().mean().item()
                progress_bar.set_postfix({
                    "return": f"{mean_return:.1f}",
                    "ep_len": f"{mean_length:.0f}",
                    "step": step,
                })

                log_rows.append({
                    "timestep": step * self.n_envs,
                    "mean_reward": reward_t[done].mean().item(),
                    "mean_return": mean_return,
                    "mean_episode_length": mean_length,
                    "policy_loss": self.policy_loss[-1] if self.policy_loss else float("nan"),
                    "critic_loss": self.critic_loss[-1] if self.critic_loss else float("nan"),
                    "eta": self.log_eta.data.exp(),
                    "alpha_mu": self.alpha_mu.data,
                    "alpha_sigma": self.alpha_sigma.data,
                })

                episode_returns[done] = 0.0
                episode_lengths[done] = 0

            if step >= (self.learning_starts + self.policy_learning_starts):
                critic_only = False

            if step >= self.learning_starts:
                self._train()
                for _ in range(self.utd_ratio):
                    self.update(critic_only)

            current_interaction = (step + 1) * self.n_envs
            if current_interaction >= next_checkpoint_at:
                self.save_checkpoint(current_interaction, checkpoints_folder)
                next_checkpoint_at = ((current_interaction // self.save_checkpoint_rate) + 1) \
                                    * self.save_checkpoint_rate

        envs.close()
        csv_path = os.path.join(experiment_folder, "performance.csv")
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        print(f"[INFO]: Training log saved to {csv_path}")

    def save_checkpoint(self, current_env_interaction: int, folder:str) -> None:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "policy_optimizer": self.policy.optimizer.state_dict(),

            "q_function": self.q_function.state_dict(),
            "target_q":   self.target_q.state_dict(),
            "q_optimizer": self.q_function.optimizer.state_dict(),

            "log_eta": self.log_eta.data,
            "alpha_mu": self.alpha_mu.data,
            "alpha_sigma": self.alpha_sigma.data,

            "dual_temp_optimizer": self.dual_temp_optimizer.state_dict(),

            "global_step": current_env_interaction,
        }
        torch.save(checkpoint, os.path.join(folder, f"checkpoint_{current_env_interaction}.pth"))


def main():
    env = gym.make_vec(args.task, args.n_envs)

    cwd = os.getcwd()
    config_path = os.path.join(cwd, 'config.yaml')
    cfg = load_config(config_path, args)
    cfg['environment']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['environment']['obs_dim'] = env.observation_space.shape[-1]
    cfg['environment']['act_dim'] = env.action_space.shape[-1]
    cfg['environment']['act_lim'] = [env.action_space.low.item(), env.action_space.high.item()]
    cfg['environment']['dt'] = env.unwrapped.envs[0].unwrapped.dt

    seed = cfg['environment'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent = MPO_Agent(cfg=cfg)
    log_dir = os.path.join(cwd, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    agent.train_agent(env, args.run_name, log_dir)
        
    
if __name__ == "__main__":
    main()