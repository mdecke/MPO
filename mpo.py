import os
import copy
import json
import yaml
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
parser.add_argument('--ensemble', type=int, default=1, help="number of Q functions")
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

def init_model_weights(model:nn.Module, mean:float=0.0, std:float=0.1) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "weight" in name:
                nn.init.normal_(param, mean=mean, std=std)
            elif "bias" in name:
                nn.init.normal_(param, mean=mean, std=std)

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
        'ensemble' : ('agent', 'critic', 'ensemble')
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
        self.rewards = torch.empty((self.N, self.envs, 1), dtype=torch.float32, device=self.device)
        self.truncation = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=self.device)
        self.termination = torch.empty((self.N, self.envs, 1), dtype=torch.bool, device=self.device)
        self.infos = torch.empty((self.N, self.envs), dtype=torch.bool, device=self.device)
        
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

class Critic(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dims:List[int],
                 lr:float,
                 activation_fct:str,
                 layer_norm:bool=False,
                 output_dim:int=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation_fct))
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

        self.obs_dim = self.cfg.get('environment',{}).get('obs_dim')
        self.act_dim = self.cfg.get('environment',{}).get('act_dim')
        self.act_lim = self.cfg.get('environment',{}).get('act_lim', 1.0)
        self.n_envs = self.cfg.get('environment',{}).get('n_envs')
        self.reward_scale = self.cfg.get('training',{}).get('reward_scale', 1.0)
        self.device = self.cfg.get('environment',{}).get('device', 'cpu')

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

        self.critic_layers = self.cfg.get('agent',{}).get('critic',{}).get('hidden_layers',[])
        self.critic_lr = self.cfg.get('agent',{}).get('critic',{}).get('lr', 0.001)
        self.critic_actv_fct = self.cfg.get('agent',{}).get('critic',{}).get('act_fct', 'relu')
        self.critic_layer_norm = self.cfg.get('agent',{}).get('critic',{}).get('layer_norm', False)
        self.critic_gradient_clipping = self.cfg.get('agent',{}).get('critic',{}).get('gradient_clip', None)
        self.n_critics = self.cfg.get('agent',{}).get('critic',{}).get('ensemble',1)
        self.p_bootstrap = self.cfg.get('agent', {}).get('critic', {}).get('bootstrap_p', 0.9)
        
        self.interactions = self.cfg.get('training',{}).get('max_interactions', 1_000_000)
        self.training_steps = torch.ceil(torch.tensor(self.interactions/self.n_envs)).int()
        self.warm_up_steps = self.cfg.get('training',{}).get('warm_up', 1_000) #training has started, ramp up LR over these nb of steps -> stable grads
        self.learning_starts = self.cfg.get('training',{}).get('learning_starts', 10_000) #don't take any gradient for these first n steps
        self.save_checkpoint_rate = self.cfg.get('training',{}).get('save_checkpoint_rate', 500)
        self.utd_ratio = self.cfg.get('training',{}).get('utd_ratio', 1)
        self.policy_learning_start = self.cfg.get('training',{}).get('policy_learning_start', 0)

        self.buffer_sz = self.cfg.get('buffer', {}).get('buffer_size', 1_000_000)
        self.batch_sz = self.cfg.get('buffer', {}).get('batch_size', 256)
        self.td_horizon = self.cfg.get('buffer', {}).get('td_horizon', 1)
        self.save_buffer = self.cfg.get('buffer',{}).get('save_buffer', False) 

        self._init_buffer()
        self._init_models()

        #dual pb init:
        self.log_eta = torch.tensor(1.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_temp_optimizer = optim.Adam([self.log_eta], lr=1e-2)

        self.log_alpha_mu = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_kl_mu_optimizer = optim.Adam([self.log_alpha_mu], lr=1e-2)

        self.log_alpha_sigma = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.dual_kl_sigma_optimizer = optim.Adam([self.log_alpha_sigma], lr=1e-2)

        self.evaluted_q = torch.empty((self.batch_sz, self.policy_samples), dtype=torch.float32, device=self.device)

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
        init_model_weights(self.policy)

        self.target_policy = copy.deepcopy(self.policy)
        #Targets only updated via polyak interpolation, no need to track grads
        for p in self.target_policy.parameters():
            p.requires_grad = False
        
        #--- Critic ensemble ---
        self.q_functions = nn.ModuleList([Critic(input_dim=self.obs_dim + self.act_dim,
                                                 hidden_dims=self.critic_layers,
                                                 lr=self.critic_lr,
                                                 activation_fct=self.critic_actv_fct,
                                                 layer_norm=self.critic_layer_norm) for _ in range(self.n_critics)]).to(self.device)
        for q in self.q_functions:
            init_model_weights(q)
        
        self.target_qs = copy.deepcopy(self.q_functions)
        for p in self.target_qs.parameters():
            p.requires_grad = False
        
        
        print("[INFO]: Models initialized")

        self.policy = torch.compile(self.policy)
        self.target_policy = torch.compile(self.target_policy)
        self.q_functions = nn.ModuleList([torch.compile(q) for q in self.q_functions])
        self.target_qs = nn.ModuleList([torch.compile(q) for q in self.target_qs])

        print("[INFO]: Models compiled")

    def _train(self) -> None:
        self.policy.train()
        self.target_policy.train()
        for q, target_q in zip(self.q_functions, self.target_qs):
            q.train()
            target_q.train()
        
    
    def _eval(self) -> None:
        self.policy.eval()
        self.target_policy.eval()
        for q, target_q in zip(self.q_functions, self.target_qs):
            q.eval()
            target_q.eval()
        

    def update_critic(self,
                      batch_data:Dict[str,torch.Tensor]) -> None:
        next_action, _, _, _ = self.target_policy.get_action(obs=batch_data["next_obs"][:,-1,:])
        next_action = next_action.squeeze(1)
        q_target = self.aggregation_operator(state=batch_data['next_obs'][:,-1,:],
                                             action=next_action,
                                             critics=self.target_qs,
                                             mode='min_subset',
                                             subset_size=2)

        #temporal diff
        y = q_target
        for k in range(self.td_horizon-1,-1,-1):
            termination = batch_data['term'][:,k,:]
            reward =batch_data['r'][:,k,:]
            y = reward + self.gamma * (~termination) * y

        masks = torch.bernoulli(torch.full(size=(self.batch_sz,self.n_critics), fill_value=1), p=self.p_bootstrap).bool() #bootstrap over batch and ensemble member: different q functions see different transitions
        losses_this_step = []
        for k,q in enumerate(self.q_functions):
            m = masks[:,k]
            if m.sum() == 0:
                continue #safeguard: if all mask over data is all 0

            q_value = q.forward(state=batch_data['obs'][m,-1,:], action=batch_data['acts'][m,-1,:])
            critic_loss = F.mse_loss(q_value, y[m])
            
            q.optimizer.zero_grad()
            critic_loss.backward()
            if self.critic_gradient_clipping:
                nn.utils.clip_grad_norm_(q.parameters(), self.critic_gradient_clipping)
            q.optimizer.step()

            losses_this_step.append(critic_loss.item())

        # ---- Logging ----
        if losses_this_step:
            self.critic_loss.append(float(np.mean(losses_this_step)))

        with torch.no_grad():
            q_mean_diag = self.aggregation_operator(
                state=batch_data['obs'][m,-1,:], action=batch_data['acts'][m,-1,:], critics=self.q_functions, mode='mean'
            )
            self.mean_q_value.append(q_mean_diag.mean().item())

    def aggregation_operator(self,
                             state:torch.Tensor,
                             action: torch.Tensor,
                             critics:nn.ModuleList,
                             mode:str='mean',
                             beta:float=1.0,
                             subset_size:int=2) -> torch.Tensor:
        next_q_values = torch.stack([q.forward(state,action) for q in critics],dim=0) #shape: (n_critis x batch_sz x output_dim)
        if mode == "mean":
            return next_q_values.mean(dim=0) #mean across ensemble dim
        elif mode == "LCB":
            return next_q_values.mean(dim=0) - beta*(next_q_values.std(dim=0) + 1e-6) #safeguard for collapsing std
        elif mode == "UCB":
            return next_q_values.mean(dim=0) + beta*(next_q_values.std(dim=0) + 1e-6)
        elif mode == "min_subset":
            idx = torch.randperm(next_q_values.shape[0], device=self.device)[:subset_size]
            return next_q_values[idx].min(dim=0).values
        elif mode == "median":
            return next_q_values.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {mode}")

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

    def e_step(self,
               batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch_data['obs'][:, -1, :]  # (batch_sz, obs_dim)

        bounded_actions, _, _, raw_actions = self.target_policy.get_action(obs, n_samples=self.policy_samples)

        # Expand obs and flatten bounded actions for critic (critic trained on post-tanh actions)
        obs_exp = obs.unsqueeze(1).expand(-1, self.policy_samples, -1).reshape(-1, obs.shape[-1])
        acts_flat = bounded_actions.reshape(-1, bounded_actions.shape[-1])

        # self.evaluted_q = self.target_critic.forward(state=obs_exp, action=acts_flat).reshape(self.batch_sz, self.policy_samples)
        self.evaluted_q = self.aggregation_operator(state=obs_exp, action=acts_flat, critics=self.target_qs, mode='mean').reshape(self.batch_sz, self.policy_samples)

        eta, weights = self.solve_temp_dual(self.evaluted_q, self.e_step_epsilon, self.n_temp_dual_steps)
        
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
            dist.Normal(mu_old, sigma_old)
        ).sum(-1).mean()

        # D_KL^Σ: sg on mu_theta — gradients flow only through sigma_theta
        kl_sigma = dist.kl_divergence(
            dist.Normal(curr_d.loc.detach(), curr_d.scale),
            dist.Normal(mu_old, sigma_old)
        ).sum(-1).mean()

        # alpha_mu = self.solve_kl_dual(kl_mu, self.m_step_epsilon_mu, self.n_kl_dual_steps)
        # alpha_sigma = self.solve_kl_dual(kl_sigma, self.m_step_epsilon_sigma, self.n_kl_dual_steps)

        # No loop needed
        alpha_mu = torch.clamp(kl_mu.detach() - self.m_step_epsilon_mu, min=0.0)
        alpha_sigma = torch.clamp(kl_sigma.detach() - self.m_step_epsilon_sigma, min=0.0)

        policy_loss = (nll
                       + alpha_mu    * (kl_mu    - self.m_step_epsilon_mu)
                       + alpha_sigma * (kl_sigma - self.m_step_epsilon_sigma))

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

        for p, p_tgt in zip(self.q_functions.parameters(), self.target_qs.parameters()):
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
        self._update_targets()
        

    def train_agent(self, envs: gym.Env, run_tag: str, log_dir: str = "train_logs") -> None:
        current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_folder = os.path.join(log_dir, f"{run_tag}_{current_date}")
        checkpoints_folder = os.path.join(experiment_folder, "checkpoints")
        os.makedirs(checkpoints_folder, exist_ok=True)

        with open(os.path.join(experiment_folder, "hyperparams.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)

        self._train()
        obs, _ = envs.reset(seed=self.cfg.get('environment', {}).get('seed', 42))

        episode_returns = torch.zeros(self.n_envs, device=self.device)
        episode_lengths = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)

        log_rows = []
        critic_only = True

        progress_bar = tqdm(range(self.training_steps))
        for step in progress_bar:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action_t, _, _, _ = self.policy.get_action(obs_t)
                action_t = action_t.squeeze(1)

            next_obs, reward, terminated, truncated, _ = envs.step(action_t.cpu().numpy())

            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
            truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

            episode_returns += reward_t

            self.buffer.add_sample(obs_t, action_t, next_obs_t, reward_t * self.reward_scale, truncated_t, terminated_t)
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
                    "mean_ep_len": mean_length,
                    "policy_loss": self.policy_loss[-1] if self.policy_loss else float("nan"),
                    "critic_loss": self.critic_loss[-1] if self.critic_loss else float("nan"),
                })
                episode_returns[done] = 0.0
                episode_lengths[done] = 0
            if step >= (self.learning_starts + self.policy_learning_start):
                critic_only = False
            
            if step >= self.learning_starts:
                for _ in range(self.utd_ratio):
                    self.update(critic_only)
            
            current_interaction = step * self.n_envs
            if current_interaction % self.save_checkpoint_rate == 0:
                self.save_checkpoint(current_interaction, checkpoints_folder)

        envs.close()
        csv_path = os.path.join(experiment_folder, "performance.csv")
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        print(f"[INFO]: Training log saved to {csv_path}")

    def save_checkpoint(self, current_env_interaction: int, folder:str) -> None:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "policy_optimizer": self.policy.optimizer.state_dict(),

            "q_functions": [q.state_dict() for q in self.q_functions],
            "target_qs":   [q.state_dict() for q in self.target_qs],
            "q_optimizers": [q.optimizer.state_dict() for q in self.q_functions],

            "log_eta": self.log_eta.data,
            "log_alpha_mu": self.log_alpha_mu.data,
            "log_alpha_sigma": self.log_alpha_sigma.data,

            "dual_temp_optimizer": self.dual_temp_optimizer.state_dict(),
            "dual_kl_mu_optimizer": self.dual_kl_mu_optimizer.state_dict(),
            "dual_kl_sigma_optimizer": self.dual_kl_sigma_optimizer.state_dict(),

            "global_step": current_env_interaction,
            "n_critics" : self.n_critics,

            # "torch_rng_state": torch.random.get_rng_state(),
            # "numpy_rng_state": np.random.get_state(),
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