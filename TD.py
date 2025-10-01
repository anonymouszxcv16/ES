import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer

from dataclasses import dataclass
from typing import Callable
from torch.distributions import Normal

@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    target_update_rate: int = 250

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # SAC
    LOG_SIG_MAX: float = 2
    LOG_SIG_MIN: float = -20
    ACTION_BOUND_EPSILON: float = 1E-6
    alpha_sac: float = .01

    # TD3+BC
    lmbda: float = 0.1

    # CQL
    alpha_cql: float = .01

    # ES
    alpha_max: float = .2
    alpha_min: float = 0

    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4

# Layer normalization.
def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


# Huber.
def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


# Actor.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, args, LOG_SIG_MIN, LOG_SIG_MAX, ACTION_BOUND_EPSILON, hdim=256, activ=F.relu):
        super(Actor, self).__init__()

        self.activ = activ

        # Noisy linear.
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

        self.log_std = nn.Linear(hdim, action_dim)

        # SAC
        self.LOG_SIG_MIN = LOG_SIG_MIN
        self.LOG_SIG_MAX = LOG_SIG_MAX
        self.ACTION_BOUND_EPSILON = ACTION_BOUND_EPSILON

        self.args = args

    def forward(self, state, deterministic=False, return_log_prob=True):
        # Normalization.
        a = AvgL1Norm(self.l0(state))

        # Fully connected.
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))

        # Log prob.
        mean = self.l3(a)

        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob = log_prob.mean(1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob

# Critic.
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, args, hdim=256, activ=F.elu):
        super(Critic, self).__init__()

        self.activ = activ

        # Fully connected.
        self.q0 = nn.ParameterList([nn.Linear(state_dim + action_dim, hdim) for _ in range(args.N)])
        self.q1 = nn.ParameterList([nn.Linear(hdim, hdim) for _ in range(args.N)])
        self.q2 = nn.ParameterList([nn.Linear(hdim, hdim) for _ in range(args.N)])
        self.q3 = nn.ParameterList([nn.Linear(hdim, 1) for _ in range(args.N)])

        self.args = args

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q_values = []
        activations_std = 0

        # Ensemble.
        for i in range(self.args.N):
            # Normalization.
            q = AvgL1Norm(self.q0[i](sa))

            # Fully connected.
            q1 = self.activ(self.q1[i](q))
            q2 = self.activ(self.q2[i](q1))
            q = self.q3[i](q2)

            q_values.append(q)
            activations_std += (q1.std() + q2.std()) / 2

        return torch.cat([q_value for q_value in q_values], 1), activations_std / self.args.N

class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, args, hp=Hyperparameters()):
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp
        self.args = args

        # Environment.
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.args.device = self.device

        self.init()

    def init(self):
        self.actor = Actor(self.state_dim, self.action_dim, self.args, self.hp.LOG_SIG_MIN, self.hp.LOG_SIG_MAX, self.hp.ACTION_BOUND_EPSILON,
                           self.hp.actor_hdim, self.hp.actor_activ).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, self.args, self.hp.critic_hdim, self.hp.critic_activ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)
        self.checkpoint_actor = copy.deepcopy(self.actor)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        # Experience Replay
        self.replay_buffer = buffer.LAP(self.state_dim, self.action_dim, self.device, self.args, self.args.buffer_size, self.hp.batch_size, self.max_action,
                                        normalize_actions=True)

        if "CQL" in self.args.policy:
            self.cql_alpha = torch.tensor(1.0, requires_grad=True, device=self.device)
            self.cql_alpha_optimizer = torch.optim.Adam([self.cql_alpha], lr=1e-4)

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

        # Auto
        self.ee_inverse_sum = 0

        # Activations
        self.activations_std = 0

    def scaled_sigmoid(self, x):
        return self.hp.alpha_min + (self.hp.alpha_max - self.hp.alpha_min) / (1 + np.exp(-x))

    def compute_cql_loss(self, state, action):
        with torch.no_grad():
            # Actor
            actor, _ = self.actor_target(state, deterministic=False)
            q_values, _ = self.critic_target(state, actor).mean(1, keepdim=True)

            # Current Q-values (for data actions)
            y, _ = self.critic_target(state, action).mean(1, keepdim=True)

        # Q values actor log sum exp.
        penalty_target = torch.logsumexp(q_values, dim=1, keepdim=True)
        cql_loss = (penalty_target - y).mean()

        return cql_loss

    def select_action(self, state, use_checkpoint=False, use_exploration=True, deterministic=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            action, _ = self.actor(state, deterministic=deterministic)

            if use_exploration:
                noise = torch.randn_like(action) * (0 if "SAC" in self.args.policy else self.args.exploration_noise)
                action = action + noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # Update Critic.
        with torch.no_grad():
            # State next.
            next_action, next_action_log_prob = self.actor_target(next_state, deterministic=False if "SAC" in self.args.policy else True)

            noise = (torch.randn_like(action) * (0 if "SAC" in self.args.policy else self.hp.target_policy_noise)).clamp(-self.hp.noise_clip, self.hp.noise_clip)
            next_action = (next_action + noise).clamp(-1, 1)

            # Q-values
            Q_next, activations_std = self.critic_target(next_state, next_action)
            ee_value = Q_next.std(1, keepdim=True).mean().item()

            Q_target_next = Q_next.min(1, keepdim=True)[0]
            self.activations_std += activations_std

            # SAC
            entropy_bonus =  next_action_log_prob if "SAC" in self.args.policy else 0

            if self.args.auto_alpha == 1:
                ee_inverse = 1 / ee_value

                self.ee_inverse_sum += ee_inverse

                if (self.training_steps + self.args.timesteps_before_training) % self.args.auto_alpha_interval == 0:
                    ee_inverse_mean = self.ee_inverse_sum / self.args.auto_alpha_interval
                    self.args.alpha = self.scaled_sigmoid(ee_inverse_mean)

                    self.ee_inverse_sum = 0

            Q_target_next = Q_target_next - self.hp.alpha_sac * entropy_bonus - (self.args.alpha * ee_value if "EE" in self.args.policy else 0)
            Q_target = reward + not_done * self.args.discount * Q_target_next

        # TD loss.
        Q, _ = self.critic(state, action)
        td_loss = (Q - Q_target).abs()

        if "CQL" in self.args.policy:
            cql_loss = self.compute_cql_loss(state, action)

            critic_loss = .5 * td_loss + self.hp.alpha_cql * cql_loss
            critic_loss = LAP_huber(critic_loss)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        else:
            # Critic step.
            critic_loss = LAP_huber(td_loss)

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()


        # Update Actor.
        if self.training_steps % self.hp.policy_freq == 0:
            actor, _ = self.actor(state)
            Q, _ = self.critic(state, actor)
            actor_loss = -Q.mean()

            # BC
            if self.args.offline == 1 and ("BC" in self.args.policy):
                BC_loss = F.mse_loss(actor, action)
                actor_loss += self.hp.lmbda * Q.abs().mean().detach() * BC_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update Iteration
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())