import copy

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

	# LAP
	alpha: float = 0.4
	min_priority: float = 1

	# TD3+BC
	lmbda: float = 0.1

	# Checkpointing
	max_eps_when_checkpointing: int = 20
	steps_before_checkpointing: int = 75e4
	reset_weight: float = 0.9

	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-4

	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-4

	# Average Model
	average_hdim: int = 256
	average_activ: Callable = F.elu
	average_lr: float = 3e-4

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
	def __init__(self, state_dim, action_dim, args, LOG_SIG_MIN, LOG_SIG_MAX, ACTION_BOUND_EPSILON, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		# Noisy linear.
		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)

		self.log_std = nn.Linear(hdim, action_dim)

		# SAC
		self.LOG_SIG_MIN = LOG_SIG_MIN
		self.LOG_SIG_MAX = LOG_SIG_MAX
		self.ACTION_BOUND_EPSILON = ACTION_BOUND_EPSILON

		self.args = args

	def forward(self, state, zs, deterministic=False, return_log_prob=True):
		# Normalization.
		a = AvgL1Norm(self.l0(state))

		# Embedding.
		a = torch.cat([a, zs], 1)

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


# Encoder.
class Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, args, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)

		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)

		self.args = args

	def zs(self, state):
		# Fully connected.
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))

		# Normalization.
		zs = AvgL1Norm(self.zs3(zs))

		return zs

	def zsa(self, zs, action):
		# Fully connected.
		zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)

		return zsa


# Critic.
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, args, zs_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ

		# Fully connected.
		self.q0 = nn.ParameterList([nn.Linear(state_dim + action_dim, hdim) for _ in range(args.N)])
		self.q1 = nn.ParameterList([nn.Linear(2 * zs_dim + hdim, hdim) for _ in range(args.N)])
		self.q2 = nn.ParameterList([nn.Linear(hdim, hdim) for _ in range(args.N)])
		self.q3 = nn.ParameterList([nn.Linear(hdim, 1) for _ in range(args.N)])

		self.args = args

	def forward(self, state, action, zsa, zs):
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q_values = []

		# Ensemble.
		for i in range(self.args.N):
			# Normalization.
			q = self.q0[i](sa)
			q = AvgL1Norm(q)

			# Embedding.
			q = torch.cat([q, embeddings], 1)

			# Fully connected.
			q = self.activ(self.q1[i](q))
			q = self.activ(self.q2[i](q))
			q = self.q3[i](q)

			q_values.append(q)

		return torch.cat([q_value for q_value in q_values], 1)


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
		# Actor
		self.actor = Actor(self.state_dim, self.action_dim, self.args, self.hp.LOG_SIG_MIN, self.hp.LOG_SIG_MAX, self.hp.ACTION_BOUND_EPSILON, self.hp.zs_dim,
						   self.hp.actor_hdim, self.hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)
		self.checkpoint_actor = copy.deepcopy(self.actor)

		# Critic
		self.critic = Critic(self.state_dim, self.action_dim, self.args, self.hp.zs_dim, self.hp.critic_hdim, self.hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)

		# Encoder
		self.encoder = Encoder(self.state_dim, self.action_dim, self.args, self.hp.zs_dim, self.hp.enc_hdim, self.hp.enc_activ).to(self.device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)

		# Experience Replay
		self.replay_buffer = buffer.LAP(self.state_dim, self.action_dim, self.device, self.args, self.args.buffer_size, self.hp.batch_size, self.max_action,
										normalize_actions=True, prioritized="TD7" in self.args.policy)

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

	def select_action(self, state, use_checkpoint=False, use_exploration=True, deterministic=True):
		with torch.no_grad():
			state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

			zs = self.fixed_encoder.zs(state)
			action, _ = self.actor(state, zs, deterministic=deterministic)

			if use_checkpoint and "TD7" in self.args.policy:
				zs = self.checkpoint_encoder.zs(state)
				action, _ = self.checkpoint_actor(state, zs, deterministic=deterministic)

			if use_exploration:
				noise = torch.randn_like(action) * (0 if "SAC" in self.args.policy else self.args.exploration_noise)
				action = action + noise

			return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

	def train(self):
		self.training_steps += 1
		state, action, next_state, reward, not_done = self.replay_buffer.sample(prioritized="TD7" in self.args.policy)

		# Update Encoder.
		if "TD7" in self.args.policy:
			with torch.no_grad():
				next_zs = self.encoder.zs(next_state)

			zs = self.encoder.zs(state)
			pred_zs = self.encoder.zsa(zs, action)

			# Loss.
			encoder_loss = F.mse_loss(pred_zs, next_zs)
			self.encoder_optimizer.zero_grad()
			encoder_loss.backward()
			self.encoder_optimizer.step()

		# Update Critic.
		with torch.no_grad():
			# State next.
			fixed_target_zs = self.fixed_encoder_target.zs(next_state)
			next_action, next_action_log_prob = self.actor_target(next_state, fixed_target_zs, deterministic=False if "SAC" in self.args.policy else True)

			# Noise.
			noise = (torch.randn_like(action) * (0 if "SAC" in self.args.policy else self.hp.target_policy_noise)).clamp(-self.hp.noise_clip, self.hp.noise_clip)

			# Action
			next_action = (next_action + noise).clamp(-1, 1)

			# Embedding
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
			fixed_zs = self.fixed_encoder.zs(state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

			# Q-values
			Q_next = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs)

			# TD3
			Q_target_next = Q_next.min(1, keepdim=True)[0]

			if "TD7" in self.args.policy:
				# TD7
				Q_target_next = Q_target_next.clamp(self.min_target, self.max_target)

			# SAC
			entropy_bonus = -self.args.alpha_sac * next_action_log_prob if "SAC" in self.args.policy else 0

			# SQT
			std_q_target = self.args.alpha * Q_next.std(dim=1).mean() if "SQT" in self.args.policy else 0

			# Ablation
			std_q_target = self.args.alpha * Q_next.std(dim=1).unsqueeze(1) if "SQTA" in self.args.policy else 0

			with torch.no_grad():
				actor, _ = self.actor(state, fixed_zs)

			Q_target_next = Q_target_next + entropy_bonus - std_q_target
			Q_target = reward + not_done * self.args.discount * Q_target_next

			if "TD7" in self.args.policy:
				self.max, self.min = max(self.max, float(Q_target.max())), min(self.min, float(Q_target.min()))

		# TD loss.
		Q = self.critic(state, action, fixed_zsa, fixed_zs)
		td_loss = (Q - Q_target).abs()

		# Critic step.
		critic_loss = LAP_huber(td_loss)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update LAP.
		if "TD7" in self.args.policy:
			priority = td_loss.max(1)[0]
			priority = priority.clamp(min=self.hp.min_priority).pow(self.hp.alpha)
			self.replay_buffer.update_priority(priority)

		# Update Actor.
		if self.training_steps % self.hp.policy_freq == 0:
			actor, log_prob_a_tilda = self.actor(state, fixed_zs)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
			Q = self.critic(state, actor, fixed_zsa, fixed_zs)

			if "SAC" in self.args.policy:
				actor_loss = (self.args.alpha_sac * log_prob_a_tilda - Q.mean()).mean()
			else:
				actor_loss = -Q.mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		# Update Iteration
		if self.training_steps % self.hp.target_update_rate == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())

			if "TD7" in self.args.policy:
				self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
				self.fixed_encoder.load_state_dict(self.encoder.state_dict())

				self.replay_buffer.reset_max_priority()

				self.max_target = self.max
				self.min_target = self.min

	# If using checkpoints: run when each episode terminates
	def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
		self.eps_since_update += 1
		self.timesteps_since_update += ep_timesteps
		self.min_return = min(self.min_return, ep_return)

		# End evaluation of current policy early
		if self.min_return < self.best_min_return:
			self.train_and_reset()

		# Update checkpoint
		elif self.eps_since_update == self.max_eps_before_update:
			self.best_min_return = self.min_return

			self.checkpoint_actor.load_state_dict(self.actor.state_dict())
			self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

			self.train_and_reset()

	# Batch training
	def train_and_reset(self):
		# UTD
		for _ in range(self.timesteps_since_update):
			if self.training_steps == self.hp.steps_before_checkpointing:
				self.best_min_return *= self.hp.reset_weight
				self.max_eps_before_update = self.hp.max_eps_when_checkpointing

			self.train()

		self.eps_since_update = 0
		self.timesteps_since_update = 0
		self.min_return = 1e8