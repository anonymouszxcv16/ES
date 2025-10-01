import argparse
import os
import time
import numpy as np
import torch
import gym

import TD

def train_offline(RL_agent, env, eval_env, args):
    # Performance.
    evals = []
    times = []
    alphas = []

    # Load offline dataset.
    RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))
    start_time = time.time()

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, t, start_time, args)

        # Train.
        RL_agent.train()

# Train online RL agent.
def train_online(RL_agent, env, eval_env, args):
    # Reward
    evals = []

    # Time
    times = []

    # Alpha
    alphas = []

    # Activations
    activations = []

    # Initialize
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset()[0], False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    # Train loop.
    for t in range(int(args.max_timesteps + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, activations, t, start_time, args)

        # Select action.
        if allow_train:
            action = RL_agent.select_action(np.array(state), deterministic=False if "SAC" in args.policy else True)
        else:
            action = env.env.action_space.sample()

        # Do a step.
        next_state, reward, ep_finished, truncated, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1
        done = float(ep_finished or truncated)

        # Store tuple.
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train:
            # Train.
            RL_agent.train()

        if ep_finished:
            if t >= args.timesteps_before_training:
                allow_train = True

            state, done = env.reset()[0], False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


# Logs.
def maybe_evaluate_and_print(RL_agent, eval_env, evals, times, alphas, activations, t, start_time, args):
    if t % args.eval_freq == 0:
        # Rewards
        total_reward = np.zeros(args.eval_eps)
        discounted_reward = np.zeros(args.eval_eps)

        for ep in range(args.eval_eps):
            state, done = eval_env.reset()[0], False
            step = 0

            # Episode
            while not done:
                # Action selection.
                action = RL_agent.select_action(state, args.use_checkpoints, use_exploration=False)

                # Step.
                state, reward, done, truncated, _ = eval_env.step(action)
                done = done or truncated

                # Reward sum.
                discounted_reward[ep] += reward * RL_agent.args.discount ** step
                total_reward[ep] += reward

                step += 1

        # Time
        time_total = (time.time() - start_time) / 60

        # Reward
        score = eval_env.get_normalized_score(total_reward.mean()) * 100 if RL_agent.args.offline == 1 else total_reward.mean().item()

        activation_std = RL_agent.activations_std / args.eval_eps
        RL_agent.activations_std = 0

        print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}\tAlpha: {RL_agent.args.alpha:,.3f}\tActivation Std: {activation_std:,.1f}")
        # Reward
        evals.append(score)

        # Time
        times.append(time_total)

        # Alpha
        alphas.append(RL_agent.args.alpha)

        # Activations
        activations.append(activation_std)

        # file.
        with open(f"./results/{args.env}/{args.file_name}", "w") as file:
            file.write(f"{evals}\n{times}\n{alphas}\n{activations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Algorithm.
    parser.add_argument("--policy", default="TD3", type=str)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--auto_alpha", default=0, type=int)
    parser.add_argument("--auto_alpha_interval", default=100_000, type=int)
    parser.add_argument('--use_checkpoints', default=True)
    parser.add_argument('--offline', default=0, type=int)

    # Exploration.
    parser.add_argument("--timesteps_before_training", default=25_000, type=int)
    parser.add_argument("--exploration_noise", default=.1, type=float)
    parser.add_argument("--discount", default=.99, type=float)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--buffer_size", default=1e6, type=int)

    # Environment.
    parser.add_argument("--env", default="Hopper-v2", type=str)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)

    # Evaluation
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)

    # File
    parser.add_argument('--file_name', default=None)
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = f"{args.policy}_{args.seed}"

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    # Offline.
    if args.offline == 1:
        import d4rl

        d4rl.set_dataset_path(args.d4rl_path)
        args.use_checkpoints = False

    # environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # Seed.
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = TD.Agent(state_dim, action_dim, max_action, args)
    name = f"{args.policy}_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Algorithm: {args.policy}, Alpha: {args.alpha:,.1f}, Environment: {args.env}, Seed: {args.seed}, Device: {RL_agent.device}")
    print("---------------------------------------")

    # Optimize.
    if args.offline == 1:
        train_offline(RL_agent, env, eval_env, args)
    else:
        train_online(RL_agent, env, eval_env, args)