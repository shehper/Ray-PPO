# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs[0].env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class Logging_Data:
    def __init__(self):
        self.global_step = 0

    def increment_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return self.global_step

class Rollout:
    def __init__(self, env_callable):
        # random.seed(env_seed)
        # np.random.seed(env_seed)
        # torch.manual_seed(env_seed)
        # torch.backends.cudnn.deterministic = args.torch_deterministic # not sure what this does
        self.env = env_callable()
        self.obs = torch.zeros((args.num_steps,) + self.env.observation_space.shape)
        self.actions = torch.zeros((args.num_steps,) + self.env.action_space.shape)
        self.logprobs = torch.zeros((args.num_steps,))
        self.rewards = torch.zeros((args.num_steps,))
        self.dones = torch.zeros((args.num_steps,))
        self.values = torch.zeros((args.num_steps,))
        self.advantages = torch.zeros((args.num_steps,))
        self.returns = torch.zeros((args.num_steps,))
        
        self.next_obs = torch.Tensor(self.env.reset()).to(device)
        self.next_done = torch.zeros(1).to(device)

        self.episode_return = 0
        self.episode_length = 0
        

    def rollout(self, agent, logging_data):
        for step in range(args.num_steps):
            logging_data.increment_global_step()
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten() # num_envs
            self.actions[step] = action
            self.logprobs[step] = logprob

            self.next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(device).view(-1) # different
            self.next_obs = torch.Tensor(self.next_obs).to(device)
            self.next_done = torch.Tensor([done]).to(device) # different
            
            self.episode_return += reward
            self.episode_length += 1

            if done:
                global_step = logging_data.get_global_step()
                print(f"global_step={global_step}, episodic_return={self.episode_return}")
                writer.add_scalar("charts/episodic_return", self.episode_return, global_step)
                writer.add_scalar("charts/episodic_length", self.episode_length, global_step)
                self.episode_length = self.episode_return = 0
                self.env.reset()


        with torch.no_grad():
            next_value = agent.get_value(self.next_obs).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            self.returns = self.advantages + self.values 

        rollout_data = {'obs': self.obs,
             'actions': self.actions, 
             'logprobs': self.logprobs,
             'values': self.values,
             'returns': self.returns,
             'advantages': self.advantages}

        return rollout_data

def update_parameters(agent, optimizer, rollout_data, batch_size, minibatch_size, update_epochs, 
                  vf_coef, ent_coef, clip_vloss, clip_coef, target_kl, max_grad_norm, norm_adv):
    
    # How important is max_grad_norm?
    # Don't have to give batch size. It's just the length of b_actions, for example.

    # Can I compute minibatch size from other things?
    b_inds = np.arange(batch_size) # indices of batch_size

    b_obs = torch.cat([result['obs'] for result in rollout_data], axis=0)
    b_logprobs = torch.cat([result['logprobs'] for result in rollout_data], axis=0)
    b_actions = torch.cat([result['actions'] for result in rollout_data], axis=0)
    b_advantages = torch.cat([result['advantages'] for result in rollout_data], axis=0)
    b_returns = torch.cat([result['returns'] for result in rollout_data], axis=0)
    b_values = torch.cat([result['values'] for result in rollout_data], axis=0)

    clipfracs = []

    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size): #start of minbatch: 0, m, 2*m, ..., (n-1)*m; here m=1024, n = 4
            end = start + minibatch_size # end of minibatch: m, 2*m, ..., n*m; here m=1024, n = 4
            mb_inds = b_inds[start:end] # indices of minibatch

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) #.long() converts dtype to int64
            logratio = newlogprob - b_logprobs[mb_inds] 
            ratio = logratio.exp() # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.

            with torch.no_grad():
                old_approx_kl = (-logratio).mean() # mean of -log(pi(a|s) / pi_old(a|s))
                approx_kl = ((ratio - 1) - logratio).mean() # mean of (pi(a|s) / pi_old(a|s) - 1 - log(pi(a|s) / pi_old(a|s)))
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]  

            mb_advantages = b_advantages[mb_inds]
            if norm_adv: 
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1) # value computed by NN with updated parameters
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm) # clip gradients before updating them.
            optimizer.step()

        # TODO: Why is this if statement outside of the loop?
        if target_kl is not None:
            if approx_kl > target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy() 
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return pg_loss, v_loss, entropy_loss, approx_kl, explained_var, clipfracs

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = [Rollout(make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)) for i in range(args.num_envs)]
    assert isinstance(envs[0].env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    logging_data = Logging_Data()

    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates+1):

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_data = [env.rollout(agent, logging_data) for env in envs]
        
        pg_loss, v_loss, entropy_loss, approx_kl, explained_var, clipfracs = update_parameters(
        agent, optimizer, rollout_data, args.batch_size, args.minibatch_size, args.update_epochs, 
        args.vf_coef, args.ent_coef, args.clip_vloss, args.clip_coef, args.target_kl, args.max_grad_norm, 
        args.norm_adv)

        global_step = logging_data.get_global_step()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        #writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()