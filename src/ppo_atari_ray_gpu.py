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

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

print(torch.__version__)
print(torch.cuda.is_available())

import ray
# ray.init(log_to_driver=False)
ray.init()

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
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
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
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--kl-penalty", type=float, default=0.0,
        help="the coefficient of KL penalty when using policy loss with kl penalty")
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
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
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
    #def __init__(self, obs_space_shape, action_space_n):
    def __init__(self, action_space_n):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_space_n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))
        #return self.critic(x)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        #logits = self.actor(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        #return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

@ray.remote
class Logging_Data:
    def __init__(self, run_name, args):
        self.global_step = 0
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
                mode="offline"
            )
        self.writer = SummaryWriter(f"atari-runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def increment_global_step(self):
        self.global_step += 1

    def log_data(self, data):
        for key, value in data.items():
            self.writer.add_scalar(key, value, self.global_step)

    def get_global_step(self):
        return self.global_step

@ray.remote
class Rollout:
    def __init__(self, env_callable):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = args.torch_deterministic
        eval('setattr(torch.backends.cudnn, "benchmark", True)') 
        # https://github.com/ray-project/ray/issues/8569

        self.env = env_callable()
        self.obs = torch.zeros((args.num_steps,) + self.env.observation_space.shape)
        self.actions = torch.zeros((args.num_steps,) + self.env.action_space.shape)
        self.logprobs = torch.zeros((args.num_steps,))
        self.rewards = torch.zeros((args.num_steps,))
        self.dones = torch.zeros((args.num_steps,))
        self.values = torch.zeros((args.num_steps,))
        self.advantages = torch.zeros((args.num_steps,))
        self.returns = torch.zeros((args.num_steps,))
        
        self.next_obs = torch.Tensor(self.env.reset())
        self.next_done = torch.zeros(1)

    def get_env_spaces_data(self):
        return self.env.observation_space.shape, self.env.action_space.n
        
    def rollout(self, agent, logging_data):
        for step in range(args.num_steps):
            ray.get(logging_data.increment_global_step.remote())
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(torch.unsqueeze(self.next_obs,dim=0))
                self.values[step] = value.flatten() # num_envs
            self.actions[step] = action
            self.logprobs[step] = logprob

            self.next_obs, reward, done, info = self.env.step(torch.squeeze(action).cpu().numpy())
            self.rewards[step] = torch.tensor(reward).view(-1) # different
            self.next_obs = torch.Tensor(self.next_obs)
            self.next_done = torch.Tensor([done]) # different       

            if 'episode' in info.keys(): # RecordEpisodeStatistics includes 'episode' in info when done
                global_step = ray.get(logging_data.get_global_step.remote())
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                ray.get(logging_data.log_data.remote(
                    {"charts/episodic_return": info['episode']['r'],
                     "charts/episodic_length": info['episode']['l']}
                ))
                self.env.reset()

        with torch.no_grad():
            next_value = agent.get_value(torch.unsqueeze(self.next_obs,dim=0)).reshape(1, -1)
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

def update_parameters(agent, optimizer, rollout_data, args):

    b_inds = np.arange(args.batch_size) # indices of batch_size
    b_obs = torch.cat([result['obs'] for result in rollout_data], axis=0).to(device)
    b_logprobs = torch.cat([result['logprobs'] for result in rollout_data], axis=0).to(device)
    b_actions = torch.cat([result['actions'] for result in rollout_data], axis=0).to(device)
    b_advantages = torch.cat([result['advantages'] for result in rollout_data], axis=0).to(device)
    b_returns = torch.cat([result['returns'] for result in rollout_data], axis=0).to(device)
    b_values = torch.cat([result['values'] for result in rollout_data], axis=0).to(device)

    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, args.batch_size, args.minibatch_size): 
            end = start + args.minibatch_size 
            mb_inds = b_inds[start:end] # indices of minibatch

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds]) #.long() converts dtype to int64
            logratio = newlogprob - b_logprobs[mb_inds] 
            ratio = logratio.exp() # pi(a|s) / pi_old(a|s); is a tensor of 1s for epoch=0.

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean() # mean of (pi(a|s) / pi_old(a|s) - 1 - log(pi(a|s) / pi_old(a|s)))
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv: 
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            # Usage: pass --clip-coef=0.0 --kl-penalty=.. when using KL penalty loss instead of clip loss
            if args.clip_coef > 0.0: # clip loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            else: # kl penalty loss
                pg_loss = (-mb_advantages * ratio + args.kl_penalty * 0.5 * logratio ** 2).mean()
                # The estimator 0.5 * (logratio ** 2).mean() of KL penalty is biased but has low variance. It is described in 
                # http://joschu.net/blog/kl-approx.html and is used in PPG implementation:
                # https://github.com/openai/phasic-policy-gradient/blob/7295473f0185c82f9eb9c1e17a373135edd8aacc/phasic_policy_gradient/ppo.py#L104
                
                # WARNING: In the original paper on PPO, kl_penalty coefficient is either fixed (which is what we use here) or 
                # adjusted after each update (see eq 8 in the paper). Here we use fixed KL penalty as this is used in the PPG paper (see section 3.5)
                # and our goal is to implement PPO-EWMA with KL penalty loss.

            # Value loss
            newvalue = newvalue.view(-1) # value computed by NN with updated parameters
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # clip gradients before updating them.
            optimizer.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    y_pred, y_true = b_values, b_returns
    var_y = torch.var(y_true)
    explained_var = np.nan if var_y.item() == 0 else 1 - torch.var(y_true - y_pred) / var_y

    output = {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy_loss": entropy_loss.item(), 
              "approx_kl": approx_kl, "explained_var": explained_var, "clipfrac": np.mean(clipfracs)}
    return output

if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    logging_data = Logging_Data.remote(run_name, args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = [Rollout.remote(make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)) for i in range(args.num_envs)]
    _, action_space_n = ray.get(envs[0].get_env_spaces_data.remote())
    agent = Agent(action_space_n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates+1):
        # Annealing the learning rate if required
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collecting data through N parallel actors
        agent.to("cpu")
        rollout_data = ray.get([env.rollout.remote(agent, logging_data) for env in envs])

        # Updating parameters of neural networks
        agent.to(device)
        output = update_parameters(agent, optimizer, rollout_data, args)
        global_step = ray.get(logging_data.get_global_step.remote())
        SPS = int(global_step / (time.time() - start_time))

        # for key, value in output.items():
        #     print('for key ', key, ' type of value is: ',  type(value))
        # print(output['approx_kl'].device)
        # print(output['explained_var'].device)
        # print(type(optimizer.param_groups[0]["lr"]))

        ray.get(logging_data.log_data.remote(
                    {"charts/learning_rate": optimizer.param_groups[0]["lr"],
                     "losses/value_loss": output['v_loss'],
                     "losses/policy_loss": output['pg_loss'],
                     "losses/entropy": output['entropy_loss'],
                     "losses/approx_kl": output['approx_kl'].cpu(),
                     "losses/clipfrac": output['clipfrac'],
                     "losses/explained_variance": output['explained_var'].cpu(),
                     "charts/SPS": SPS
                     }
                ))
        print("update: ", update, " SPS: ", SPS)