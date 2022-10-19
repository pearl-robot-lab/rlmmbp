# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import os
from datetime import datetime
import argparse

import numpy as np
import hydra
from omegaconf import DictConfig

import learned_robot_placement
from learned_robot_placement.utils.hydra_cfg.hydra_utils import *
from learned_robot_placement.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from learned_robot_placement.utils.task_util import initialize_task
from learned_robot_placement.envs.isaac_env_mushroom import IsaacEnvMushroom

# Use Mushroom RL library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import *
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange

# (Optional) Logging with Weights & biases
# from learned_robot_placement.utils.wandb_utils import wandbLogger
# import wandb


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(cfg: DictConfig = None, cfg_file_path: str = "", seed: int = 0, results_dir: str = ""):
    
    # Get configs
    if(cfg_file_path):
        # Get config file from path
        cfg = OmegaConf.load(cfg_file_path)
    if(cfg.checkpoint):
        print("Loading task and train config from checkpoint config file....")
        try:
            cfg_new = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint), '..', 'config.yaml'))
            cfg.task = cfg_new.task
            cfg.train = cfg_new.train
        except Exception as e:
            print("Loading checkpoint config failed!")
            print(e)
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    headless = cfg.headless
    render = cfg.render
    sim_app_cfg_path = cfg.sim_app_cfg_path
    rl_params_cfg = cfg.train.params.config
    algo_map = {"SAC_hybrid": SAC_hybrid,    # Mappings from strings to algorithms
                "SAC":SAC,
                "BHyRL":BHyRL,}
    algo = algo_map[cfg.train.params.algo.name]

    # Set up environment
    env = IsaacEnvMushroom(headless=headless,render=render,sim_app_cfg_path=sim_app_cfg_path)
    task = initialize_task(cfg_dict, env)

    # Set up logging paths/directories
    exp_name = cfg.train.params.config.name
    exp_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # append datetime for logging
    results_dir = os.path.join(learned_robot_placement.__path__[0],'logs',cfg.task.name,exp_name)
    if(cfg.test): results_dir = os.path.join(results_dir,'test')
    results_dir = os.path.join(results_dir,exp_stamp)
    os.makedirs(results_dir, exist_ok=True)
    # log experiment config
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Test/Train
    if(cfg.test):
        if(cfg.checkpoint):
            np.random.seed()
            # Logger
            logger = Logger(results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Test: {exp_name}')
            logger.info(f'Test: Agent stored at '+ cfg.checkpoint)
        
            # Algorithm
            agent = algo.load(cfg.checkpoint)
            # Runner
            core = Core(agent, env)

            env._run_sim_rendering = ((not cfg.headless) or cfg.render)
            dataset = core.evaluate(n_episodes=50, render=cfg.render)
            J = np.mean(compute_J(dataset, env.info.gamma))
            R = np.mean(compute_J(dataset))
            s, *_ = parse_dataset(dataset)
            E = agent.policy.entropy(s)
            logger.info("Test: J="+str(J)+", R="+str(R)+", E="+str(E))
        else:
            raise TypeError("Test option chosen but no valid checkpoint provided")
        env._simulation_app.close()
    else:
        # Approximators
        use_cuda = ('cuda' in cfg.rl_device)
        actor_input_shape = env.info.observation_space.shape
        # Need to set these for hybrid action space!
        action_space_continous = (cfg.task.env.continous_actions,)
        action_space_discrete = (cfg.task.env.discrete_actions,)
        # Discrete approximator takes state and continuous action as input
        actor_discrete_input_shape = (env.info.observation_space.shape[0]+action_space_continous[0],)
        actor_mu_params = dict(network=ActorNetwork,
                            n_features=rl_params_cfg.n_features,
                            input_shape=actor_input_shape,
                            output_shape=action_space_continous,
                            use_cuda=use_cuda)
        actor_sigma_params = dict(network=ActorNetwork,
                                n_features=rl_params_cfg.n_features,
                                input_shape=actor_input_shape,
                                output_shape=action_space_continous,
                                use_cuda=use_cuda)
        actor_discrete_params = dict(network=ActorNetwork,
                                n_features=rl_params_cfg.n_features,
                                input_shape=actor_discrete_input_shape,
                                output_shape=action_space_discrete,
                                use_cuda=use_cuda)
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': rl_params_cfg.lr_actor_net}}

        critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],) # full action space
        critic_params = dict(network=CriticNetwork,
                            optimizer={'class': optim.Adam,
                                        'params': {'lr': rl_params_cfg.lr_critic_net}},
                            loss=F.mse_loss,
                            n_features=rl_params_cfg.n_features,
                            input_shape=critic_input_shape,
                            output_shape=(1,),
                            use_cuda=use_cuda)

        # Loop over num_seq_seeds:
        for exp in range(cfg.num_seeds):
            np.random.seed()

            # Logger
            logger = Logger(results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Experiment: {exp_name}, Trial: {exp}')
            exp_eval_dataset = list() # This will be a list of dicts with datasets from every epoch
            # wandb_logger = wandbLogger(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp) # Optional
            
            # Agent
            agent = algo(env.info, actor_mu_params, actor_sigma_params, actor_discrete_params, actor_optimizer, critic_params,
                        batch_size=rl_params_cfg.batch_size, initial_replay_size=rl_params_cfg.initial_replay_size,
                        max_replay_size=rl_params_cfg.max_replay_size, warmup_transitions=rl_params_cfg.warmup_transitions,
                        tau=rl_params_cfg.tau, lr_alpha=rl_params_cfg.lr_alpha, temperature=rl_params_cfg.temperature)

            # Setup boosting (for BHyRL):
            if rl_params_cfg.prior_agent is not None:
                prior_agents = list()
                prior_agents.append(algo.load(rl_params_cfg.prior_agent))
                agent.setup_boosting(prior_agents=prior_agents, use_kl_on_pi=rl_params_cfg.use_kl_on_pi, kl_on_pi_alpha=rl_params_cfg.kl_on_pi_alpha)

            # Algorithm
            core = Core(agent, env)

            # RUN
            eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render)
            s, _, _, _, _, info, last = parse_dataset(eval_dataset)
            J = np.mean(compute_J(eval_dataset, env.info.gamma))
            R = np.mean(compute_J(eval_dataset))
            E = agent.policy.entropy(s)
            success_rate = np.sum(info)/np.sum(last) # info contains successes. rate=num_successes/num_episodes
            avg_episode_length = rl_params_cfg.n_steps_test/np.sum(last)
            logger.epoch_info(0, success_rate=success_rate, J=J, R=R, entropy=E, avg_episode_length=avg_episode_length)
            # Optional wandb logging
            # wandb.watch(agent.policy._mu_approximator.model.network)
            # wandb.watch(agent.policy._sigma_approximator.model.network)
            # wandb.watch(agent._critic_approximator.model._model[0].network)
            exp_eval_dataset.append({"Epoch": 0, "success_rate": success_rate, "J": J, "R": R, "entropy": E, "avg_episode_length": avg_episode_length})
            # initialize replay buffer
            core.learn(n_steps=rl_params_cfg.initial_replay_size, n_steps_per_fit=rl_params_cfg.initial_replay_size, render=cfg.render)

            for n in trange(rl_params_cfg.n_epochs, leave=False):
                core.learn(n_steps=rl_params_cfg.n_steps, n_steps_per_fit=1, render=cfg.render)
                
                eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render)
                s, _, _, _, _, info, last = parse_dataset(eval_dataset)
                J = np.mean(compute_J(eval_dataset, env.info.gamma))
                R = np.mean(compute_J(eval_dataset))
                E = agent.policy.entropy(s)
                success_rate = np.sum(info)/np.sum(last) # info contains successes. rate=num_successes/num_episodes
                avg_episode_length = rl_params_cfg.n_steps_test/np.sum(last)
                q_loss = core.agent._critic_approximator[0].loss_fit
                actor_loss = core.agent._actor_last_loss

                logger.epoch_info(n+1, success_rate=success_rate, J=J, R=R, entropy=E, avg_episode_length=avg_episode_length,
                                  q_loss=q_loss, actor_loss=actor_loss)
                if(rl_params_cfg.log_checkpoints):
                    logger.log_agent(agent, epoch=n+1) # Log agent every epoch
                    # logger.log_best_agent(agent, J) # Log best agent
                current_log={"success_rate": success_rate, "J": J, "R": R, "entropy": E, 
                             "avg_episode_length": avg_episode_length, "q_loss": q_loss, "actor_loss": actor_loss}
                exp_eval_dataset.append(current_log)
                # wandb_logger.run_log_wandb(success_rate, J, R, E, avg_episode_length, q_loss)

            # Get video snippet of final learnt behavior (enable internal rendering for this)
            # prev_env_render_setting = bool(env._run_sim_rendering)
            # env._run_sim_rendering = True
            # img_dataset = core.evaluate(n_episodes=5, get_renders=True)
            # env._run_sim_rendering = prev_env_render_setting
            # log dataset and video
            # logger.log_dataset(exp_eval_dataset)
            # run_log_wandb(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp, dataset=exp_eval_dataset)
            # img_dataset = img_dataset[::15] # Reduce size of img_dataset. Take every 15th image
            # wandb_logger.vid_log_wandb(img_dataset=img_dataset)

    # Shutdown
    env._simulation_app.close()


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_run_exp(cfg: DictConfig):
    experiment(cfg)


if __name__ == '__main__':
    parse_hydra_configs_and_run_exp()