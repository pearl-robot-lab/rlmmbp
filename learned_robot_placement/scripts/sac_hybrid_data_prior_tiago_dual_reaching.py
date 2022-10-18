import os
import gc
import numpy as np
import argparse
from datetime import datetime
import json
from viz_dataset import run_log_wandb, vid_log_wandb
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC_hybrid # For mixed action space
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange

ISAAC_PATH = os.environ.get('ISAAC_PATH')

## Experiment config
n_epochs = 350
n_steps = 400 # per epoch
n_steps_test = 80 # per eval
# MDP
world_xy_radius = 5.0 # 5m reaching task!
action_xy_radius = 1.0
action_xy_ang_lim = np.pi
horizon = 10
gamma = 0.95
# reward_dist_weight = 0. # zero out all rewards except IK ones to train an IK-only agent
# reward_timeout = 0.
# reward_collision = 0.
# Agent
use_data_prior = False
prior_eps = 0.5
prior_eps_decay = prior_eps/(0.75*n_epochs) # start with prior_eps and decay by prior_eps_decay every epoch. Set to high value to only fill initial replay buffer
use_kl_on_q = False # Use a KL between prior task policy and new task policy as a reward
kl_on_q_alpha = 1e-3 # Alpha parameter to weight the KL divergence reward
kl_on_q_alpha_decay = kl_on_q_alpha/(2*n_epochs/3) # decay kl_on_q_alpha over epochs
use_kl_on_pi = False # Use a KL between prior task policy and new task policy as a policy loss
kl_on_pi_alpha = 1e-3 # Alpha parameter to weight the KL divergence policy loss
kl_on_pi_alpha_decay = kl_on_pi_alpha/(2*n_epochs/3) # decay kl_on_q_alpha over epochs
initial_replay_size = 512
warmup_transitions = 512 # set to same as initial_replay_size
max_replay_size = 75000
batch_size = 128
n_features = 128
tau = 0.005
lr_actor_net = 0.0003
lr_critic_net = 0.0003
lr_alpha = 0.0003
temperature = 1.0 # For the softmax of gumbel

exp_config = dict() # Variable for saving the experiment config
for variable in ["n_epochs", "n_steps", "n_steps_test", "world_xy_radius", "action_xy_radius", "action_xy_ang_lim", 
                "horizon", "gamma", "use_data_prior", "prior_eps", "prior_eps_decay",
                "use_kl_on_q", "kl_on_q_alpha", "kl_on_q_alpha_decay", "use_kl_on_pi", "kl_on_pi_alpha", "kl_on_pi_alpha_decay",
                "initial_replay_size", "max_replay_size", "batch_size", "n_features", "warmup_transitions",
                "tau", "lr_actor_net", "lr_critic_net", "lr_alpha", "temperature"]:
    exp_config[variable] = eval(variable)


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


def experiment(test_only:bool=False, test_agent_path:str=None, render:bool=False, log_agent:bool=True,
               exp_name:str=None, num_seq_seeds:int=5, **kwargs):
    
    # Logging paths/directories
    isaac_path = str(ISAAC_PATH)
    results_dir=isaac_path+"/tiago_isaac/experiments/logs/SAC_hybrid"
    if exp_name is None:
        exp_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # timestamp

    use_cuda = torch.cuda.is_available()

    # MDP
    from envs.tiago_dual_reaching_env import TiagoIsaacReaching
    mdp = TiagoIsaacReaching(world_xy_radius=world_xy_radius, action_xy_radius=action_xy_radius, action_xy_ang_lim=action_xy_ang_lim,
                            gamma=gamma, horizon=horizon)#, reward_dist_weight=reward_dist_weight, reward_timeout = reward_timeout,
                            #reward_collision=reward_collision)
    # Need to set these for hybrid action space!
    action_space_continous = (3,)
    action_space_discrete = (2,)

    if(test_only):
        np.random.seed()

        logger = Logger(results_dir=results_dir+'/test', log_console=True)
        logger.strong_line()
        logger.info('Test: Experiment Algorithm: SAC_hybrid, Environment: TiagoIsaacReaching')
        logger.info('Test: Agent stored at '+test_agent_path)
        
        agent = SAC_hybrid.load(test_agent_path)
    
        # Algorithm
        core = Core(agent, mdp)
        
        if(True):
            dataset = core.evaluate(n_episodes=25, render=render, get_renders=True) # dummy to load meshes
            img_dataset = core.evaluate(n_episodes=50, render=render, get_renders=True)
            vid_log_wandb(exp_config=exp_config, run_name=logger._log_id, group_name=exp_name, img_dataset=img_dataset)
            dataset = img_dataset[1:]
        else:
            dataset = core.evaluate(n_episodes=50, render=render)

            J = np.mean(compute_J(dataset, gamma))
            R = np.mean(compute_J(dataset))
            s, *_ = parse_dataset(dataset)
            E = agent.policy.entropy(s)

            logger.info("Test: J="+str(J)+", R="+str(R)+", E="+str(E))

        mdp.shutdown()
    else:
        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        # Discrete approximator takes state and continuous action as input
        actor_discrete_input_shape = (mdp.info.observation_space.shape[0]+action_space_continous[0],)
        actor_mu_params = dict(network=ActorNetwork,
                            n_features=n_features,
                            input_shape=actor_input_shape,
                            output_shape=action_space_continous,
                            use_cuda=use_cuda)
        actor_sigma_params = dict(network=ActorNetwork,
                                n_features=n_features,
                                input_shape=actor_input_shape,
                                output_shape=action_space_continous,
                                use_cuda=use_cuda)
        actor_discrete_params = dict(network=ActorNetwork,
                                n_features=n_features,
                                input_shape=actor_discrete_input_shape,
                                output_shape=action_space_discrete,
                                use_cuda=use_cuda)
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': lr_actor_net}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)# full action space
        critic_params = dict(network=CriticNetwork,
                            optimizer={'class': optim.Adam,
                                        'params': {'lr': lr_critic_net}},
                            loss=F.mse_loss,
                            n_features=n_features,
                            input_shape=critic_input_shape,
                            output_shape=(1,),
                            use_cuda=use_cuda)
        
        # Log experiment config file (dictionary)
        exp_config["num_seq_seeds"] = num_seq_seeds
        try:
            os.makedirs(results_dir+'/'+exp_name)
        except FileExistsError as err:
            print("Appending existing File/Directory...")
        with open(results_dir+'/'+exp_name+'/exp_config.json', 'w') as f:
            json.dump(exp_config, f, indent=4)
        
        # Loop over num_seq_seeds:
        for exp in range(num_seq_seeds):
            np.random.seed()

            # Logging
            logger = Logger(results_dir=results_dir+'/'+exp_name, log_console=True)
            logger.strong_line()
            logger.info('Experiment Algorithm: SAC_hybrid, Environment: TiagoIsaacReaching, Trial: ' + str(exp))
            exp_eval_dataset = list() # This will be a list of dicts with datasets from every epoch
            
            # Optional: Load agents from prior tasks (to calculate KL with a prior policy)
            prior_agents = list()
            # prior_agents.append(SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-02-17-02-23-19/2022-02-17-05-33-31/agent-123.msh'))
            
            # Load agent Q function from prior task (to continue the learning)
            prior_agent_to_reuse = None
            # prior_agent_to_reuse = SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-05-04-16-23-50-padded/2022-05-04-16-24-10/agent-36.msh')

            # Agent
            agent = SAC_hybrid(mdp.info, actor_mu_params, actor_sigma_params, actor_discrete_params,
                        actor_optimizer, critic_params, batch_size, initial_replay_size,
                        max_replay_size, warmup_transitions, tau, lr_alpha, temperature=temperature,
                        prior_agents=prior_agents, mdp_get_prior_state_fn=mdp.get_prior_task_states,
                        use_kl_on_q=use_kl_on_q, kl_on_q_alpha=kl_on_q_alpha,
                        use_kl_on_pi=use_kl_on_pi, kl_on_pi_alpha=kl_on_pi_alpha, prior_agent_to_reuse=prior_agent_to_reuse)
                        # Unfortunately we need to pass the mdp get_prior_task_states function above (because the agent needs to do mdp state specific filtering to use the prior agents)

            # Algorithm
            core = Core(agent, mdp, use_data_prior=use_data_prior, prior_eps=prior_eps) # if using prior to bias data collection (with epsilon probability)

            # RUN
            eval_dataset = core.evaluate(n_steps=n_steps_test, render=render)
            s, *_ = parse_dataset(eval_dataset)
            J = np.mean(compute_J(eval_dataset, mdp.info.gamma))
            R = np.mean(compute_J(eval_dataset))
            E = agent.policy.entropy(s)

            core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, render=render)
            Prior_sample_cnt = core._prior_sample_count # number of prior samples used during learning
            Prior_success_rate = (100.0*core._prior_success_count)/max(core._prior_sample_count,1) # % of prior samples that led to success
            logger.epoch_info(0, J=J, R=R, entropy=E, Prior_sample_cnt=Prior_sample_cnt, Prior_success_rate=Prior_success_rate)
            exp_eval_dataset.append({"Epoch": 0, "J": J, "R": R, "entropy": E, "Prior_sample_cnt": Prior_sample_cnt, "Prior_success_rate": Prior_success_rate})

            for n in trange(n_epochs, leave=False):
                core._prior_eps -= max(prior_eps_decay,0.0) # decay the rate at which the prior is sampled (every epoch)
                core.learn(n_steps=n_steps, n_steps_per_fit=1, render=render)
                agent._kl_on_q_alpha = max(agent._kl_on_q_alpha-kl_on_q_alpha_decay,0.0) # decay the kl-divergence reward weight alpha
                agent._kl_on_pi_alpha = max(agent._kl_on_pi_alpha-kl_on_pi_alpha_decay,0.0) # decay the kl-divergence loss weight alpha
                Prior_sample_cnt = core._prior_sample_count # number of prior samples used during learning
                Prior_success_rate = (100.0*core._prior_success_count)/max(core._prior_sample_count,1) # % of prior samples that led to success
                
                eval_dataset = core.evaluate(n_steps=n_steps_test, render=render)
                s, _, rewards, _, _, last = parse_dataset(eval_dataset)
                J = np.mean(compute_J(eval_dataset, mdp.info.gamma))
                R = np.mean(compute_J(eval_dataset))
                E = agent.policy.entropy(s)
                Success_rate = np.sum(rewards >= 0.5)/np.sum(last) # num_successes/num_episodes
                Avg_episode_length = n_steps_test/np.sum(last)
                Q_loss = core.agent._critic_approximator[0].loss_fit
                Actor_loss = core.agent._actor_last_loss
                KL_with_prior = agent._kl_with_prior[agent._kl_with_prior.nonzero()[0]].mean()

                logger.epoch_info(n+1, Success_rate=Success_rate, J=J, R=R, entropy=E, Q_loss=Q_loss, Actor_loss=Actor_loss, KL_with_prior=KL_with_prior, 
                                Avg_episode_length=Avg_episode_length, Prior_sample_cnt=Prior_sample_cnt, 
                                Prior_success_rate=Prior_success_rate)
                if(log_agent):
                    logger.log_agent(agent, epoch=n+1) # Log agent every epoch
                    # logger.log_best_agent(agent, J) # Log best agent
                exp_eval_dataset.append({"Epoch": n+1, "Success_rate": Success_rate, "J": J, "R": R, "entropy": E, 
                                        "Q_loss": Q_loss, "Actor_loss": Actor_loss, "KL_with_prior": KL_with_prior, "Avg_episode_length": Avg_episode_length, 
                                        "Prior_sample_cnt": Prior_sample_cnt, "Prior_success_rate": Prior_success_rate})

            # Get video snippet of final learnt behavior
            img_dataset = core.evaluate(n_episodes=10, get_renders=True)
            del core, agent # save some CPU memory
            gc.collect()
            # log dataset and video
            logger.log_dataset(exp_eval_dataset)
            run_log_wandb(exp_config=exp_config, run_name=logger._log_id, group_name=exp_name, dataset=exp_eval_dataset)
            vid_log_wandb(exp_config=exp_config, run_name=logger._log_id, group_name=exp_name, img_dataset=img_dataset)
            

        # logger.info('Press a button to visualize evaluation')
        # input()
        # core.evaluate(n_episodes=5, render=render)
        mdp.shutdown()

if __name__ == '__main__':
    # For running on local PC:
    parser = argparse.ArgumentParser("SAC_hybrid Tiago Reach")
    parser.add_argument("--test", default=False, action="store_true", help="Test an agent only")
    parser.add_argument("--test_agent", type=str, default="", help="Filename (with path) of the agent\'s .msh file")
    parser.add_argument("--render", default=False, action="store_true", help="Render environment")
    parser.add_argument("--log_agent", default=True, action="store_true", help="Log agent every epoch")
    parser.add_argument("--num_seq_seeds", type=int, default=5, help="Number of sequential seeds to run")
    args, unknown = parser.parse_known_args()

    if(args.test):
        if not args.test_agent:
            print("Error. Missing test agent.msh file string.")
            exit()

    experiment(test_only=args.test, test_agent_path=args.test_agent, num_seq_seeds=args.num_seq_seeds, 
               render=args.render, log_agent=args.log_agent, exp_name=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    # # For running on cluster with experiment_launcher:
    # from experiment_launcher import run_experiment
    # run_experiment(experiment)