import os
import gc
import numpy as np
import argparse
from datetime import datetime
import json
from viz_dataset import run_log_wandb, vid_log_wandb
import pdb
# import pinocchio as pin # Temporary to use older c++ lib with pinocchio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import BHyRL # mixed action space
# from mushroom_rl.algorithms.actor_critic import AWAC_hybrid # mixed action space
# from awac_hybrid_data_prior_tiago_dual_reaching import CriticNetwork, ActorNetwork # Need these networks to load the previous task's agent
from mushroom_rl.algorithms.actor_critic import SAC_hybrid # mixed action space
from sac_hybrid_data_prior_tiago_dual_reaching import CriticNetwork, ActorNetwork # Need these networks to load the previous task's agent
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange

ISAAC_PATH = os.environ.get('ISAAC_PATH')

## Experiment config
n_epochs = 300
n_steps = 800 # per epoch
n_steps_test = 110 # per eval
# MDP
world_xy_radius = 4.0 # Obstacle labroom multiobj reaching task!
action_xy_radius = 1.5
action_xy_ang_lim = np.pi
horizon = 10
gamma = 0.95
# Agent
use_kl_on_q = False # Use a KL between prior task policy and new task policy as a reward
kl_on_q_alpha = 1e-3 # Alpha parameter to weight the KL divergence reward
kl_on_q_alpha_decay = kl_on_q_alpha/(2*n_epochs/3) # decay kl_on_q_alpha over epochs
use_kl_on_pi = False # Use a KL between prior task policy and new task policy as a policy loss
kl_on_pi_alpha = 1e-3 # Alpha parameter to weight the KL divergence policy loss
kl_on_pi_alpha_decay = kl_on_pi_alpha/(2*n_epochs/3) # decay kl_on_q_alpha over epochs
use_entropy = False
log_std_min = -3 # Default -20. Clip the agent's policy log std to avoid very narrow gaussian policies
gauss_noise_cov = 0.01 # adds gaussian noise of zero mean and this covariance to the behavior policy
initial_replay_size = 2048
warmup_transitions = 2048 # set to same as initial_replay_size
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
                "horizon", "gamma", "use_kl_on_q", "kl_on_q_alpha", "kl_on_q_alpha_decay", "use_kl_on_pi",
                "kl_on_pi_alpha", "kl_on_pi_alpha_decay", "use_entropy", "log_std_min", "gauss_noise_cov",
                "initial_replay_size", "max_replay_size", "batch_size", "n_features", "warmup_transitions",
                "tau", "lr_actor_net", "lr_critic_net", "lr_alpha", "temperature"]:
    exp_config[variable] = eval(variable)


class BHyRL_CriticNetwork(nn.Module):
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


class BHyRL_ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(BHyRL_ActorNetwork, self).__init__()

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


def experiment(test_only:bool=False, test_agent_path:str=None, test_log_video:bool=False, render:bool=False, log_agent:bool=True,
               train_prev_agent:bool=False, exp_name:str=None, num_seq_seeds:int=5, **kwargs):
    
    # Logging paths/directories
    isaac_path = str(ISAAC_PATH)
    results_dir=isaac_path+"/tiago_isaac/experiments/logs/BHyRL_labroom_multiobj"
    if exp_name is None:
        exp_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # timestamp

    use_cuda = torch.cuda.is_available()

    if(test_only):
        np.random.seed()

        # MDP
        from envs.tiago_dual_labroom_multiobj_reaching_env import TiagoIsaacReaching
        mdp = TiagoIsaacReaching(world_xy_radius=world_xy_radius, action_xy_radius=action_xy_radius, action_xy_ang_lim=action_xy_ang_lim,
                                cubic_approx=False, gamma=gamma, horizon=horizon) # Don't use cubic approximation

        logger = Logger(results_dir=results_dir+'/test', log_console=True)
        logger.strong_line()
        logger.info('Test: Experiment Algorithm: BHyRL_labroom_multiobj, Environment: TiagoIsaacReaching')
        logger.info('Test: Agent stored at '+test_agent_path)
        
        agent = BHyRL.load(test_agent_path)
    
        # Algorithm
        core = Core(agent, mdp)
        
        if(test_log_video):
            img_dataset = core.evaluate(n_episodes=25, get_renders=True)
            vid_log_wandb(exp_config=exp_config, run_name=logger._log_id, group_name=exp_name, img_dataset=img_dataset)
            dataset = img_dataset[1:]
        else:
            dataset = core.evaluate(n_episodes=25, render=render)

            J = np.mean(compute_J(dataset, gamma))
            R = np.mean(compute_J(dataset))
            s, *_ = parse_dataset(dataset)
            E = agent.policy.entropy(s)

            logger.info("Test: J="+str(J)+", R="+str(R)+", E="+str(E))

        mdp.shutdown()
    else:
        # MDP
        from envs.tiago_dual_labroom_multiobj_reaching_env import TiagoIsaacReaching
        mdp = TiagoIsaacReaching(world_xy_radius=world_xy_radius, action_xy_radius=action_xy_radius, action_xy_ang_lim=action_xy_ang_lim,
                                gamma=gamma, horizon=horizon)
        # Need to set these for hybrid action space!
        action_space_continous = (3,)
        action_space_discrete = (2,)

        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        # Discrete approximator takes state and continuous action as input
        actor_discrete_input_shape = (mdp.info.observation_space.shape[0]+action_space_continous[0],)
        actor_mu_params = dict(network=BHyRL_ActorNetwork,
                            n_features=n_features,
                            input_shape=actor_input_shape,
                            output_shape=action_space_continous,
                            use_cuda=use_cuda)
        actor_sigma_params = dict(network=BHyRL_ActorNetwork,
                                n_features=n_features,
                                input_shape=actor_input_shape,
                                output_shape=action_space_continous,
                                use_cuda=use_cuda)
        actor_discrete_params = dict(network=BHyRL_ActorNetwork,
                                n_features=n_features,
                                input_shape=actor_discrete_input_shape,
                                output_shape=action_space_discrete,
                                use_cuda=use_cuda)
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': lr_actor_net}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)# full action space
        critic_params = dict(network=BHyRL_CriticNetwork,
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
            logger.info('Experiment Algorithm: BHyRL, Environment: TiagoIsaacLabRoomMultiobjReaching, Trial: ' + str(exp))
            exp_eval_dataset = list() # This will be a list of dicts with datasets from every epoch
            
            # Load agents from prior tasks (since we are learning the residuals). NOTE: Mind the order of the prior tasks
            prior_agents = list()
            # prior_agents.append(SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-02-17-02-23-19/2022-02-17-05-33-31/agent-123.msh')) # higher entropy
            # prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL/2022-02-20-17-59-58-5mEntropy/2022-02-20-18-02-05/agent-213.msh'))
            prior_agents.append(SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-05-08-21-56-11-nobackIK/2022-05-08-21-56-32/agent-211.msh'))

            # Agent
            agent = BHyRL(mdp.info, actor_mu_params, actor_sigma_params, actor_discrete_params,
                        actor_optimizer, critic_params, batch_size, initial_replay_size,
                        max_replay_size, warmup_transitions, tau, lr_alpha=lr_alpha, log_std_min=log_std_min, temperature=temperature,
                        use_entropy=use_entropy, gauss_noise_cov=gauss_noise_cov,
                        prior_agents=prior_agents, mdp_get_prior_state_fn=mdp.get_prior_task_states,
                        use_kl_on_q=use_kl_on_q, kl_on_q_alpha=kl_on_q_alpha,
                        use_kl_on_pi=use_kl_on_pi, kl_on_pi_alpha=kl_on_pi_alpha)
                        # Unfortunately we need to pass the mdp get_prior_task_states function above (because the agent needs to do mdp state specific filtering to use the prior agents)

            if (train_prev_agent and exp==0):
                start = 160
                prev_agent = BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL_labroom_multiobj/2022-07-14-17-02-34/2022-07-14-17-02-54/agent-160.msh')
                agent._critic_fit_params = prev_agent._critic_fit_params
                agent._batch_size=prev_agent._batch_size
                agent._warmup_transitions=prev_agent._warmup_transitions
                agent._tau=prev_agent._tau
                agent._target_entropy=prev_agent._target_entropy
                agent._replay_memory=prev_agent._replay_memory
                agent._critic_approximator=prev_agent._critic_approximator
                agent._target_critic_approximator=prev_agent._target_critic_approximator
                agent._log_alpha=prev_agent._log_alpha
                agent._alpha_optim=prev_agent._alpha_optim
                agent._optimizer=prev_agent._optimizer
                agent._clipping=prev_agent._clipping
                agent._clipping_params=prev_agent._clipping_params
                agent.mdp_info=prev_agent.mdp_info
                agent.policy=prev_agent.policy
                agent.phi=prev_agent.phi
                agent.next_action=prev_agent.next_action
                agent._logger=prev_agent._logger
            else:
                start = 0

            # Algorithm
            core = Core(agent, mdp)

            # RUN
            eval_dataset = core.evaluate(n_steps=n_steps_test, render=render)
            s, *_ = parse_dataset(eval_dataset)
            J = np.mean(compute_J(eval_dataset, mdp.info.gamma))
            R = np.mean(compute_J(eval_dataset))
            E = agent.policy.entropy(s)

            core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, render=render)
            logger.epoch_info(0, J=J, R=R, entropy=E)
            exp_eval_dataset.append({"Epoch": 0, "J": J, "R": R, "entropy": E})

            for n in trange(start, n_epochs, leave=False):
                core.learn(n_steps=n_steps, n_steps_per_fit=1, render=render)
                agent._kl_on_q_alpha = max(agent._kl_on_q_alpha-kl_on_q_alpha_decay,0.0) # decay the kl-divergence reward weight alpha
                agent._kl_on_pi_alpha = max(agent._kl_on_pi_alpha-kl_on_pi_alpha_decay,0.0) # decay the kl-divergence loss weight alpha

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
                                Avg_episode_length=Avg_episode_length)
                if(log_agent):
                    logger.log_agent(agent, epoch=n+1) # Log agent every epoch
                    # logger.log_best_agent(agent, J) # Log best agent
                exp_eval_dataset.append({"Epoch": n+1, "Success_rate": Success_rate, "J": J, "R": R, "entropy": E, 
                                        "Q_loss": Q_loss, "Actor_loss": Actor_loss, "KL_with_prior": KL_with_prior, "Avg_episode_length": Avg_episode_length})
                gc.collect()

                if (n == 2): # log video once just to test
                    img_dataset = core.evaluate(n_episodes=80, get_renders=True)
                    vid_log_wandb(exp_config=exp_config, run_name=logger._log_id, group_name=exp_name, img_dataset=img_dataset)
            # Get video snippet of final learnt behavior
            img_dataset = core.evaluate(n_episodes=50, get_renders=True)
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
    parser = argparse.ArgumentParser("BHyRL Tiago LabRoom Reach")
    parser.add_argument("--test", default=False, action="store_true", help="Test an agent only")
    parser.add_argument("--test_agent", type=str, default="", help="Filename (with path) of the agent\'s .msh file")
    parser.add_argument("--render", default=False, action="store_true", help="Render environment")
    parser.add_argument("--test_log_video", default=False, action="store_true", help="Render environment")
    parser.add_argument("--log_agent", default=True, action="store_true", help="Log agent every epoch")
    parser.add_argument("--num_seq_seeds", type=int, default=2, help="Number of sequential seeds to run")
    args, unknown = parser.parse_known_args()

    if(args.test):
        if not args.test_agent:
            print("Error. Missing test agent.msh file string.")
            exit()

    experiment(test_only=args.test, test_agent_path=args.test_agent, test_log_video=args.test_log_video, num_seq_seeds=args.num_seq_seeds, 
               render=args.render, log_agent=args.log_agent, train_prev_agent=True, exp_name='2022-07-14-17-02-54')#datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    # # For running on cluster with experiment_launcher:
    # from experiment_launcher import run_experiment
    # run_experiment(experiment)