import os

import numpy as np
import torch
import math

import copy
import argparse
import h5py
import pdb

from mushroom_rl.algorithms.actor_critic import BHyRL # mixed action space
from mushroom_rl.algorithms.actor_critic import SAC_hybrid # mixed action space
from sac_hybrid_data_prior_tiago_dual_reaching import CriticNetwork, ActorNetwork # Need these networks to load the previous task's agent
from bhyrl_tiago_dual_room_reaching import BHyRL_CriticNetwork, BHyRL_ActorNetwork

ISAAC_PATH = os.environ.get('ISAAC_PATH')
results_dir=ISAAC_PATH+"/tiago_isaac/experiments/logs/Q_maps"
if torch.cuda.is_available():
    device = "cuda"
    print("[GPU MEMORY size in GiB]: " + str((torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))/1024**3))
else:
    device = "cpu"
dtype = torch.float64 # Choose float32 or 64 etc.

# Map settings
map_xy_resolution = 0.03 # in metres
map_xy_radius = 1.0 # metres
angular_res = np.pi/8 # or 22.5 degrees per bin)
Q_scaling = 100.0 # TODO: play with this
x_bins = math.ceil(map_xy_radius*2/map_xy_resolution)
y_bins = math.ceil(map_xy_radius*2/map_xy_resolution)
theta_bins = math.ceil((2*np.pi)/angular_res)
x_ind_offset = y_bins*theta_bins
y_ind_offset = theta_bins
theta_ind_offset = 1
num_voxels = x_bins*y_bins*theta_bins

# Load agents from prior tasks (since we are learning the residuals). NOTE: Mind the order of the prior tasks
prior_agents = list()
prior_agents.append(SAC_hybrid.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/SAC_hybrid/2022-02-17-02-23-19/2022-02-17-05-33-31/agent-123.msh')) # higher entropy
prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL/2022-02-20-17-59-58-5mEntropy/2022-02-20-18-02-05/agent-213.msh'))
# Agent
# prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL_labroom/2022-02-23-04-57-46/2022-02-23-04-59-18/agent-310.msh'))
prior_agents.append(BHyRL.load(ISAAC_PATH+'/tiago_isaac/experiments/logs/BHyRL_labroom/2022-04-08-09-09-12/2022-04-08-09-09-47/agent-242.msh'))

agent_net_batch_size = 128
agent_world_xy_radius = 3.0
agent_world_norm_theta = np.pi
agent_action_xy_radius = 1.0
agent_action_ang_lim = np.pi

# mdp get prior state function. Needed for Boosted RL agents
def get_prior_task_states(new_task_states, ik_task=False, reach_task=False): # convert the state from a new task's state space to the prior task's state space                
    if(len(new_task_states.shape) < 2):
        # If received state is a single array, unsqueeze it
        new_task_states = np.array([new_task_states])
    weights = np.ones(shape=new_task_states.shape[0]) # also return weighting for the states
    # Remove excess state variables as per prior task

    if(ik_task):
        prior_task_states = copy.deepcopy(new_task_states[:,0:6]) # only using the 6D goal state
        # Prior task is simple IK 6D reaching in a 1m world radius
        prior_task_states[:,0:3] *= agent_world_xy_radius # Re-scale xyz distances
        xy_distances = np.linalg.norm(prior_task_states[:,0:2], axis=1)
        # weights[np.linalg.norm(xy_distances > 1.0] = 0.0 # accept only states where xy distance is less than 1 metre
        # weights = np.clip(1/xy_distances, 0.0, 1.0) # weight as per 1/distance. Clip to maximum weight of 1.0
        weights = np.clip(1.0 - np.tanh(xy_distances-1.0), 0.0, 1.0) # Weights as per distance metric: 1-tanh. Clip to maximum weight of 1.0
    elif(reach_task):
        prior_task_states = copy.deepcopy(new_task_states[:,0:6]) # only using the 6D goal state
        # prior task is a 5m free reaching task
        prior_task_states[:,0:3] *= agent_world_xy_radius # Re-scale xyz distances
        # convert to distances for prior 5m reaching task
        prior_task_states[:,0:3] /= 5.0
    else:
        # Prior task is the same (final task)
        prior_task_states = copy.deepcopy(new_task_states)

    return weights, prior_task_states

# Get saved, un-transformed states from file:
with open(ISAAC_PATH+'/tiago_isaac/experiments/state_poses0.np','rb') as f:
    state_poses = np.load(f)

# Modify and get relevant states
grasp_obj_trans = state_poses[0,:]
grasp_obj_rot = state_poses[1,:]
obstacle_trans = state_poses[2,:]
obstacle_rot = state_poses[3,:]
# Find max, min vertices
max_xy_offset = np.array([0.55/2, 0.77/2])
min_xy_offset = -max_xy_offset
# rotate by theta
bbox_tf = np.zeros((3,3))
bbox_tf[:2,:2] = np.array([[np.cos(obstacle_rot[2]), -np.sin(obstacle_rot[2])],[np.sin(obstacle_rot[2]), np.cos(obstacle_rot[2])]])
bbox_tf[:,-1] = np.array([0.0, 0.0, 1.0])
max_xy_vertex = np.array([[max_xy_offset[0],max_xy_offset[1],1.0]]).T
min_xy_vertex = np.array([[min_xy_offset[0],min_xy_offset[1],1.0]]).T
new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze() + obstacle_trans[0:2]
new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze() + obstacle_trans[0:2]

obs_z = 0.48 # fixed
obs_theta = obstacle_rot[2]
# Generate grasp
grasp_obj_trans[2] += 0.08 # z offset for top grasps
# array([ 0.31196376,  0.82159081, -0.31096255])
# Adding z from ground: 1.0530189275741577
grasp_obj_trans[2] += 1.0530189275741577
print(f"grasp_obj_trans: {grasp_obj_trans}")
# array([ 0.31196376,  0.82159081, 0.7420563775741578])
grasp_obj_rot[1] += np.radians(88) # Pitch by almost 90 degrees
print(f"grasp_obj_rot: {grasp_obj_rot}")
# array([-0.00640782,  1.52538862,  2.32906671])

state = np.array([grasp_obj_trans[0], grasp_obj_trans[1], grasp_obj_trans[2],
                    grasp_obj_rot[0], grasp_obj_rot[1], grasp_obj_rot[2],
                    new_max_xy_vertex[0], new_max_xy_vertex[1], new_min_xy_vertex[0], new_min_xy_vertex[1], 
                    obs_z, obs_theta])
state_normalized = np.array([state])
state_normalized[0:3] /= agent_world_xy_radius
state_normalized[3:6] /= agent_world_norm_theta
state_normalized[6:-1] /= agent_world_xy_radius
state_normalized[-1] /= agent_world_norm_theta

# Optional: Query policy for optimal (in-theory) action:
act = prior_agents[-1].policy.draw_action(state_normalized)[0]
x_scaled = act[0]*np.cos(act[1]*np.pi)
y_scaled = act[0]*np.sin(act[1]*np.pi)
theta_scaled = np.degrees(act[2]*np.pi)
print(f"Action from policy: x:{x_scaled}, y:{y_scaled}, theta(deg):{theta_scaled}")

# Create base inverse reachability map:

# Actions: generate discrete actions (x-y-theta) around the robot as per resolution
# Use a map_xy_radius (try 1 or 1.2 metres) radius around the robot
# TODO: Figure out meshgrids so we don't have to use for loops
# xs = torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins)
# ys = torch.linspace(-map_xy_radius, map_xy_radius, steps=y_bins)
# thetas = torch.linspace(-np.pi, np.pi, steps=theta_bins)
# mesh_x, mesh_y, mesh_theta = torch.meshgrid(xs, ys, thetas, indexing='xy')
base_reach_map = torch.zeros((num_voxels,4), dtype=dtype, device=device) # 4 values in each row: x,y,theta and Q
idx = 0
for x_val in torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins):
    for y_val in torch.linspace(-map_xy_radius, map_xy_radius, steps=x_bins):
        for theta_val in torch.linspace(-np.pi, np.pi, steps=theta_bins):
            base_reach_map[idx,0:-1] = torch.hstack([x_val, y_val, theta_val])
            idx+=1

# Only take spheres in polar co-ordinates within polar co-ord range (radius)
valid_idxs = torch.linalg.norm(base_reach_map[:,0:2],dim=1) <= agent_action_xy_radius
base_reach_map = base_reach_map[valid_idxs]

# convert x-y to r-phi
action_query = torch.zeros(base_reach_map.shape[0],5, device=device) # This is as per the exact action space. For these agents it is 5 dim, 3 cont and 2 discrete
action_query[:,0] = torch.linalg.norm(base_reach_map[:,0:2],dim=1) # norm of x and y gives r
action_query[:,1] = torch.atan2(base_reach_map[:,1],base_reach_map[:,0]) # arctan2(y/x) gives phi
action_query[:,2] = base_reach_map[:,2].clone()
action_query[:,3] = torch.tensor([1.0],device=device)
# Normalize actions
action_query /= torch.tensor([agent_action_xy_radius,agent_action_ang_lim,agent_action_ang_lim,1.0,1.0], device=device)

# query agent's critics in batch and get the q values (scores)
# Loop over prior critics (boosting case)
# Unfortunately you can only have a batch size equal to what the networks use, so we have to loop
num_loops = math.ceil(action_query.shape[0]/agent_net_batch_size)
for n in range(num_loops):
    action_query_batch = action_query[n*agent_net_batch_size:(n+1)*agent_net_batch_size,:]
    for idx, prior_agent in enumerate(prior_agents):
        # Use weights for the prior rho values. Also use appropriate state-spaces as per the prior task
        weights, prior_state =  get_prior_task_states(state_normalized, ik_task=(idx==0), reach_task=(idx==1)) # task[0],task[1] are IK prior and 5m reaching tasks resp
        weights = torch.tensor(weights, device=device).repeat(action_query_batch.shape[0],1) # resize to be same as num of actions
        state_query_batch = torch.tensor(prior_state, device=device).repeat(action_query_batch.shape[0],1) # resize to be same as num of actions
        rho_prior = weights.squeeze() * prior_agent._target_critic_approximator.predict(state_query_batch, action_query_batch, output_tensor=True, prediction='min').values
        base_reach_map[n*agent_net_batch_size:(n+1)*agent_net_batch_size,-1] += rho_prior # Store Q values in base_reach_map

base_reach_map = base_reach_map.detach().cpu().numpy() # Move to numpy
# Accumulate 2D+orientation scores into every 2D voxel
indx = 0
first = True
while(indx < base_reach_map.shape[0]):
    sphere_2d = base_reach_map[indx][:2]
    # Count num_repetitions of current 2D sphere (in the next y_ind_offset subarray)
    num_repetitions = (base_reach_map[indx:indx+y_ind_offset][:,:2] == sphere_2d).all(axis=1).sum().astype(dtype=np.int16)    
    sphere_3d = np.hstack((sphere_2d, 0.0))
    # Store sphere and average Q as the score. (Also, scale by a factor)
    Q_avg = base_reach_map[indx:indx+num_repetitions, 3].mean()
    if first:
        first = False
        sphere_array = np.append(base_reach_map[indx][:2], [0.0, Q_avg])
        # sphere_array = np.append(reach_map_nonzero[indx][:3], num_repetitions) # Optional: Use num_repetitions as score instead
        pose_array = np.append(base_reach_map[indx][:2], np.array([0., 0., 0., 0., 0., 0., 0., 1.])).astype(np.single) # dummy value
    else:
        sphere_array = np.vstack((sphere_array, np.append(base_reach_map[indx][:2], [0.0, Q_avg])))
        # sphere_array = np.vstack((sphere_array, np.append(reach_map_nonzero[indx][:3], num_repetitions)))  # Optional: Use num_repetitions as score instead
        pose_array = np.vstack((pose_array, np.append(base_reach_map[indx][:2], np.array([0., 0., 0., 0., 0., 0., 0., 1.])).astype(np.single))) # dummy value
    indx += num_repetitions


# # Optional: Normalize Q values in the map
min_Q = sphere_array[:,-1].min()
max_Q = sphere_array[:,-1].max()
sphere_array[:,-1] -= min_Q
sphere_array[:,-1] /= (max_Q-min_Q)
sphere_array[:,-1] *= Q_scaling

# # Save 3D map as hdf5 file for visualizer (Mimic reuleux data structure)
with h5py.File(results_dir+"/3D_Q_state_poses0.h5", 'w') as f:
    sphereGroup = f.create_group('/Spheres')
    sphereDat = sphereGroup.create_dataset('sphere_dataset', data=sphere_array)
    sphereDat.attrs.create('Resolution', data=map_xy_resolution)
    # (Optional) Save all the 6D poses in each 3D sphere. Currently only dummy pose values (10 dimensional)
    poseGroup = f.create_group('/Poses')
    poseDat = poseGroup.create_dataset('poses_dataset', dtype=float, data=pose_array)