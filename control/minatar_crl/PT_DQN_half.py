import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import copy
import csv
import os
import math
import collections

from model import *
from replay import *
from CL_envs import *

from argparse import ArgumentParser
from configparser import ConfigParser

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=50000000, help="total number of steps")
parser.add_argument('--switch', type=int, default=5000000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--lr2', type=float, default=0.1, help="learning rate for transient values")
parser.add_argument('--update', type=int, default=50000, help="PM update frequency")
parser.add_argument('--decay', type=float, default=0, help="decay transient weights after transfer")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument('--constant-pnet', type=float, default=None, help="Replace P-Net with constant value (e.g. 0)")
parser.add_argument('--log-interval', type=int, default=1000, help="Diagnostic CSV logging interval in steps")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])

def train_T_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		T_next_pred = Target_net(next_states)
		if args.constant_pnet is not None:
			P_next_pred = torch.full_like(T_next_pred, args.constant_pnet)
			P_pred = torch.full_like(T_next_pred, args.constant_pnet)
		else:
			P_next_pred = P_Net(next_states)
			P_pred = P_Net(states)
		P_pred = P_pred.gather(1, actions)
	T_pred = T_Net(states)
	T_pred = T_pred.gather(1, actions)
	targets = rewards + (1 - done) * gamma * ((P_next_pred + T_next_pred).max(1)[0]).reshape(-1, 1)
	loss = T_criterion(T_pred+P_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net():
	loss_u = 0
	u_steps = (exp_replay_PM.size()//args.batch_size) - 1
	for p_update in range(u_steps):
		curr_batch = list(itertools.islice(exp_replay_PM.memory, p_update*args.batch_size, (p_update+1)*args.batch_size))
		states, actions, old_p_vals = map(torch.stack, zip(*curr_batch))
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			T_pred = T_Net(states).gather(1, actions)
		P_pred = P_Net(states).gather(1, actions)
		loss = P_criterion(P_pred, T_pred+old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

def get_action(c_obs):
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_T_vals = T_Net(c_obs.unsqueeze(0))
		if args.constant_pnet is not None:
			curr_P_vals = torch.full_like(curr_T_vals, args.constant_pnet)
		else:
			curr_P_vals = P_Net(c_obs.unsqueeze(0))
		curr_Q_vals = curr_T_vals + curr_P_vals
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	# Compute supplementary metrics
	t_q_mean = curr_T_vals.mean().item()
	p_q_mean = curr_P_vals.mean().item()
	denom = curr_Q_vals.mean().item()
	t_ratio = t_q_mean / denom if abs(denom) >= 1e-8 else float('nan')
	t_pt_agree = 1.0 if curr_T_vals.argmax().item() == curr_Q_vals.argmax().item() else 0.0
	metrics = {
		'val_p': curr_P_vals[0][action],
		't_q_mean': t_q_mean,
		'p_q_mean': p_q_mean,
		't_ratio': t_ratio,
		't_pt_agree': t_pt_agree,
	}
	return action, metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("mps")#
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "PT_DQN_0.5x"+"_env_name_"+str(args.env_name)+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_update_"+str(args.update)+"_decay_"+str(args.decay)+\
		"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_batch_"+str(args.batch_size)+"_seed_"+str(args.seed)
if args.constant_pnet is not None:
	filename += "_constant_pnet_" + str(args.constant_pnet)

switch_count = 0
env = CL_envs_func(args.env_name, args.seed, switch_count)
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

T_Net = CNN_half(in_channels, num_actions).to(device)
T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)
T_criterion = torch.nn.MSELoss()

P_Net = CNN_half(in_channels, num_actions).to(device)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()

Target_net = CNN_half(in_channels, num_actions).to(device)
Target_net.load_state_dict(T_Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)
exp_replay_PM = expReplay_PM(max_size=args.update, batch_size=args.batch_size, device=device)

returns_array = np.zeros(args.t_steps)

# CSV logging setup
os.makedirs("results", exist_ok=True)
training_csv = open("results/" + filename + "_training.csv", "w", newline="")
training_writer = csv.writer(training_csv)
training_writer.writerow(["step", "t_net_loss", "p_net_loss"])

training_supp_csv = open("results/" + filename + "_training_supp.csv", "w", newline="")
training_supp_writer = csv.writer(training_supp_csv)
training_supp_writer.writerow(["step", "t_net_q_mean", "p_net_q_mean", "t_contribution_ratio", "t_pt_agreement_rate", "t_net_weight_norm", "p_net_weight_norm", "env_name"])

episodes_csv = open("results/" + filename + "_episodes.csv", "w", newline="")
episodes_writer = csv.writer(episodes_csv)
episodes_writer.writerow(["step", "episode", "return_", "length", "avg_return_100", "env_name", "env_switch"])

pnet_training_csv = open("results/" + filename + "_p_net_training.csv", "w", newline="")
pnet_training_writer = csv.writer(pnet_training_csv)
pnet_training_writer.writerow(["step", "p_net_loss"])

# Metric accumulators
t_net_losses = []
p_net_losses_window = []
t_q_means = []
p_q_means = []
t_ratios = []
t_pt_agrees = []

# Episode tracking
recent_returns = collections.deque(maxlen=100)
episode_count = 0
epi_length = 0
env_switch_flag = 0
cur_env_name = env.game_name

avg_return = 0
epi_return = 0
done = False
cs = env.reset()

for step in range(args.t_steps):

	if (step+1)%args.switch == 0:
		switch_count += 1
		env = CL_envs_func(args.env_name, args.seed, switch_count)
		cs = env.reset()
		epi_return = 0
		epi_length = 0
		env_switch_flag = 1
		cur_env_name = env.game_name

	c_action, metrics = get_action(cs)
	val_p = metrics['val_p']
	t_q_means.append(metrics['t_q_mean'])
	p_q_means.append(metrics['p_q_mean'])
	t_ratios.append(metrics['t_ratio'])
	t_pt_agrees.append(metrics['t_pt_agree'])

	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	epi_length += 1
	exp_replay.store(cs, c_action, ns, rew, done)
	if args.constant_pnet is None:
		exp_replay_PM.store(cs, c_action, val_p)

	if exp_replay.size() >= args.batch_size:
		loss = train_T_Net()
		t_net_losses.append(loss)

	cs = ns

	if (step+1)%args.update == 0:
		if args.constant_pnet is None:
			p_loss = train_P_Net()
			p_net_losses_window.append(p_loss)
			pnet_training_writer.writerow([step+1, p_loss])
			pnet_training_csv.flush()
		for params in T_Net.parameters():
			params.data *= args.decay

	if (step+1)%1000 == 0:
		Target_net.load_state_dict(T_Net.state_dict())

	# Diagnostic CSV logging
	if (step+1) % args.log_interval == 0:
		t_loss_mean = sum(t_net_losses) / len(t_net_losses) if t_net_losses else float('nan')
		p_loss_val = sum(p_net_losses_window) / len(p_net_losses_window) if p_net_losses_window else float('nan')
		training_writer.writerow([step+1, t_loss_mean, p_loss_val])
		training_csv.flush()

		t_q_mean_agg = sum(t_q_means) / len(t_q_means) if t_q_means else float('nan')
		p_q_mean_agg = sum(p_q_means) / len(p_q_means) if p_q_means else float('nan')
		valid_ratios = [r for r in t_ratios if not math.isnan(r)]
		t_ratio_agg = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float('nan')
		t_pt_agree_agg = sum(t_pt_agrees) / len(t_pt_agrees) if t_pt_agrees else float('nan')
		t_weight_norm = torch.cat([p.data.flatten() for p in T_Net.parameters()]).norm().item()
		if args.constant_pnet is not None:
			p_weight_norm = 0.0
		else:
			p_weight_norm = torch.cat([p.data.flatten() for p in P_Net.parameters()]).norm().item()
		training_supp_writer.writerow([step+1, t_q_mean_agg, p_q_mean_agg, t_ratio_agg, t_pt_agree_agg, t_weight_norm, p_weight_norm, cur_env_name])
		training_supp_csv.flush()

		t_net_losses.clear()
		p_net_losses_window.clear()
		t_q_means.clear()
		p_q_means.clear()
		t_ratios.clear()
		t_pt_agrees.clear()

	if done:
		cs = env.reset()
		avg_return = 0.99 * avg_return + 0.01 * epi_return
		recent_returns.append(epi_return)
		avg_return_100 = sum(recent_returns) / len(recent_returns)
		episodes_writer.writerow([step+1, episode_count, epi_return, epi_length, avg_return_100, cur_env_name, env_switch_flag])
		episodes_csv.flush()
		episode_count += 1
		epi_return = 0
		epi_length = 0
		env_switch_flag = 0

	returns_array[step] = copy.copy(avg_return)

training_csv.close()
training_supp_csv.close()
episodes_csv.close()
pnet_training_csv.close()

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_array, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_array, f)