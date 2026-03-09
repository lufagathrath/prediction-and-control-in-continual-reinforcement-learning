import numpy as np
from gym_wrapper import BaseEnv

def CL_envs_func(env_name, seed, switch_count=None):
	if env_name == "all":
		all_envs = ["breakout", "space_invaders", "freeway"]
		if switch_count is not None:
			sample_env = all_envs[switch_count % len(all_envs)]
		else:
			sample_env = np.random.choice(all_envs)
		return BaseEnv(sample_env, seed=seed, use_minimal_action_set=False, use_minimal_observation=False)

	else:
		raise NotImplementedError