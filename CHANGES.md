# Diff Report: PT-DQN Ablation Study & Diagnostic Logging

**Branch:** `claude/pt-dqn-ablation-study-1WUFu`
**Base:** `origin/main`
**Commits:** 3 (excluding merge base)

---

## Summary

This branch adds two capabilities to the PT-DQN MinAtar experiment:

1. **Constant P-Net ablation** (`--constant-pnet <float>`) — replaces the learned P-Net with a fixed constant, isolating the effect of T-Net's periodic weight decay
2. **Diagnostic CSV logging** — writes four CSV files with training losses, Q-value diagnostics, episode metrics, and P-Net distillation events

Additionally, environment switching was changed from random to deterministic cycling for reproducibility.

---

## Files Changed

### 1. `.gitignore` (new file)

```
+__pycache__/
```

Prevents Python bytecode cache directories from being tracked.

---

### 2. `control/minatar_crl/CL_envs.py`

**Lines changed:** +6, -2

#### Function signature

```python
# Before
def CL_envs_func(env_name, seed):

# After
def CL_envs_func(env_name, seed, switch_count=None):
```

#### Deterministic environment cycling

When `env_name == "all"`, the original code selected a random environment on each call:

```python
# Before
sample_env = np.random.choice(all_envs)
```

The new code cycles deterministically when `switch_count` is provided:

```python
# After
if switch_count is not None:
    sample_env = all_envs[switch_count % len(all_envs)]
else:
    sample_env = np.random.choice(all_envs)
```

The cycle order is: `breakout` (0) -> `space_invaders` (1) -> `freeway` (2) -> `breakout` (3) -> ...

**Backward compatibility:** The default `switch_count=None` preserves the original random behavior for all other scripts (`DQN.py`, `DQN_multi_task.py`, `DQN_large_buffer.py`, `random_act.py`).

---

### 3. `control/minatar_crl/PT_DQN_half.py`

**Lines changed:** +134, -12

This is the main file with substantial modifications. Changes are organized by section.

#### 3a. New imports (line 7-10)

```python
+import csv
+import os
+import math
+import collections
```

#### 3b. New CLI arguments (line 32-33)

```python
+parser.add_argument('--constant-pnet', type=float, default=None,
+                    help="Replace P-Net with constant value (e.g. 0)")
+parser.add_argument('--log-interval', type=int, default=1000,
+                    help="Diagnostic CSV logging interval in steps")
```

- `--constant-pnet`: When set, P-Net is never evaluated. Instead, all P-Net outputs are replaced with `torch.full_like(..., value)`. The P-Net replay buffer (`exp_replay_PM`) is not populated, and `train_P_Net()` is never called.
- `--log-interval`: Controls how often aggregated training metrics are flushed to CSV (default: every 1000 steps).

#### 3c. `train_T_Net()` modification (lines 46-51)

```python
# Before
P_next_pred = P_Net(next_states)
P_pred = P_Net(states)

# After
if args.constant_pnet is not None:
    P_next_pred = torch.full_like(T_next_pred, args.constant_pnet)
    P_pred = torch.full_like(T_next_pred, args.constant_pnet)
else:
    P_next_pred = P_Net(next_states)
    P_pred = P_Net(states)
```

When constant P-Net is active, `T_next_pred` (already computed from `Target_net`) provides the correct shape `(batch_size, num_actions)` for `torch.full_like`. The rest of the Bellman target computation proceeds unchanged — the constant simply shifts the target values uniformly.

#### 3d. `get_action()` return signature change (lines 83-108)

**Before:** Returns `(val_p, action)` — a single P-Net value and the chosen action.

**After:** Returns `(action, metrics_dict)` — the action first, then a dictionary:

```python
metrics = {
    'val_p':      curr_P_vals[0][action],   # P-Net value for chosen action
    't_q_mean':   curr_T_vals.mean().item(), # Mean T-Net Q-value across actions
    'p_q_mean':   curr_P_vals.mean().item(), # Mean P-Net Q-value across actions
    't_ratio':    t_q_mean / denom,          # T-Net contribution to composite Q
    't_pt_agree': 1.0 or 0.0,               # Does T-Net agree with composite on best action?
}
```

When `--constant-pnet` is active, `curr_P_vals` is `torch.full_like(curr_T_vals, args.constant_pnet)` instead of `P_Net(...)`.

The `t_ratio` metric is `NaN` when `abs(denom) < 1e-8` to avoid division by zero.

#### 3e. Output filename extension (lines 120-121)

```python
+if args.constant_pnet is not None:
+    filename += "_constant_pnet_" + str(args.constant_pnet)
```

Example: `PT_DQN_0.5x_env_name_all_gamma_0.99_..._seed_0_constant_pnet_0.0`

#### 3f. Deterministic environment switching (lines 123-124, 160-165)

```python
+switch_count = 0
+env = CL_envs_func(args.env_name, args.seed, switch_count)
```

At each switch boundary:
```python
# Before
env = CL_envs_func(args.env_name, args.seed)

# After
switch_count += 1
env = CL_envs_func(args.env_name, args.seed, switch_count)
```

`cur_env_name` is tracked via `env.game_name` and written to CSV logs.

#### 3g. CSV logging infrastructure (lines 145-176)

Four CSV files are opened at startup in a `results/` directory (auto-created via `os.makedirs`):

| CSV File | Written | Columns |
|----------|---------|---------|
| `{filename}_training.csv` | Every `log_interval` steps | `step`, `t_net_loss`, `p_net_loss` |
| `{filename}_training_supp.csv` | Every `log_interval` steps | `step`, `t_net_q_mean`, `p_net_q_mean`, `t_contribution_ratio`, `t_pt_agreement_rate`, `t_net_weight_norm`, `p_net_weight_norm`, `env_name` |
| `{filename}_episodes.csv` | Every episode end | `step`, `episode`, `return_`, `length`, `avg_return_100`, `env_name`, `env_switch` |
| `{filename}_p_net_training.csv` | Every P-Net distillation | `step`, `p_net_loss` |

Metric accumulators (lists) are cleared after each CSV flush. Weight norms are point-in-time snapshots.

#### 3h. Main loop changes (lines 178-260)

**Metric accumulation per step:**
```python
c_action, metrics = get_action(cs)
val_p = metrics['val_p']
t_q_means.append(metrics['t_q_mean'])
p_q_means.append(metrics['p_q_mean'])
t_ratios.append(metrics['t_ratio'])
t_pt_agrees.append(metrics['t_pt_agree'])
```

**Gated P-Net operations:**
```python
# Replay buffer store — skipped when constant P-Net
if args.constant_pnet is None:
    exp_replay_PM.store(cs, c_action, val_p)

# P-Net training — skipped when constant P-Net
if (step+1) % args.update == 0:
    if args.constant_pnet is None:
        p_loss = train_P_Net()
        p_net_losses_window.append(p_loss)
        pnet_training_writer.writerow([step+1, p_loss])
```

**Episode tracking:**
```python
# New variables
episode_count = 0
epi_length = 0
env_switch_flag = 0
recent_returns = collections.deque(maxlen=100)

# On episode end (done == True)
recent_returns.append(epi_return)
avg_return_100 = sum(recent_returns) / len(recent_returns)
episodes_writer.writerow([step+1, episode_count, epi_return, epi_length,
                          avg_return_100, cur_env_name, env_switch_flag])
episode_count += 1
epi_return = 0
epi_length = 0
env_switch_flag = 0
```

**Diagnostic CSV flush (every `log_interval` steps):**
- Computes mean of accumulated T-Net losses, P-Net losses, Q-value means, ratios, agreement rates
- Computes weight norms as snapshots: `torch.cat([p.data.flatten() for p in T_Net.parameters()]).norm()`
- P-Net weight norm is `0.0` when `--constant-pnet` is active
- NaN ratios are filtered before averaging
- All accumulator lists are cleared after flush

**File handle cleanup:**
```python
+training_csv.close()
+training_supp_csv.close()
+episodes_csv.close()
+pnet_training_csv.close()
```

---

## What is NOT changed

- `model.py`, `replay.py`, `gym_wrapper.py` — untouched
- `DQN.py`, `DQN_multi_task.py`, `DQN_large_buffer.py`, `random_act.py` — untouched (backward-compatible `CL_envs_func` default)
- `run_minatar.sh` — untouched (ablation runs require separate invocation)
- `train_P_Net()` function body — untouched (just conditionally skipped)
- Original pickle output (`returns_array`) — untouched
- Hyperparameters (`gamma`, `epsilon`, learning rates, decay) — no defaults changed
