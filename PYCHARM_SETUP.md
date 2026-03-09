# PyCharm Setup Guide

How to get the MinAtar PT-DQN experiments running after a fresh clone.

---

## 1. Clone the repository

```bash
git clone https://github.com/<owner>/prediction-and-control-in-continual-reinforcement-learning.git
cd prediction-and-control-in-continual-reinforcement-learning
git submodule update --init --recursive
```

The submodule init is needed for the JellyBeanWorld dependency. If you only care about MinAtar experiments, you can skip it.

## 2. Open in PyCharm

**File -> Open** and select the `prediction-and-control-in-continual-reinforcement-learning` root directory.

## 3. Create a Python virtual environment

1. Go to **Settings -> Project -> Python Interpreter**
2. Click the gear icon -> **Add Interpreter -> Add Local Interpreter**
3. Select **Virtualenv Environment -> New**
4. Choose **Python 3.8, 3.9, or 3.10** as the base interpreter (3.10 recommended)

> Python 3.11+ may cause issues with some pinned dependencies.

## 4. Install dependencies

Open the **Terminal** tab at the bottom of PyCharm (it will auto-activate your venv) and run:

```bash
pip install numpy torch torchvision
pip install minatar
pip install gym==0.23.0
pip install matplotlib seaborn
```

### Why not `pip install -r requirements.txt`?

The `requirements.txt` pins exact versions from the original authors' environment, including `torch==1.13.0.dev20220915` (a nightly build) and `jbw==1.0` (requires manual compilation). Installing the core packages above is more reliable.

If you want to try the full requirements file anyway:

```bash
pip install -r requirements.txt
```

You will likely need to fix the torch version manually if it fails:

```bash
pip install torch>=1.13.0
```

## 5. Mark source roots

The MinAtar scripts use bare imports (`from model import *`, `from replay import *`, `from gym_wrapper import BaseEnv`). PyCharm needs to know where these modules live.

1. In the **Project** panel, right-click `control/minatar_crl`
2. Select **Mark Directory as -> Sources Root**

This resolves red-underline import errors and enables code navigation/autocomplete for these modules.

## 6. Create a Run Configuration

1. Go to **Run -> Edit Configurations**
2. Click **+** -> **Python**
3. Fill in:

| Field | Value |
|-------|-------|
| **Name** | `PT-DQN (standard)` |
| **Script path** | `<project>/control/minatar_crl/PT_DQN_half.py` |
| **Working directory** | `<project>/control/minatar_crl` |
| **Parameters** | `--env-name=all --t-steps=3500000 --switch=500000 --lr1=1e-8 --lr2=1e-4 --decay=0.75 --seed=0 --save` |

> **The working directory MUST be `control/minatar_crl`**, not the project root. The script reads `misc_params.cfg` via a relative path and uses relative imports.

### Quick test run (shorter)

To verify everything works without waiting hours, create a second configuration:

| Field | Value |
|-------|-------|
| **Name** | `PT-DQN (quick test)` |
| **Script path** | `<project>/control/minatar_crl/PT_DQN_half.py` |
| **Working directory** | `<project>/control/minatar_crl` |
| **Parameters** | `--env-name=all --t-steps=10000 --switch=3000 --lr1=1e-8 --lr2=1e-4 --decay=0.75 --seed=0 --save` |

This runs 10k steps instead of 3.5M and finishes in under a minute.

### Constant P-Net ablation run

To run the ablation experiment (P-Net replaced with constant 0):

| Field | Value |
|-------|-------|
| **Name** | `PT-DQN (ablation, constant P-Net=0)` |
| **Script path** | `<project>/control/minatar_crl/PT_DQN_half.py` |
| **Working directory** | `<project>/control/minatar_crl` |
| **Parameters** | `--env-name=all --t-steps=3500000 --switch=500000 --lr1=1e-8 --lr2=1e-4 --decay=0.75 --seed=0 --save --constant-pnet 0` |

## 7. GPU support

PyTorch auto-detects CUDA. No PyCharm configuration is needed. The script selects the device with:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If you have a GPU and want to confirm it's being used, add this to your run parameters or check in the terminal:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 8. Output locations

All output is written to `control/minatar_crl/results/` (auto-created on first run):

| File | Contents |
|------|----------|
| `*_training.csv` | T-Net and P-Net loss (aggregated every 1000 steps) |
| `*_training_supp.csv` | Q-value means, weight norms, contribution ratios |
| `*_episodes.csv` | Per-episode returns, lengths, environment info |
| `*_p_net_training.csv` | Per-distillation P-Net loss |

The original pickle output (exponential moving average of returns) is also saved here.

## 9. Running other algorithms

The shell script `control/minatar_crl/run_minatar.sh` shows all five algorithms and their hyperparameters. To run the baselines, create similar Run Configurations for:

| Script | Parameters |
|--------|-----------|
| `DQN.py` | `--env-name=all --t-steps=3500000 --switch=500000 --lr1=1e-5 --seed=0 --save` |
| `DQN_multi_task.py` | `--env-name=all --t-steps=3500000 --switch=500000 --lr1=1e-5 --seed=0 --save` |
| `DQN_large_buffer.py` | `--env-name=all --t-steps=3500000 --switch=500000 --lr1=1e-4 --seed=0 --save` |
| `random_act.py` | `--env-name=all --t-steps=3500000 --switch=500000 --seed=0 --save` |

All use the same working directory: `control/minatar_crl`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'model'` | Set working directory to `control/minatar_crl` and mark it as Sources Root |
| `FileNotFoundError: misc_params.cfg` | Working directory is wrong; must be `control/minatar_crl` |
| `torch` version conflict | `pip install torch>=1.13.0` instead of the exact pinned version |
| `minatar` import error | `pip install minatar` separately |
| Red underlines in editor but script runs fine | Mark `control/minatar_crl` as Sources Root (step 5) |
