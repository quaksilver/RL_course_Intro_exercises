# Deep Q-Network (DQN) for CartPole-v1

This project implements a Deep Q-Network (DQN) agent to solve the `CartPole-v1` environment using PyTorch and the Gymnasium library. The agent learns through experience replay and uses a target network to stabilize training.

---

## Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Or manually install the key packages:

```bash
pip install gymnasium[classic_control] torch matplotlib
```

---

## Running the Code

You can run the full training session locally with:

```bash
python DQN.py
```

If you're using **Google Colab**, clone this repo and run:

```python
!git clone https://github.com/quaksilver/RL_course_Intro_exercises.git
%cd RL_course_Intro_exercises
!pip install -r requirements.txt
!python DQN.py
```

---

## Project Structure

- `DQN.py` – main training script that builds, trains, and evaluates the DQN agent
- `requirements.txt` – list of required Python packages
- `run1.png` – output image from the agent’s training (optional for visual analysis)

---

## Core Concepts

- **Environment**: `CartPole-v1` from Gymnasium
- **Replay Memory**: Stores transitions to break correlation between samples
- **DQN Architecture**:
  - 2 hidden layers of 128 neurons each
  - ReLU activations
- **Target Network**: Soft-updated clone of the policy network to stabilize learning
- **Epsilon-Greedy Policy**: Controls exploration with exponential decay

---

## Hyperparameters

| Parameter      | Value      |
|----------------|------------|
| `BATCH_SIZE`   | 128        |
| `GAMMA`        | 0.99       |
| `EPS_START`    | 0.9        |
| `EPS_END`      | 0.01       |
| `EPS_DECAY`    | 2500       |
| `TAU`          | 0.005      |
| `LR`           | 3e-4       |
| `Episodes`     | 50 (CPU) or 600 (GPU/MPS) |

---

## Output

During training, the script live-plots the duration of each episode and computes the rolling average over the last 100 episodes.

---

## 📌 Notes

- Reproducibility: Seed-setting code is included but commented out.
- Device-agnostic: Automatically uses GPU (CUDA or MPS) if available.


