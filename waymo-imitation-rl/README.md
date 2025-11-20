# Imitation Is Not Enough: BC-PPO for Autonomous Driving

**A Robust Imitation Learning Implementation using Waymo Open Dataset & MetaDrive**

## 🚗 Project Overview
This project replicates the core concepts of the paper *"Imitation Is Not Enough: Robustifying Imitation with Reinforcement Learning"*. It tackles the issue of **covariate shift** in autonomous driving by combining **Behavior Cloning (BC)** with **Reinforcement Learning (PPO)**.

## 🛠 Tech Stack
- **Manager:** [uv](https://github.com/astral-sh/uv) (Fast Python package installer)
- **Simulator:** MetaDrive (ScenarioNet)
- **Data:** Waymo Open Motion Dataset
- **Algorithm:** PPO (Proximal Policy Optimization) + Behavior Cloning

## 🚀 Getting Started

### 1. Prerequisites
Install \`uv\`:
\`\`\`bash
curl -LsSf https://astral.sh/uv/install.sh | sh
\`\`\`

### 2. Installation & Sync
Initialize the virtual environment and install dependencies instantly:
\`\`\`bash
uv sync
\`\`\`

### 3. Data Preparation
Download Waymo Motion Dataset (.tfrecord files) to \`data/waymo_raw/\`, then run:
\`\`\`bash
uv run scripts/convert_data.py --raw data/waymo_raw --out data/waymo_processed
\`\`\`

### 4. Training
Run the training script inside the environment:
\`\`\`bash
uv run scripts/train.py
\`\`\`

## 📂 Structure
- \`pyproject.toml\`: Project dependencies managed by uv.
- \`src/\`: Custom PPO implementation and Environment wrappers.
- \`scripts/\`: Entry points for data conversion and training.

## 📜 References
- Lu et al., "Imitation Is Not Enough", IROS.
- Knox et al., "Reward (Mis)design for Autonomous Driving".
