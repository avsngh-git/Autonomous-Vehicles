
import sys
import os
import yaml
import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env_wrapper import WaymoImitationEnv
from src.algorithms import BC_PPO

# CONFIGURATION
CONFIG = {
    "data_directory": "data/waymo_processed",
    "logs": "./logs/",
    "models": "./models/",
    "total_timesteps": 200_000
}

def main():
    # Ensure output dirs exist
    os.makedirs(CONFIG["logs"], exist_ok=True)
    os.makedirs(CONFIG["models"], exist_ok=True)

    # 1. Setup Environment Config for MetaDrive
    env_config = {
        "use_render": False,
        "data_directory": os.path.abspath(CONFIG['data_directory']),
        "num_scenarios": 3, 
        "horizon": 500,
        "start_scenario_index": 0,
        "show_coordinates": True,
        "vehicle_config": {
            "lidar": {"num_lasers": 60, "distance": 50, "num_others": 0},
        }
    }

    print(f"🔌 Connecting to Environment with data at: {env_config['data_directory']}")
    
    try:
        env = WaymoImitationEnv(env_config)
        print("✅ Environment Initialized Successfully")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        return

    # 2. Define Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Training on: {device}")

    model = BC_PPO(
        "MlpPolicy",   # <--- FIXED: Use MlpPolicy for Box observations
        env,
        verbose=1,
        bc_coef=0.5,
        learning_rate=3e-4,
        tensorboard_log=CONFIG['logs'],
        device=device
    )
    
    # 3. Train
    print("🧠 Starting Training Loop...")
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=CONFIG['models'], 
        name_prefix='bc_ppo'
    )
    
    model.learn(
        total_timesteps=CONFIG['total_timesteps'], 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    final_path = os.path.join(CONFIG['models'], "final_waymo_agent")
    model.save(final_path)
    print(f"🏆 Training Finished. Model Saved to {final_path}")

if __name__ == "__main__":
    main()
