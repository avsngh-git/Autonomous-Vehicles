import sys
import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Import the NEW wrapper
from src.env_wrapper import DirectWaymoEnv 
from src.algorithms import BC_PPO

# CONFIGURATION
CONFIG = {
    "data_directory": "data/waymo_processed",
    "logs": "./logs/",
    "models": "./models/",
    "total_timesteps": 10_000_000, 
    "num_envs": 8,
}

def make_env(rank, seed=0):
    def _init():
        env_config = {
            "use_render": False,
            # We pass the dir, the wrapper will scan it manually
            "data_directory": os.path.abspath(CONFIG['data_directory']),
            "horizon": 500,
            "vehicle_config": {
                "lidar": {"num_lasers": 60, "distance": 50, "num_others": 0},
            }
        }
        # Use the Direct Loader
        env = DirectWaymoEnv(env_config)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    os.makedirs(CONFIG["logs"], exist_ok=True)
    os.makedirs(CONFIG["models"], exist_ok=True)

    print(f"🚀 Starting Parallel Training (Direct Loading Mode)...")

    # 1. Create Vectorized Environment
    # Note: We don't need to pre-count scenarios. Each worker scans the folder itself.
    env_fns = [make_env(i) for i in range(CONFIG['num_envs'])]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env) 

    # 2. Define Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Training on: {device}")

    model = BC_PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        bc_coef=0.5, 
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=1024,      
        tensorboard_log=CONFIG['logs'],
        device=device
    )
    
    # 3. Train
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=CONFIG['models'], 
        name_prefix='bc_ppo_direct'
    )
    
    try:
        model.learn(
            total_timesteps=CONFIG['total_timesteps'], 
            callback=checkpoint_callback,
            progress_bar=True
        )
        model.save(os.path.join(CONFIG['models'], "waymo_direct_final"))
        print("🏆 Training Finished.")
        
    except KeyboardInterrupt:
        print("🛑 Training stopped manually.")
        model.save(os.path.join(CONFIG['models'], "waymo_direct_interrupted"))
    finally:
        env.close()

if __name__ == "__main__":
    main()