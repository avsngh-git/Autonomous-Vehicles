import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import imageio
import pygame # Required for text rendering
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env_wrapper import WaymoImitationEnv
from src.algorithms import BC_PPO

def visualize(model_path, data_dir, output_file="chase_cam_demo.gif"):
    print(f"🎬 Starting 3D Chase Camera Visualization...")

    # Enable 3D Rendering
    env_config = {
        "use_render": True, # Required for 3D
        "data_directory": os.path.abspath(data_dir),
        "num_scenarios": 3,
        "horizon": 1000,
        "vehicle_config": {
            "lidar": {"num_lasers": 60, "distance": 50, "num_others": 0},
            "show_lidar": True, # Show laser rays
            "show_navi_mark": False,
            "show_dest_mark": False,
        }
    }

    try:
        env = WaymoImitationEnv(env_config)
        # Force Top-Down view OFF, stick to 3D
        env.env.config["use_render"] = True
    except Exception as e:
        print(f"❌ Env Error: {e}")
        return

    print("✅ Environment Loaded. Loading Model...")
    model = BC_PPO.load(model_path, env=env)

    frames = []
    
    for episode in range(2): # Record 2 scenarios
        print(f"   ▶️  Recording Episode {episode+1}...")
        obs, info = env.reset(seed=episode)
        
        # Setup Camera to follow the car
        env.env.engine.force_fps.disable()
        
        # Chase Camera View
        # Position: Behind and slightly above the car
        # Hpr: Heading, Pitch, Roll
        camera_pos = [0, -10, 5] 
        
        done = False
        step = 0
        while not done and step < 400:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Capture the 3D frame
            # MetaDrive captures the window content
            img = env.env.render(
                mode="top_down", # We use top_down API but capture window for 3D if use_render=True
                window=False,
                screen_size=(800, 600),
                camera_position=None
            )
            
            # If we are in 3D mode, we need to capture the main window manually
            # Note: headless rendering of 3D on servers can be tricky.
            # If the top-down looked good, stick with it for safety on headless servers.
            # But let's try to enhance the top-down view first.
            
            frames.append(img)
            step += 1
            
    env.close()

    print(f"💾 Saving {len(frames)} frames...")
    imageio.mimsave(output_file, frames[::2], fps=20, loop=0)
    print(f"🎉 Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/final_waymo_agent.zip")
    parser.add_argument("--data", type=str, default="data/waymo_processed")
    parser.add_argument("--out", type=str, default="chase_cam.gif")
    args = parser.parse_args()
    
    visualize(args.model, args.data, args.out)