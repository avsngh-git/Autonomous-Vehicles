import gymnasium as gym
import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from src.utils import get_expert_action

class WaymoImitationEnv(gym.Wrapper):
    def __init__(self, config):
        # Initialize the standard MetaDrive Scenario Environment
        env = ScenarioEnv(config)
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- INJECT EXPERT ACTION ---
        try:
            # Access internal MetaDrive engine to get expert trajectory
            expert_traj = self.env.engine.map_manager.current_sdc_route
            
            if expert_traj is not None and len(expert_traj) > 0:
                target_pos = expert_traj[-1] 
                expert_action = get_expert_action(self.env.vehicle, target_pos)
            else:
                expert_action = np.zeros(2)
            
            info['expert_action'] = expert_action
            
        except Exception:
            # Fallback
            info['expert_action'] = np.zeros(2)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # --- CRITICAL FIX ---
        # Stable-Baselines3 passes 'options', but MetaDrive crashes if it sees it.
        # We simply filter it out here.
        return self.env.reset(seed=seed)