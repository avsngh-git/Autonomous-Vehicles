import gymnasium as gym
import numpy as np
import os
import glob
import pickle
from metadrive.envs.scenario_env import ScenarioEnv
from src.utils import get_expert_action

class DirectWaymoEnv(gym.Wrapper):
    def __init__(self, config):
        data_dir = config.get("data_directory")
        
        # 1. Scan files ourselves
        self.scenario_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        self.scenario_files = [f for f in self.scenario_files if "dataset_summary" not in f]
        self.scenario_files.sort()
        
        if len(self.scenario_files) == 0:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")

        # 2. Config for MetaDrive
        # We tell MetaDrive to look at the dir, so it finds the summary file we just made
        md_config = config.copy()
        # IMPORTANT: Set num_scenarios to 1 so it doesn't try to load files that aren't in the summary
        md_config["num_scenarios"] = 1 
        
        env = ScenarioEnv(md_config)
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        try:
            if self.env.engine and self.env.engine.map_manager:
                expert_traj = self.env.engine.map_manager.current_sdc_route
                if expert_traj is not None and len(expert_traj) > 0:
                    target_pos = expert_traj[-1] 
                    expert_action = get_expert_action(self.env.vehicle, target_pos)
                else:
                    expert_action = np.zeros(2)
                info['expert_action'] = expert_action
            else:
                info['expert_action'] = np.zeros(2)
        except Exception:
            info['expert_action'] = np.zeros(2)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # 1. Ensure Engine is Ready
        if self.env.engine is None:
            # This triggers the standard load using the Summary File
            # Since the summary is valid (pointing to 1 real file), this succeeds.
            self.env.lazy_init()

        # 2. Select our target file
        if seed is None:
            seed = self.env.np_random.integers(0, len(self.scenario_files))
        
        file_index = seed % len(self.scenario_files)
        file_path = self.scenario_files[file_index]

        # 3. Manual Load & Inject
        try:
            with open(file_path, "rb") as f:
                scenario_data = pickle.load(f)
                
            # --- THE STEALTH SWAP ---
            # We overwrite the data manager's internal state just before reset
            self.env.engine.data_manager.current_scenario_data = scenario_data
            self.env.engine.data_manager.current_scenario_file_name = os.path.basename(file_path)
            # We also trick the manager into thinking it "randomly selected" this file
            # by setting internal indices if necessary, but injecting data is usually enough.
            
        except Exception as e:
            print(f"❌ Read Error {file_path}: {e}")
        
        # 4. Reset
        # MetaDrive sees 'current_scenario_data' is populated and uses it
        return self.env.reset(seed=seed)