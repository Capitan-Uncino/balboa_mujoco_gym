import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import mujoco.viewer 
import torch as th
from gymnasium.wrappers import TimeLimit

class RobustEvalCallback(EvalCallback):
    def __init__(self, *args, variance_penalty=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_robust_score = -np.inf
        self.variance_penalty = variance_penalty

    def _on_step(self) -> bool:
        # 1. Let the standard EvalCallback do its evaluation
        result = super()._on_step()
        
        # 2. Check if an evaluation actually happened this step
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Grab the raw rewards from the 5 evaluation episodes it just ran
            latest_rewards = self.evaluations_results[-1]
            mean = np.mean(latest_rewards)
            std = np.std(latest_rewards)
            
            # 3. Calculate our custom robust score
            robust_score = mean - (self.variance_penalty * std)
            
            # 4. Save the model if it is the most robust one we've seen
            if robust_score > self.best_robust_score:
                self.best_robust_score = robust_score
                print(f"--> New most robust policy! Score: {robust_score:.2f} (Mean: {mean:.2f}, Std: {std:.2f})")
                
                if self.best_model_save_path is not None:
                    robust_path = os.path.join(self.best_model_save_path, "robust_model")
                    self.model.save(robust_path)
                    
        return result



class BalboaEnv(gym.Env):
    def __init__(self):
        super(BalboaEnv, self).__init__()
        
        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path("balboa.xml")
        self.data = mujoco.MjData(self.model)
        
        # Action Space: Normalized torque [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: [phi, theta, dphi, dtheta]
        # We use large bounds because the system can technically spin fast before falling
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Hardware constraints
        self.CONTROL_FREQ = 100.0 
        self.steps_per_ctrl = int((1.0 / self.CONTROL_FREQ) / self.model.opt.timestep)
        self.MAX_TORQUE = 0.100  
        self.MAX_SPEED = 40.0    
        
        # Delay Buffer
        self.k_delay = 2 
        self.torque_buffer = deque([(0.0, 0.0)] * self.k_delay, maxlen=self.k_delay)
        
    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            # --- DOMAIN RANDOMIZATION ---
            
            # 1. Friction & Armature (The "Sticky Motor" fix)
            rand_friction = np.random.uniform(0.005, 0.03)
            rand_armature = np.random.uniform(0.001, 0.004)
            self.model.dof_frictionloss[6] = rand_friction
            self.model.dof_frictionloss[7] = rand_friction
            self.model.dof_armature[6] = rand_armature
            self.model.dof_armature[7] = rand_armature

            # 2. Motor/Battery Strength (+/- 15%)
            self.current_max_torque = self.MAX_TORQUE * np.random.uniform(0.85, 1.15)

            # 3. Mass and Center of Mass
            # Find the ID for the main chassis body
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "balboa_base")
            
            # Randomize Mass (Nominal 0.316 kg +/- 10%)
            self.model.body_mass[body_id] = 0.316 * np.random.uniform(0.9, 1.1)
            
            # Randomize CoM Z-Height (Nominal 0.023m +/- 2mm)
            # self.model.body_ipos is the local CoM position [x, y, z]
            self.model.body_ipos[body_id][2] = 0.023 + np.random.uniform(-0.002, 0.002)
            
            # Randomize CoM X-Offset (Shift forward/backward by +/- 1.5mm)
            # This teaches the network to handle a robot that is naturally "front-heavy" or "back-heavy"
            self.model.body_ipos[body_id][0] = np.random.uniform(-0.0015, 0.0015)

            # Apply the randomized physics
            mujoco.mj_resetData(self.model, self.data)
            
            # Add slight random perturbation to starting tilt
            self.data.qpos[3] = np.random.uniform(-0.05, 0.05) 
            mujoco.mj_forward(self.model, self.data)
            
            self.torque_buffer.clear()
            self.torque_buffer.extend([(0.0, 0.0)] * self.k_delay)
            
            self.current_step = 0
            self.last_action = 0.0 
            
            return self._get_obs(), {}

    def _get_obs(self):
        # Simulate IMU/Encoder Jitter
        theta_noisy = (2.0 * np.arctan2(self.data.qpos[5], self.data.qpos[3])) + np.random.normal(0, 0.005)
        dtheta_noisy = self.data.qvel[4] + np.random.normal(0, 0.02)
        phi_noisy = ((self.data.qpos[7] + self.data.qpos[8]) / 2.0) + np.random.normal(0, 0.002)
        dphi_noisy = ((self.data.qvel[6] + self.data.qvel[7]) / 2.0) + np.random.normal(0, 0.01)
        
        return np.array([phi_noisy, theta_noisy, dphi_noisy, dtheta_noisy], dtype=np.float32)


    def step(self, action):
            # Scale normalized action [-1, 1] to raw torque
            raw_tau = action[0] * self.current_max_torque

            # Apply DC Motor Curve constraints
            w_l, w_r = self.data.qvel[6], self.data.qvel[7]
            avail_l = self.MAX_TORQUE * max(0.0, 1.0 - (abs(w_l) / self.MAX_SPEED))
            avail_r = self.MAX_TORQUE * max(0.0, 1.0 - (abs(w_r) / self.MAX_SPEED))

            tau_l = np.clip(raw_tau, -avail_l, avail_l)
            tau_r = np.clip(raw_tau, -avail_r, avail_r)

            # Push through delay buffer
            self.torque_buffer.append((tau_l, tau_r))
            applied_tau_l, applied_tau_r = self.torque_buffer.popleft()

            # Step the MuJoCo simulation for `steps_per_ctrl` loops to match 100Hz
            for _ in range(self.steps_per_ctrl):
                self.data.ctrl[0] = applied_tau_l
                self.data.ctrl[1] = applied_tau_r
                mujoco.mj_step(self.model, self.data)

            obs = self._get_obs()
            
            # --- SCALED LQR Reward Function ---
            # We divide the original Q and R matrices by 100 to keep rewards small.
            # Original Q = [1, 100, 1, 10]. Scaled Q = [0.01, 1.0, 0.01, 0.1]
            # Original R = 300. Scaled R = 3.0
            
            state_cost = (0.05 * obs[0]**2) + (1.0 * obs[1]**2) + (0.01 * obs[2]**2) + (0.1 * obs[3]**2)
            control_cost = 3.0 * (raw_tau**2) 
                
            # Survival bonus is scaled down to 1.0
            survival_bonus = 1.5
            reward = survival_bonus - (state_cost + control_cost)

            # --- Termination Condition ---
            # If it tilts past ~30 degrees, it has fallen over. End episode.
            terminated = bool(abs(obs[1]) > 0.5)
            if terminated:
                # Massive penalty for failing the LQR objective entirely
                reward -= 20.0 

            truncated = False 
            
            return obs, reward, terminated, truncated, {}



# --- Execution Block ---
if __name__ == "__main__":
    choice = input("Do you want to (t) train, (v) visualize, (vq) visualize quantized, (e) export or (eq) export +quantize  for Arduino? [t/v/e]: ").strip().lower()

    if choice == 't':
        print("Initializing Training Environment...")
        env = BalboaEnv() 
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env) 

        print("Initializing Evaluation Environment...")
        eval_env = BalboaEnv() 
        eval_env = TimeLimit(eval_env, max_episode_steps=1000)
        eval_env = Monitor(eval_env)

        policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=dict(pi=[16, 16], vf=[16, 16]))

        print("Building PPO Model...")
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1) 
        

        print("Setting up Robust Evaluation Callback...")
        # Use our custom callback instead of the default one
        eval_callback = RobustEvalCallback(
                                     eval_env, 
                                     variance_penalty=1.0,      # Adjust this to penalize variance more or less
                                     best_model_save_path='./', 
                                     log_path='./logs/',        
                                     eval_freq=10000,
                                     n_eval_episodes=10,        # Test 10 random physical realities per evaluation
                                     deterministic=True, 
                                     render=False)

        print("Starting Training (Press Ctrl+C to stop early)...")
        
        try:
            model.learn(total_timesteps=500000, callback=eval_callback)
            print("\nTraining completed successfully!")
        except KeyboardInterrupt:
            print("\n\nTraining manually interrupted by user.")
            model.save("latest_model_interrupted")

    elif choice == 'v':
        if not os.path.exists("robust_model.zip"):
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            print("Initializing Environment...")
            env = BalboaEnv()
            
            print("Loading saved model...")
            model = PPO.load("best_model", env=env)

            obs, _ = env.reset()
            print("Launching viewer... (Close the viewer window to exit)")
            
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    viewer.sync()
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                        
                    dt = env.model.opt.timestep * env.steps_per_ctrl
                    elapsed = time.time() - step_start
                    if dt > elapsed:
                        time.sleep(dt - elapsed) 


    elif choice == 'vq':
        if not os.path.exists("robust_model.zip"):
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            import numpy as np
            import time
            print("Initializing Environment...")
            env = BalboaEnv()
            
            print("Loading saved model...")
            model = PPO.load("best_model", env=env)
            
            print("Extracting and Quantizing weights for C++ Simulation...")
            policy = model.policy
            weights = []
            biases = []
            
            # Extract weights and biases
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if "weight" in name: weights.append(param.detach().numpy())
                elif "bias" in name: biases.append(param.detach().numpy())
            for name, param in policy.action_net.named_parameters():
                if "weight" in name: weights.append(param.detach().numpy())
                elif "bias" in name: biases.append(param.detach().numpy())
            
            q_weights = []
            q_biases = []
            w_scales = []
            b_scales = []
            
            # Pre-calculate quantization arrays (mimicking the PROGMEM arrays)
            for w, b in zip(weights, biases):
                w_max = np.max(np.abs(w))
                w_scale = w_max / 127.0 if w_max != 0 else 1.0
                q_weights.append(np.clip(np.round(w / w_scale), -127, 127).astype(np.int8))
                w_scales.append(w_scale)

                b_max = np.max(np.abs(b))
                b_scale = b_max / 127.0 if b_max != 0 else 1.0
                q_biases.append(np.clip(np.round(b / b_scale), -127, 127).astype(np.int8))
                b_scales.append(b_scale)

            # Manual Inference Function mimicking the Arduino C++ implementation
            def run_policy_nn_quantized(obs_in):
                obs_in = obs_in.flatten() # Ensure 1D array
                
                # --- 1. Quantize Inputs ---
                in_max = max(np.max(np.abs(obs_in)), 0.00001)
                in_scale = in_max / 127.0
                q_in = np.clip(np.round(obs_in / in_scale), -127, 127).astype(np.int8)

                # --- 2. Layer 0 -> Layer 1 ---
                # np.dot with int32 natively matches your C++ nested loops
                dot_sum_0 = np.dot(q_weights[0].astype(np.int32), q_in.astype(np.int32))
                val_0 = (dot_sum_0 * in_scale * w_scales[0]) + (q_biases[0] * b_scales[0])
                layer1 = np.maximum(val_0, 0.0) # ReLU Activation
                
                # --- 3. Quantize Layer 1 Outputs ---
                l1_max = max(np.max(np.abs(layer1)), 0.00001)
                l1_scale = l1_max / 127.0
                q_l1 = np.clip(np.round(layer1 / l1_scale), -127, 127).astype(np.int8)
                
                # --- 4. Layer 1 -> Layer 2 ---
                dot_sum_1 = np.dot(q_weights[1].astype(np.int32), q_l1.astype(np.int32))
                val_1 = (dot_sum_1 * l1_scale * w_scales[1]) + (q_biases[1] * b_scales[1])
                layer2 = np.maximum(val_1, 0.0) # ReLU Activation
                
                # --- 5. Quantize Layer 2 Outputs ---
                l2_max = max(np.max(np.abs(layer2)), 0.00001)
                l2_scale = l2_max / 127.0
                q_l2 = np.clip(np.round(layer2 / l2_scale), -127, 127).astype(np.int8)
                
                # --- 6. Layer 2 -> Output Layer ---
                dot_sum_2 = np.dot(q_weights[2].astype(np.int32), q_l2.astype(np.int32))
                raw_action = (dot_sum_2 * l2_scale * w_scales[2]) + (q_biases[2] * b_scales[2])
                
                # Tanh-like clipping (-1.0 to 1.0 limits)
                return np.clip(raw_action, -1.0, 1.0)

            obs, _ = env.reset()
            print("Launching viewer with C++ Quantization simulation... (Close the viewer window to exit)")
            
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    
                    # BYPASS standard SB3 predict and use our custom integer emulator
                    action = run_policy_nn_quantized(obs)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    viewer.sync()
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                        
                    dt = env.model.opt.timestep * env.steps_per_ctrl
                    elapsed = time.time() - step_start
                    if dt > elapsed:
                        time.sleep(dt - elapsed)

    elif choice == 'vq10':
        if not os.path.exists("robust_model.zip"):
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            import numpy as np
            import time
            print("Initializing Environment...")
            env = BalboaEnv()
            
            print("Loading saved model...")
            model = PPO.load("best_model", env=env)
            
            print("Extracting and Quantizing weights to Q10 Fixed-Point...")
            policy = model.policy
            weights = []
            biases = []
            
            # Extract weights and biases
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if "weight" in name: weights.append(param.detach().numpy())
                elif "bias" in name: biases.append(param.detach().numpy())
            for name, param in policy.action_net.named_parameters():
                if "weight" in name: weights.append(param.detach().numpy())
                elif "bias" in name: biases.append(param.detach().numpy())
            
            q_weights = []
            q_biases = []
            SCALE = 1024.0
            
            # Pre-calculate Q10 fixed-point arrays (mimicking the PROGMEM arrays)
            for w, b in zip(weights, biases):
                w_q = np.clip(np.round(w * SCALE), -32768, 32767).astype(np.int16)
                b_q = np.clip(np.round(b * SCALE), -32768, 32767).astype(np.int16)
                q_weights.append(w_q)
                q_biases.append(b_q)

            # Manual Inference Function mimicking the Q10 C++ implementation
            def run_policy_nn_quantized(obs_in):
                obs_in = obs_in.flatten() # Ensure 1D array
                
                # --- 1. Convert Inputs to Q10 Fixed-Point ---
                q_in = (obs_in * SCALE).astype(np.int32)

                # --- 2. Layer 0 -> Layer 1 ---
                # np.dot with int32 natively matches your C++ 32-bit accumulation
                dot_sum_0 = np.dot(q_weights[0].astype(np.int32), q_in)
                # Bit-shift by 10 to divide by 1024, then add bias
                dot_sum_0 = (dot_sum_0 >> 10) + q_biases[0].astype(np.int32)
                layer1 = np.maximum(dot_sum_0, 0) # ReLU Activation
                
                # --- 3. Layer 1 -> Layer 2 ---
                dot_sum_1 = np.dot(q_weights[1].astype(np.int32), layer1)
                dot_sum_1 = (dot_sum_1 >> 10) + q_biases[1].astype(np.int32)
                layer2 = np.maximum(dot_sum_1, 0) # ReLU Activation
                
                # --- 4. Layer 2 -> Output Layer ---
                dot_sum_2 = np.dot(q_weights[2].astype(np.int32), layer2)
                out_dot_sum = (dot_sum_2 >> 10) + q_biases[2].astype(np.int32)
                
                # --- 5. Final Float Conversion ---
                # Divide the final integer by 1024.0 to get the float action
                raw_action = float(out_dot_sum[0]) / SCALE
                
                # Tanh-like clipping (-1.0 to 1.0 limits)
                clipped_action = np.clip(raw_action, -1.0, 1.0)
                
                # WRAP IN ARRAY: The environment expects an array so it can call action[0]
                return np.array([clipped_action], dtype=np.float32)

            obs, _ = env.reset()
            print("Launching viewer with Q10 C++ simulation... (Close the viewer window to exit)")
            
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    
                    # BYPASS standard SB3 predict and use our custom integer emulator
                    action = run_policy_nn_quantized(obs)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    viewer.sync()
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                        
                    dt = env.model.opt.timestep * env.steps_per_ctrl
                    elapsed = time.time() - step_start
                    if dt > elapsed:
                        time.sleep(dt - elapsed)    

    
    elif choice == 'e':
        if not os.path.exists("robust_model.zip"):
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            print("Loading saved model for export...")
            model = PPO.load("best_model")
            policy = model.policy
            
            weights = []
            biases = []
            
            # Extract weights and biases while keeping their 2D/1D shapes
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
                    
            for name, param in policy.action_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
            
            # Target the specific include directory for the STEPfile
            output_dir = "."
            output_path = f"{output_dir}/STEPfile.h"
            
            # Ensure the directory exists before writing
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Writing PROGMEM C++ header directly to {output_path}...")
            with open(output_path, "w") as f:
                f.write("#ifndef STEPFILE_H\n")
                f.write("#define STEPFILE_H\n\n")
                f.write("#include <Arduino.h>\n\n")
                f.write("// Auto-generated PROGMEM Weights (0 bytes SRAM used!)\n\n")
                
                for idx, (w, b) in enumerate(zip(weights, biases)):
                    # Write Weights
                    f.write(f"const float W{idx}[{w.shape[0]}][{w.shape[1]}] PROGMEM = {{\n")
                    for row in w:
                        f.write("    {" + ", ".join([f"{val:.6f}" for val in row]) + "},\n")
                    f.write("};\n\n")
                    
                    # Write Biases
                    f.write(f"const float b{idx}[{b.shape[0]}] PROGMEM = {{\n")
                    f.write("    " + ", ".join([f"{val:.6f}" for val in b]) + "\n")
                    f.write("};\n\n")
                    
                f.write("#endif // STEPFILE_H\n")
                
            print(f"Export complete! Your Arduino sketch is ready to compile with the new policy.")




    elif choice == 'eq':
        if not os.path.exists("robust_model.zip"):
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            import numpy as np
            print("Loading saved model for export...")
            model = PPO.load("best_model")
            policy = model.policy
            
            weights = []
            biases = []
            
            # Extract weights and biases while keeping their 2D/1D shapes
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
                    
            for name, param in policy.action_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
            
            # Target the specific include directory for the STEPfile
            output_dir = "include/cleditor"
            output_path = f"{output_dir}/STEPfile.h"
            
            # Ensure the directory exists before writing
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Quantizing to 8-bit and writing PROGMEM C++ header directly to {output_path}...")
            with open(output_path, "w") as f:
                f.write("#ifndef STEPFILE_H\n")
                f.write("#define STEPFILE_H\n\n")
                f.write("#include <Arduino.h>\n")
                f.write("#include <stdint.h>\n\n")
                f.write("// Auto-generated 8-bit Quantized PROGMEM Weights\n")
                f.write("// Note: Read values using (int8_t)pgm_read_byte(&array[i][j])\n\n")
                
                for idx, (w, b) in enumerate(zip(weights, biases)):
                    # Quantize Weights
                    w_max = np.max(np.abs(w))
                    w_scale = w_max / 127.0 if w_max != 0 else 1.0
                    w_q = np.clip(np.round(w / w_scale), -127, 127).astype(np.int8)

                    # Quantize Biases
                    b_max = np.max(np.abs(b))
                    b_scale = b_max / 127.0 if b_max != 0 else 1.0
                    b_q = np.clip(np.round(b / b_scale), -127, 127).astype(np.int8)

                    # Write Scales (Required for de-quantization during inference)
                    f.write(f"const float W{idx}_scale = {w_scale:.8f};\n")
                    f.write(f"const float b{idx}_scale = {b_scale:.8f};\n\n")

                    # Write Quantized Weights
                    f.write(f"const int8_t W{idx}[{w_q.shape[0]}][{w_q.shape[1]}] PROGMEM = {{\n")
                    for row in w_q:
                        f.write("    {" + ", ".join([str(val) for val in row]) + "},\n")
                    f.write("};\n\n")
                    
                    # Write Quantized Biases
                    f.write(f"const int8_t b{idx}[{b_q.shape[0]}] PROGMEM = {{\n")
                    f.write("    " + ", ".join([str(val) for val in b_q]) + "\n")
                    f.write("};\n\n")
                    
                f.write("#endif // STEPFILE_H\n")
                
            print(f"Export complete! Your Arduino sketch is ready to compile with the new 8-bit policy.") 

    elif choice == 'eq10':
        if not os.path.exists("robust_model.zip"):  # Matched your vq10 filename
            print("Error: No saved model found. Please train one first by selecting 't'.")
        else:
            import numpy as np
            import os
            print("Loading saved model for export...")
            model = PPO.load("robust_model")
            policy = model.policy
            
            weights = []
            biases = []
            
            # Extract weights and biases while keeping their 2D/1D shapes
            for name, param in policy.mlp_extractor.policy_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
                    
            for name, param in policy.action_net.named_parameters():
                if "weight" in name:
                    weights.append(param.detach().numpy())
                elif "bias" in name:
                    biases.append(param.detach().numpy())
            
            # Target the specific include directory for the STEPfile
            output_dir = "include/cleditor"
            output_path = f"{output_dir}/STEPfile.h"
            
            # Ensure the directory exists before writing
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Applying Q10 Fixed-Point scaling and writing PROGMEM C++ header directly to {output_path}...")
            with open(output_path, "w") as f:
                f.write("#ifndef STEPFILE_H\n")
                f.write("#define STEPFILE_H\n\n")
                f.write("#include <Arduino.h>\n")
                f.write("#include <stdint.h>\n\n")
                f.write("// Auto-generated Q10 Fixed-Point Weights (Scale: x1024)\n")
                f.write("// Note: Read values using (int16_t)pgm_read_word(&array[i][j])\n\n")
                
                SCALE = 1024.0
                
                for idx, (w, b) in enumerate(zip(weights, biases)):
                    # Quantize Weights to Q10
                    w_q = np.clip(np.round(w * SCALE), -32768, 32767).astype(np.int16)
                    
                    # Quantize Biases to Q10
                    b_q = np.clip(np.round(b * SCALE), -32768, 32767).astype(np.int16)

                    # Write Quantized Weights (16-bit)
                    f.write(f"const int16_t W{idx}[{w_q.shape[0]}][{w_q.shape[1]}] PROGMEM = {{\n")
                    for row in w_q:
                        f.write("    {" + ", ".join([str(val) for val in row]) + "},\n")
                    f.write("};\n\n")
                    
                    # Write Quantized Biases (16-bit)
                    f.write(f"const int16_t b{idx}[{b_q.shape[0]}] PROGMEM = {{\n")
                    f.write("    " + ", ".join([str(val) for val in b_q]) + "\n")
                    f.write("};\n\n")
                    
                f.write("#endif // STEPFILE_H\n")
                
            print(f"Export complete! Your Arduino sketch is ready to compile with the new Q10 policy.")



    else:
        print("Invalid choice. Please run the script again and enter 't', 'v', or 'e'.")
