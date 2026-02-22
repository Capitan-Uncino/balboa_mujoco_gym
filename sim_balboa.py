import time
import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer

# 1. Load the model (The XML contains the "Real" physics)
try:
    model = mujoco.MjModel.from_xml_path("balboa0.xml")
except ValueError as e:
    print(f"Error loading model: {e}")
    exit()

data = mujoco.MjData(model)



# 2. Linearize using MuJoCo (Robust to future hardware changes!)
mujoco.mj_resetData(model, data)
data.qpos[5] = 0.001
mujoco.mj_forward(model, data)

nv = model.nv
nu = model.nu
dt = model.opt.timestep

# Get the full discrete matrices
A_full = np.zeros((2 * nv, 2 * nv))
B_full = np.zeros((2 * nv, nu))
mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)

# 3. Convert to Continuous Time (To match the Lab's ẋ = Ax + Bu)
# A_continuous = (A_discrete - Identity) / dt
# B_continuous = B_discrete / dt
A_c_full = (A_full - np.eye(2 * nv)) / dt
B_c_full = B_full / dt

# 4. "Squash" the Matrix to the 4-State Lab Model
# Lab State: x = [phi, theta, d_phi, d_theta]
# MuJoCo Indices:
#   Theta (Pitch) = Index 4
#   Wheel Left    = Index 6
#   Wheel Right   = Index 7

print("Reducing 3D model to 2D 'Lab' format...")

# Create the empty 4x4 A matrix and 4x1 B matrix
A_lab = np.zeros((4, 4))
B_lab = np.zeros((4, 1))

# --- Fill Kinematic Rows (Top Half) ---
# d(phi)/dt = d_phi
A_lab[0, 2] = 1.0 
# d(theta)/dt = d_theta
A_lab[1, 3] = 1.0 

# --- Fill Dynamic Rows (Bottom Half) ---
# We take the relevant accelerations from the bottom-right of A_c_full.
# Since the robot is symmetric, we average the Left/Right wheel effects.

# Indices for velocities in the full matrix
idx_v_theta = nv + 4
idx_v_wheelL = nv + 6
idx_v_wheelR = nv + 7

# Indices for positions in the full matrix
idx_p_theta = 4
idx_p_wheelL = 6
idx_p_wheelR = 7

# -- Equation for d(d_phi)/dt -- (Row 2 in Lab Matrix)
# Effect of theta on wheel accel (Average of L/R)
A_lab[2, 1] = (A_c_full[idx_v_wheelL, idx_p_theta] + A_c_full[idx_v_wheelR, idx_p_theta]) / 2.0
# Effect of d_theta on wheel accel
A_lab[2, 3] = (A_c_full[idx_v_wheelL, idx_v_theta] + A_c_full[idx_v_wheelR, idx_v_theta]) / 2.0
# Effect of d_phi on wheel accel (Average self-interaction)
A_lab[2, 2] = (A_c_full[idx_v_wheelL, idx_v_wheelL] + A_c_full[idx_v_wheelR, idx_v_wheelR]) / 2.0

# -- Equation for d(d_theta)/dt -- (Row 3 in Lab Matrix)
# Effect of theta on pitch accel
A_lab[3, 1] = A_c_full[idx_v_theta, idx_p_theta]
# Effect of d_theta on pitch accel
A_lab[3, 3] = A_c_full[idx_v_theta, idx_v_theta]
# Effect of d_phi on pitch accel (Sum of L/R effects)
A_lab[3, 2] = A_c_full[idx_v_theta, idx_v_wheelL] + A_c_full[idx_v_theta, idx_v_wheelR]

# -- Input Matrix B --
# The Lab assumes 'tau_0' drives both wheels.
# So we SUM the effect of Left and Right motors.
B_lab[2, 0] = B_c_full[idx_v_wheelL, 0] + B_c_full[idx_v_wheelL, 1] # Effect on wheel accel
B_lab[3, 0] = B_c_full[idx_v_theta, 0] + B_c_full[idx_v_theta, 1]   # Effect on pitch accel

print("\n--- EXTRACTED LAB MATRICES (Auto-Calculated) ---")
print("A_lab (4x4):\n", np.round(A_lab, 2))
print("B_lab (4x1):\n", np.round(B_lab, 2))

# 5. Design Controller on the 4-State Model
# You can now use Pole Placement (Question 4) or LQR here.
# Let's use LQR for stability:
Q_lab = np.diag([1.0, 1.0, 1.0, 1.0]) # Penalize [phi, theta, dphi, dtheta]
R_lab = np.array([[1.0]])

P = scipy.linalg.solve_continuous_are(A_lab, B_lab, Q_lab, R_lab)
K_lab = np.linalg.inv(R_lab) @ B_lab.T @ P

print("\nCalculated Lab Gains K:", K_lab)

# 6. Run Simulation
print("\nStarting Simulation... (Using 4-State Controller)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # --- EXTRACT STATE ---
        qw = data.qpos[3]
        qy = data.qpos[5]
        
        # Calculate pitch. We multiply by -1 if the robot drives the wrong way!
        # (Depending on your quaternion layout, MuJoCo's forward pitch is often negative)
        theta = -(2.0 * np.arctan2(qy, qw)) 
        dtheta = -data.qvel[4] 
        
        phi = (data.qpos[7] + data.qpos[8]) / 2.0 
        dphi = (data.qvel[6] + data.qvel[7]) / 2.0 

        x_lab = np.array([phi, theta, dphi, dtheta])

        # --- APPLY CONTROL ---
        tau_total = -K_lab @ x_lab 
        tau_per_motor = tau_total[0] / 2.0
        
        # Real-world gearmotor limit (already handled by XML, but safe to keep here)
        tau_per_motor = np.clip(tau_per_motor, -0.03, 0.03)
        
        data.ctrl[0] = tau_per_motor
        data.ctrl[1] = tau_per_motor

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
