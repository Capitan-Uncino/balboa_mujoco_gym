import time
import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer
from collections import deque
# If you add a Raspberry Pi increase `mp` and `l` here

mw = 0.0042      # Mass of the wheels
mp = 0.316       # Mass of the pendulum (Change this when you add the Pi!)
r = 0.040        # Wheel radius
l = 0.023        # Distance to CoM (Change this when you add the Pi!)
Ip = 444.43e-6   # Inertia of pendulum
Iw = 26.89e-6    # Inertia of wheels
g = 9.81

# Construct E, G, F Matrices
E = np.array([
    [Iw + (r**2)*(mw + mp), mp * r * l],
    [mp * r * l,            Ip + mp * (l**2)]
])

G = np.array([
    [0],
    [-mp * g * l]
])

F = np.array([
    [1],
    [0]
])

# Calculate A and B
E_inv = np.linalg.inv(E)
E_inv_G = E_inv @ G
E_inv_F = E_inv @ F

A_lab = np.zeros((4, 4))
A_lab[0, 2] = 1.0
A_lab[1, 3] = 1.0
A_lab[2, 1] = -E_inv_G[0, 0]
A_lab[3, 1] = -E_inv_G[1, 0]

B_lab = np.zeros((4, 1))
B_lab[2, 0] = E_inv_F[0, 0]
B_lab[3, 0] = E_inv_F[1, 0]

print("--- ANALYTICAL MATRICES  ---")
print("A Matrix:\n", np.round(A_lab, 2))
print("B Matrix:\n", np.round(B_lab, 2))


Q = np.diag([1.0, 100.0, 1.0, 10.0]) 
R = np.array([[300.0]])

P = scipy.linalg.solve_continuous_are(A_lab, B_lab, Q, R)
K = np.linalg.inv(R) @ B_lab.T @ P
print("\nCalculated LQR Gains (K):", K)


model = mujoco.MjModel.from_xml_path("balboa.xml")
data = mujoco.MjData(model)

# --- Leonardo Specific Constraints ---
CONTROL_FREQ = 100.0  # Hz (Realistic for Leonardo + I2C IMU)
PWM_RES = 255         # 8-bit PWM resolution
# --- Hardware Reality Settings ---
MAX_TORQUE_L = 0.100  
MAX_TORQUE_R = 0.100  
MAX_SPEED = 25.0      # rad/s (Back-EMF limit)

# --- Setup Buffer and Counters ---
k_delay = 6  # Physical/Comm delay
torque_buffer = deque([(0.0, 0.0)] * k_delay, maxlen=k_delay)
step_counter = 0
steps_per_ctrl = int((1.0 / CONTROL_FREQ) / model.opt.timestep)
applied_tau = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Run the "Arduino" loop only at its sample rate
        if step_counter % steps_per_ctrl == 0:
            
            # 1. Simulate IMU/Encoder Jitter
            # Leonardo's 10-bit ADC or I2C noise is significant
            theta_noisy = (2.0 * np.arctan2(data.qpos[5], data.qpos[3])) + np.random.normal(0, 0.005)
            dtheta_noisy = data.qvel[4] + np.random.normal(0, 0.02)
            phi_noisy = ((data.qpos[7] + data.qpos[8]) / 2.0) + np.random.normal(0, 0.002)
            dphi_noisy = ((data.qvel[6] + data.qvel[7]) / 2.0) + np.random.normal(0, 0.01)

            # 2. State Vector
            x = np.array([phi_noisy, theta_noisy, dphi_noisy, dtheta_noisy])

            # 3. LQR Calculation
            tau_total = -K @ x
            raw_tau = tau_total[0]/2

                    # 3. Apply DC Motor Curve (Torque drops as speed increases)
            # This prevents the "infinite save" at high speeds
            w_l, w_r = data.qvel[6], data.qvel[7]
            avail_l = MAX_TORQUE_L * max(0.0, 1.0 - (abs(w_l) / MAX_SPEED))
            avail_r = MAX_TORQUE_R * max(0.0, 1.0 - (abs(w_r) / MAX_SPEED))

            # 4. Clip to DIFFERENT limits
            tau_l = np.clip(raw_tau, -avail_l, avail_l)
            tau_r = np.clip(raw_tau, -avail_r, avail_r)

            # 5. Push through the Leonardo Delay Buffer
            torque_buffer.append((tau_l, tau_r))
            applied_tau_l, applied_tau_r = torque_buffer.popleft()

        # Apply to motors
        data.ctrl[0] = applied_tau_l
        data.ctrl[1] = applied_tau_r
        mujoco.mj_step(model, data)
        step_counter += 1
        viewer.sync()

        # Pacing
        elapsed = time.time() - step_start
        if model.opt.timestep > elapsed:
            time.sleep(model.opt.timestep - elapsed)





