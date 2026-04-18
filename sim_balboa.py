import time
import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer
import math
import random

# If you add a Raspberry Pi increase `mp` and `l` here

mw = 0.0042  # Mass of the wheels
mp = 0.316  # Mass of the pendulum (Change this when you add the Pi!)
r = 0.040  # Wheel radius
l = 0.023  # Distance to CoM (Change this when you add the Pi!)
Ip = 444.43e-6  # Inertia of pendulum
Iw = 26.89e-6  # Inertia of wheels
g = 9.81

# Construct E, G, F Matrices
E = np.array([[Iw + (r**2) * (mw + mp), mp * r * l], [mp * r * l, Ip + mp * (l**2)]])

G = np.array([[0], [-mp * g * l]])

F = np.array([[1], [0]])

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


Q = np.diag([10.0, 100.0, 1.0, 10.0])
R = np.array([[300.0]])

P = scipy.linalg.solve_continuous_are(A_lab, B_lab, Q, R)
K = np.linalg.inv(R) @ B_lab.T @ P
print("\nCalculated LQR Gains (K):", K)


model = mujoco.MjModel.from_xml_path("balboa.xml")
data = mujoco.MjData(model)


# --- Leonardo Specific Constraints ---
CONTROL_FREQ = 100.0  # Hz (Realistic for Leonardo + I2C IMU)
PWM_RES = 255  # 8-bit PWM resolution
# --- Hardware Reality Settings ---
MAX_TORQUE_L = 0.100
MAX_TORQUE_R = 0.100
MAX_SPEED = 25.0  # rad/s (Back-EMF limit)

# Start small (0.01 - 0.5). If too high, the robot will oscillate/wag its tail.
K_HEADING = 0


# 1440 ticks per revolution (12 CPR encoder * 120 gear ratio)
TICKS_PER_REV = 1440.0
RAD_PER_TICK = (2.0 * np.pi) / TICKS_PER_REV  # Approx 0.00436 radians per tick

# --- Setup Counters ---
step_counter = 0
steps_per_ctrl = int((1.0 / CONTROL_FREQ) / model.opt.timestep)


# Initialize control variables to zero
tau_l, tau_r = 0.0, 0.0


# Global state
last_noise = 0.0
THETA_OU = 0.30
SIGMA_OU = 0.80


def generate_gaussian():
    # Generate two uniform random variables between 0 and 1
    # random.uniform(0.0001, 1.0) ensures we strictly avoid log(0)
    u1 = random.uniform(0.0001, 1.0)
    u2 = random.random()

    # Box-Muller transform calculation
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return z0


def generate_exploration_noise():
    global last_noise

    # 1. Get the true Gaussian noise
    epsilon = generate_gaussian()

    # 2. OU Process formula: dx = theta * (-x) * dt + sigma * dW
    # dt = 0.01, and sqrt(dt) = 0.1
    dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1
    last_noise += dx

    return last_noise


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Run the "Arduino" loop only at its sample rate (100Hz)
        if step_counter % steps_per_ctrl == 0:
            # 1. Read Raw Encoders (and add noise)
            # data.qpos[7/8] are the motor shaft joints in the backlash model
            # 1. Read Encoders with strict digital quantization (No Gaussian noise!)
            phi_l_raw = data.qpos[7]
            phi_r_raw = data.qpos[8]

            # Round the true position to the nearest physical encoder tick
            phi_l = np.round(phi_l_raw / RAD_PER_TICK) * RAD_PER_TICK
            phi_r = np.round(phi_r_raw / RAD_PER_TICK) * RAD_PER_TICK

            # 2. State Vector for LQR
            # Theta (Pitch Angle) comes from a Complementary/Kalman Filter.
            # It is a smoothed floating-point math result, so it doesn't "stair-step"
            # like a digital encoder. A tiny Gaussian fuzz is okay here to simulate floor vibrations.
            theta_raw = 2.0 * np.arctan2(data.qpos[5], data.qpos[3])
            theta_noisy = theta_raw + np.random.normal(0, 0.001)  # Reduced from 0.005

            # 3. Velocities (Keeping Gaussian noise is mathematically correct here)
            dtheta_noisy = data.qvel[4] + np.random.normal(0, 0.02)

            phi_avg = (phi_l + phi_r) / 2.0
            dphi_avg = ((data.qvel[6] + data.qvel[7]) / 2.0) + np.random.normal(0, 0.01)
            x = np.array([phi_avg, theta_noisy, dphi_avg, dtheta_noisy])

            # 3. LQR Calculation
            tau_total = -K @ x + generate_exploration_noise()
            raw_tau = tau_total[0] / 2

            # 4. Heading Correction (Arduino-style)
            phi_diff = phi_l - phi_r
            tau_corr = K_HEADING * phi_diff

            # 5. Combine Control and Apply Motor Curve
            total_l = raw_tau - tau_corr
            total_r = raw_tau + tau_corr

            # Back-EMF constraints using motor shaft velocities
            w_l, w_r = data.qvel[6], data.qvel[7]
            avail_l = MAX_TORQUE_L * max(0.0, 1.0 - (abs(w_l) / MAX_SPEED))
            avail_r = MAX_TORQUE_R * max(0.0, 1.0 - (abs(w_r) / MAX_SPEED))

            # Final clipping
            tau_l = np.clip(total_l, -avail_l, avail_l)
            tau_r = np.clip(total_r, -avail_r, avail_r)

        # 6. Apply Directly to Motors
        # MuJoCo's dyntype="filter" in the XML now handles the lag
        data.ctrl[0] = tau_l
        data.ctrl[1] = tau_r

        mujoco.mj_step(model, data)
        step_counter += 1
        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - step_start
        if model.opt.timestep > elapsed:
            time.sleep(model.opt.timestep - elapsed)
