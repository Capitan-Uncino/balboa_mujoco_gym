import time
import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer

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

print("\nStarting Simulation... (Using Analytical LQR)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 1. Read Quaternions and convert to Pitch (Theta)
        qw = data.qpos[3]
        qy = data.qpos[5]
        
        theta = 2.0 * np.arctan2(qy, qw) 
        dtheta = data.qvel[4]
        
        # 2. Read Wheels (Phi)
        phi = (data.qpos[7] + data.qpos[8]) / 2.0
        dphi = (data.qvel[6] + data.qvel[7]) / 2.0

        # 3. Create State Vector [phi, theta, dphi, dtheta]
        x = np.array([phi, theta, dphi, dtheta])

        tau_total = -K @ x
        
        # Split torque exactly as calculated. NO minus sign here!
        tau_per_motor = tau_total[0] / 2.0
        
        # Saturate to realistic motor limits (0.1 Nm)
        tau_per_motor = np.clip(tau_per_motor, -0.1, 0.1)
        
        data.ctrl[0] = tau_per_motor
        data.ctrl[1] = tau_per_motor 

        if int(data.time)>0 and int(data.time) % 3 == 0 and (data.time % 1.0) < 0.1:
            data.xfrc_applied[1, 4] = 0.05 
        else:
            data.xfrc_applied[1, 4] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - step_start
        if model.opt.timestep > elapsed:
            time.sleep(model.opt.timestep - elapsed)






