import numpy as np

#--------------
# Parameters
#--------------

params = {
    "g": 9.81,   # gravity (m/s^2)
    "l": 1.0,    # lenght (m)
    "m": 1.0     # mass (kg)
}

T = 5.0  # = 5s
dt = 0.01  # 500 step
n_traj = 200  #  100 000 points
u_max = 0.5   

#------------
# Storage
#------------
X = []
U = []
X_next = []
traj_IDs = []

#-------------------
# Data generation
#-------------------

for k in range(n_traj):
  theta0 = np.random.uniform(-np.pi, np.pi)  #random initialisation of the angle
  omega0 = np.random.uniform(-1.0, 1.0)      #random initialisation of the angular velocity
  u_traj = np.random.uniform(-u_max, u_max, size=(int(T/dt),1))
  x0 = np.array([theta0, omega0])

  traj = trajectory_simulation(x0, u_traj, dt, T, params)

  X.append(traj[:-1]) # We don't include the last one because there is no future state to predict
  X_next.append(traj[1:])  # We don't include the first one because there is no previous state to predict the first one
  U.append(u_traj)
  traj_IDs.append( k * np.ones(int(T/dt), dtype=int) )

X = np.concatenate(X, axis=0)
X_next = np.concatenate(X_next, axis=0)
U = np.concatenate(U, axis=0)
traj_IDs = np.concatenate(traj_IDs)

#------------
#   Save
#------------

np.savez(
         "pendulum_dataset.npz",  
         X=X,   # shape (N * (T / dt),2)
         U=U,   # shape (N * (T / dt),1)
         X_next=X_next,   # shape (N * (T / dt), 2)
         traj_ID=traj_IDs,   # shape (N * (T / dt), 1)
         dt=dt 
         )