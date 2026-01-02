import numpy as np
import torch

def pendulum_dynamics(x, u, params):  # Compute the function f(x,t) such that theta'' = f(x,t)
  theta, omega = x
  m, l, g = params["m"], params["l"], params["g"]
  u = float(u)

  d_theta = omega
  d_omega = (g/l) * np.sin(theta) + u/(m * l**2)

  return np.array([d_theta, d_omega])

def rk4_step(f, x, u, dt, params):  # Compute the position after a short time dt based on the pervious position using rk4 method
  k1 = f(x, u, params)
  k2 = f(x + 0.5 * dt * k1, u, params)
  k3 = f(x + 0.5 * dt * k2, u, params)
  k4 = f(x + dt * k3, u, params)

  return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation(x0, u_control, dt, T, params):  # Compute the trajectory using rk4_step at each step
  N = int(T/dt)
  x = np.zeros((N+1, 2))
  x[0] = x0

  for t in range(N):
    u = u_control(t * dt, x[t])
    x[t+1] = rk4_step(pendulum_dynamics, x[t], u, dt, params)
  return x

def zero_control(t,x):
  return 0


# HNN
def model_input_hnn(x):
  sin = torch.sin(x[:,0].unsqueeze(1))
  cos = torch.cos(x[:,0].unsqueeze(1))
  omega = x[:,1].unsqueeze(1)
  x = torch.cat([sin, cos, omega], dim=1)
  return x
  
def rk4_step_HNN(model, x, dt):   # rk4_step but adapted for the HNN
    k1 = model( model_input_hnn(x) )
    k2 = model( model_input_hnn(x + 0.5 * dt * k1) )
    k3 = model( model_input_hnn(x + 0.5 * dt * k2) )
    k4 = model( model_input_hnn(x + dt * k3) )

    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_simulation_HNN(x0, dt, T):  # adapted version for the HNN
    N = int(T / dt)
    x = torch.zeros((N+1, 2))
    x[0] = torch.tensor(x0, dtype=torch.float32)

    for t in range(N):
        x_current = x[t].unsqueeze(0)
        x[t+1] = rk4_step_HNN(model, x_current, dt).detach()

    return x.detach().numpy()
