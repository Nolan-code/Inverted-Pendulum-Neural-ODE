import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

from src.physics.model_simulation import *
from src.physics.true_simulation import *


def build_model(model_name, input_dim):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_name}")

    model_class = MODEL_REGISTRY[model_name]
    try:
        model = model_class(input_dim=input_dim)
        print(f"Model {model_name} created with input_dim={input_dim}")
        return model
    except TypeError as e:
        print(f"Warning: {model_name} doesn't accept input_dim parameter, using default constructor")
        print(f"Error details: {e}")
        return model_class()

def get_input_dim(format):
    if format == "theta":
        return 2
    if format == "sincos":
        return 3
    else:
        raise ValueError(f"Format inconnu: {format_type}")

MODEL_REGISTRY = {
    "mlp": MLP,
    "hnn": HNN,
    "lnn": LNN,
}

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help='Model type: mlp, hnn, or lnn')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g., hnn_model.pth)')
parser.add_argument('--format', type=str, choices=['theta', 'sincos'], default='theta', help='Input format: theta (2D) or sincos (3D)')
parser.add_argument('--initial_conditions', type=float, nargs='+', required=True, help='Initial conditions [theta, omega, ...]')
parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration')
parser.add_argument('--step', type=float, default=0.01, help='Step')
parser.add_argument('--output', type=str, default='simulation.npy',help='Output file for trajectory')
parser.add_argument('--show', action='store_true', help='Show plots')

args = parser.parse_args()

#--------------
# Parameters
#--------------

params = {
    "g": 9.81,   # gravity (m/s^2)
    "l": 1.0,    # lenght (m)
    "m": 1.0     # mass (kg)
}

input_dim = get_input_dim(args.format)

model = build_model(args.model, input_dim)
ckpt = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

T = args.duration
dt = args.step
x0 = args.initial_conditions

t = np.arange(start=0, stop = T + dt, step = dt)

#--------------
# Trajectories
#--------------

model_trajectory = simulate(model, x0, dt, T, format=args.format)

true_trajectory = trajectory_simulation(x0, zero_control, dt, T, params=params)

#-------
# Plot
#-------

output_dir = Path(f"./src/simulation_figures/{args.model}")
output_dir.mkdir(parents=True, exist_ok=True)  

plt.figure(figsize=(10, 6))
plt.plot(t, np.array(model_trajectory)[:,0], "--")
plt.plot(t, np.array(true_trajectory)[:,0], "-.")
plt.title("Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.legend([f'{args.model}', "True"])
traj_path = output_dir / "trajectory.png"
plt.savefig(traj_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {traj_path}")
if args.show:
    plt.show()
else:
    plt.close()

plt.figure(figsize=(10, 6))
plt.plot(np.array(model_trajectory)[:,0], np.array(model_trajectory)[:,1], "--")
plt.plot(np.array(true_trajectory)[:,0], np.array(true_trajectory)[:,1], "-.")
plt.title("Phase Space")
plt.legend([f'{args.model}', "True"])
plt.xlabel("Theta (rad)")
plt.ylabel("Omega (rad/s)")
phase_path = output_dir / "phase_portrait.png"
plt.savefig(phase_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {phase_path}")
if args.show:
    plt.show()
else:
    plt.close()

output_dir = Path(f"./results/{args.model}")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    output_dir / "trajectory.npz",
    t=t,
    x_pred=np.array(model_trajectory),
    x_true=np.array(true_trajectory),
    model=args.model,
    checkpoint=args.checkpoint,
    dt=dt,
    x0=x0
)

