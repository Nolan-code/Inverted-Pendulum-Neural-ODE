import argparse

from src.needed_functions import hnn_fn
from src.needed_functions import lnn_fn
from src.needed_functions import vector_field_fn

MODEL_REGISTRY = {
    "mlp": VectorFieldMLP,
    "hnn": HNN,
    "lnn": LNN,
}

SIMULATE_REGISTRY = {
    "hnn": simulate_hnn,
    "lnn": simulate_lnn,
    "mlp": simulate_mlp,
}

def build_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]()
    except KeyError:
        raise ValueError(f"Unknown model type: {model_name}")

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help='Model type: mlp, hnn, or lnn')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g., hnn_model.pth)')
parser.add_argument('--initial-conditions', type=float, nargs='+', required=True, help='Initial conditions [theta, omega, ...]')
parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration')
parser.add_argument('--dt', type=float, default=0.01, help='Time step')
parser.add_argument('--output', type=str, default='simulation.npy',help='Output file for trajectory')

args = parser.parse_args()
