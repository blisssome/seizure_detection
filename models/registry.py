from models.transformer_model import Transformer_EEG
from models.cnn1d_model import CNN1D_EEG
from models.cnn2d_model import CNN2D_EEG

MODEL_REGISTRY = {
    "transformer": {
        "class": Transformer_EEG,
        "params": lambda n_channels: {
            "input_channels": n_channels,
            "n_classes": 2,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2
        }
    },
    "cnn1d": {
        "class": CNN1D_EEG,
        "params": lambda n_channels: {
            "input_channels": n_channels,  # override to fixed 4
            "n_classes": 2
        }
    },
    "cnn2d": {
        "class": CNN2D_EEG,
        "params": lambda n_channels: {
            "n_classes": 2
        }
    }
}

def get_model(name, n_channels):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[name]["class"]
    param_func = MODEL_REGISTRY[name]["params"]
    return model_class(**param_func(n_channels))