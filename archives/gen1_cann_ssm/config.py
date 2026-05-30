# DEPRECATED ¡ª legacy TemporalSNN architecture, use rina.mohe.MoHE instead
import os
import json


_DEFAULT_CONFIG = {
    "dm": 840,
    "np": 4096,
    "seq": 64,
    "bs": 8,
    "ae": 2,
    "epochs": 13,
    "lr": 3e-4,
    "error_threshold": 1.0,
    "hebbian_lr": 0.01,
    "inhibition_threshold": 0.8,
    "weight_decay": 0.01,
    "beta2": 0.999,
    "subsample": 8,
    "n_segments": 200000,
}


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config", "default.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        _DEFAULT_CONFIG.update(cfg)
    return _DEFAULT_CONFIG
