"""Lightweight model registry."""
import os
import json
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
REGISTRY_PATH = os.path.join(ROOT, "models", "registry.json")


def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def list_versions():
    reg = load_registry()
    return reg


def get_model_path(version=None):
    reg = load_registry()
    if version is None:
        version = reg.get("latest")
    if version not in reg["versions"]:
        raise KeyError(f"Unknown model version: {version}")
    fname = reg["versions"][version]["file"]
    return os.path.join(ROOT, "models", fname), version


def load_model(version=None):
    path, version = get_model_path(version)
    bundle = joblib.load(path)
    return bundle, version
