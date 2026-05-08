"""Workload registry for motivation experiments."""

from importlib import import_module


def load_workload(name: str):
    """Load a workload adapter by name."""
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid workload name: {name!r}")
    module_name = name.replace("-", "_")
    module = import_module(f"workloads.{module_name}.workload")
    return module.Workload()
