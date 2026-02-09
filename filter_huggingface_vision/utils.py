"""Shared utilities for filter_huggingface_vision."""


def get_config_value(obj, key, default=None):
    """Get a value from config whether it is a dict or an object with attributes."""
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)
    return getattr(obj, key, default)
