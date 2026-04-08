# server/__init__.py
# Re-exports from algo_reasoning_env.server for OpenEnv compatibility.
from algo_reasoning_env.server.app import app, main

__all__ = ["app", "main"]
