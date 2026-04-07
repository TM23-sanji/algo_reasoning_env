"""
FastAPI application for the Algo Reasoning Environment.

This module creates an HTTP server that exposes the AlgoReasoningEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment and get a new problem
    - POST /step: Evaluate an agent's submission
    - GET /state: Get current environment state
    - GET /health: Health check

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app as create_openenv_app
    from openenv.core.env_server.types import ServerMode
except ImportError as e:
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with 'pip install openenv>=0.1.0'."
    ) from e

from algo_reasoning_env import (
    AlgoReasoningAction,
    AlgoReasoningObservation,
    AlgoReasoningEnvironment,
)


# Create the FastAPI app
app = create_openenv_app(
    env=AlgoReasoningEnvironment,
    action_cls=AlgoReasoningAction,
    observation_cls=AlgoReasoningObservation,
    env_name="algo_reasoning_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    Args:
        host: Host address to bind to
        port: Port number to listen on
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    main(host=args.host, port=args.port)
