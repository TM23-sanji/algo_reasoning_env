"""
In-memory session store for the Algo Reasoning Environment HTTP server.

Persists environment instances across HTTP requests within the same process.
Each session maps a UUID to an AlgoReasoningEnvironment instance.

Important:
    - Uses a single worker (--workers 1) to ensure sessions are shared
    - Sessions are auto-deleted after step() returns done=True
    - On HF Spaces, this works because containers run with a single process
"""

import os
from typing import Dict, Optional, Tuple
from uuid import uuid4

from algo_reasoning_env import AlgoReasoningEnvironment


_sessions: Dict[str, AlgoReasoningEnvironment] = {}


def create_session(
    data_dir: str = "/data",
    api_key: Optional[str] = None,
) -> Tuple[str, AlgoReasoningEnvironment]:
    """
    Create a new session with a fresh environment instance.

    Args:
        data_dir: Path to dataset files
        api_key: API key for LLM judge

    Returns:
        Tuple of (session_id, environment)
    """
    session_id = str(uuid4())
    env = AlgoReasoningEnvironment(
        data_dir=data_dir,
        api_key=api_key or os.getenv("LIGHTNING_API_KEY"),
    )
    _sessions[session_id] = env
    return session_id, env


def get_session(session_id: str) -> AlgoReasoningEnvironment:
    """
    Retrieve an existing session by ID.

    Args:
        session_id: The session identifier returned by create_session

    Returns:
        The environment instance for this session

    Raises:
        KeyError: If session not found or expired
    """
    env = _sessions.get(session_id)
    if env is None:
        raise KeyError(f"Session {session_id} not found or expired. Call /reset first.")
    return env


def delete_session(session_id: str) -> None:
    """
    Delete a session and clean up its environment.

    Args:
        session_id: The session identifier to delete
    """
    _sessions.pop(session_id, None)


def session_count() -> int:
    """Return the number of active sessions."""
    return len(_sessions)
