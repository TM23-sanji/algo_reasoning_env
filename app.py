"""
HuggingFace Spaces entry point for the Algo Reasoning Environment.

This file serves as the entry point for HF Spaces Docker deployment.
It imports and runs the FastAPI app from algo_reasoning_env.server.app.
"""

from algo_reasoning_env.server.app import app, main

if __name__ == "__main__":
    main()
