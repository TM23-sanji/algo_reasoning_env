# server/app.py — re-export so OpenEnv validator can find app and main()
from algo_reasoning_env.server.app import app as _app
from algo_reasoning_env.server.app import main as _main

# Re-export so validator can see them
app = _app


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point — delegates to algo_reasoning_env.server.app.main."""
    _main(host=host, port=port)


if __name__ == "__main__":
    main()
