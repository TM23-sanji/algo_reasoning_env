# server/app.py — re-export so OpenEnv validator can find app and main()
from algo_reasoning_env.server.app import app as _app
from algo_reasoning_env.server.app import main as _main

# Re-export so validator can see them
app = _app
main = _main

if __name__ == "__main__":
    main()
