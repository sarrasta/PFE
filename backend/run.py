"""Local dev entrypoint (`python run.py`). In Docker, gunicorn imports
`run:app` directly instead — see Dockerfile."""
from app import create_app

app = create_app()

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=app.config["DEBUG"])
