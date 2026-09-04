"""Flask application factory."""
from __future__ import annotations

import logging
import os
import threading

from flask import Flask, jsonify
from flask_cors import CORS

from app import extensions
from app.auth.users import UserStore
from app.config import get_config
from app.ml.pipeline import MLPipeline
from app.routes import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def create_app() -> Flask:
    app = Flask(__name__)
    config = get_config()
    app.config.from_object(config)

    if not app.config.get("SECRET_KEY"):
        # A missing secret key would silently make auth insecure/broken —
        # fail loudly instead, at startup, not on the first login attempt.
        raise RuntimeError(
            "SECRET_KEY is not set. Generate one (e.g. `python -c "
            "'import secrets; print(secrets.token_hex(32))'`) and set it in "
            "the backend environment — see .env.example."
        )

    CORS(app, origins=app.config["CORS_ORIGINS"], supports_credentials=False)

    extensions.user_store = UserStore()
    extensions.ml_pipeline = MLPipeline(
        database_url=app.config["DATABASE_URL"],
        ltv_horizon_months=app.config["RETENTION_LTV_HORIZON_MONTHS"],
        offer_cost_tnd=app.config["RETENTION_OFFER_COST_TND"],
        artifact_path=app.config["ML_ARTIFACT_PATH"],
    )

    register_routes(app)
    _register_error_handlers(app)

    if not app.config.get("TESTING") and os.environ.get("ML_PIPELINE_AUTOSTART", "1") != "0":
        if not extensions.ml_pipeline.load_artifact():
            extensions.ml_pipeline.start_background_refresh()
        _maybe_schedule_periodic_refresh(app)

    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_err):
        return jsonify(error="not_found", message="The requested resource does not exist."), 404

    @app.errorhandler(405)
    def method_not_allowed(_err):
        return jsonify(error="method_not_allowed", message="This HTTP method is not supported here."), 405

    @app.errorhandler(500)
    def server_error(err):
        app.logger.exception("Unhandled server error: %s", err)
        return jsonify(error="internal_server_error", message="Something went wrong on our side."), 500


def _maybe_schedule_periodic_refresh(app: Flask) -> None:
    interval_minutes = app.config.get("ML_REFRESH_INTERVAL_MINUTES", 0)
    if not interval_minutes:
        return

    def _loop():
        import time

        while True:
            time.sleep(interval_minutes * 60)
            app.logger.info("Scheduled ML pipeline refresh firing (every %d min).", interval_minutes)
            extensions.ml_pipeline.refresh()

    threading.Thread(target=_loop, name="ml-pipeline-scheduler", daemon=True).start()
