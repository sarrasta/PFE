"""Blueprint registration — one module per nav section (see README)."""
from flask import Flask


def register_routes(app: Flask) -> None:
    from app.routes import (
        admin_routes,
        auth_routes,
        churn_routes,
        clients_routes,
        dashboard_routes,
        health_routes,
        models_routes,
        retention_routes,
        revenue_routes,
        segmentation_routes,
    )

    app.register_blueprint(health_routes.bp)
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(dashboard_routes.bp)
    app.register_blueprint(clients_routes.bp)
    app.register_blueprint(churn_routes.bp)
    app.register_blueprint(segmentation_routes.bp)
    app.register_blueprint(retention_routes.bp)
    app.register_blueprint(revenue_routes.bp)
    app.register_blueprint(models_routes.bp)
    app.register_blueprint(admin_routes.bp)
