"""Small shared response helpers so every route returns the same shapes."""
from __future__ import annotations

from flask import jsonify

from app import extensions


def pipeline_not_ready_response():
    pipeline = extensions.ml_pipeline
    status = pipeline.status if pipeline else "idle"
    if status == "error":
        return jsonify(
            error="ml_pipeline_error",
            message="The ML pipeline failed to initialize. Check backend logs / /api/health.",
            detail=pipeline.error_message if pipeline else None,
        ), 503
    return jsonify(
        error="ml_pipeline_warming_up",
        message="The ML pipeline is training against the data warehouse — try again shortly.",
        status=status,
    ), 503


def require_ready_pipeline(view):
    """Decorator: short-circuits with 503 only when NO scored data exists yet
    (the very first cold start, or a startup run that failed outright).

    Deliberately does NOT require `status == "ready"` — a background or
    admin-triggered refresh sets status back to "loading" while it retrains,
    but `pipeline.client_table` still holds the last good, complete result
    (it's only swapped atomically once the new run finishes — see
    `MLPipeline.refresh()`). Gating on status alone would take the whole API
    down for ~5 minutes on every refresh instead of serving the still-valid
    cached data while the new one trains underneath it.
    """
    import functools

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        pipeline = extensions.ml_pipeline
        if pipeline is None or pipeline.client_table is None:
            return pipeline_not_ready_response()
        return view(*args, **kwargs)

    return wrapped
