"""Process-wide singletons, initialized once by `create_app()`."""
from app.auth.users import UserStore
from app.ml.pipeline import MLPipeline

user_store: UserStore | None = None
ml_pipeline: MLPipeline | None = None
