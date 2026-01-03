from backend.api.diagnostics_routes import router as diagnostics_router
from backend.api.lms_routes import router as lms_router
from backend.api.perceptron_routes import router as perceptron_router

__all__ = ["diagnostics_router", "lms_router", "perceptron_router"]
