from backend.api.alexnet_routes import router as alexnet_router
from backend.api.backprop_routes import router as backprop_router
from backend.api.deep_routes import router as deep_router
from backend.api.diagnostics_routes import router as diagnostics_router
from backend.api.gd_routes import router as gd_router
from backend.api.lms_routes import router as lms_router
from backend.api.mlp_routes import router as mlp_router
from backend.api.perceptron_routes import router as perceptron_router
from backend.api.transformer_routes import router as transformer_router

__all__ = [
    "alexnet_router",
    "backprop_router",
    "deep_router",
    "diagnostics_router",
    "gd_router",
    "lms_router",
    "mlp_router",
    "perceptron_router",
    "transformer_router",
]
