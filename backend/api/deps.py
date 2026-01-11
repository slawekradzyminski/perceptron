"""Dependency injection for FastAPI routes.

This module provides service instances using FastAPI's Depends() pattern.
Each service is instantiated once (singleton) using functools.lru_cache
and can be easily overridden in tests via app.dependency_overrides.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from backend.services.alexnet_service import AlexNetService
from backend.services.backprop_service import BackpropService
from backend.services.conv_service import ConvService
from backend.services.deep_service import DeepService
from backend.services.diagnostics_service import DiagnosticsService
from backend.services.gd_service import GdService
from backend.services.lms_service import LmsService
from backend.services.mlp_service import MlpService
from backend.services.perceptron_service import PerceptronService
from backend.services.scale_service import ScaleService
from backend.services.transformer_service import TransformerService


@lru_cache
def get_perceptron_service() -> PerceptronService:
    """Get or create the PerceptronService singleton."""
    return PerceptronService()


@lru_cache
def get_lms_service() -> LmsService:
    """Get or create the LmsService singleton."""
    return LmsService()


@lru_cache
def get_mlp_service() -> MlpService:
    """Get or create the MlpService singleton."""
    return MlpService()


@lru_cache
def get_gd_service() -> GdService:
    """Get or create the GdService singleton."""
    return GdService()


@lru_cache
def get_backprop_service() -> BackpropService:
    """Get or create the BackpropService singleton."""
    return BackpropService()


@lru_cache
def get_deep_service() -> DeepService:
    """Get or create the DeepService singleton."""
    return DeepService()


@lru_cache
def get_alexnet_service() -> AlexNetService:
    """Get or create the AlexNetService singleton."""
    return AlexNetService()


@lru_cache
def get_transformer_service() -> TransformerService:
    """Get or create the TransformerService singleton."""
    return TransformerService()


@lru_cache
def get_scale_service() -> ScaleService:
    """Get or create the ScaleService singleton."""
    return ScaleService()


@lru_cache
def get_conv_service() -> ConvService:
    """Get or create the ConvService singleton."""
    return ConvService()


@lru_cache
def get_diagnostics_service() -> DiagnosticsService:
    """Get or create the DiagnosticsService singleton."""
    return DiagnosticsService()


# Type aliases for use in route function signatures
PerceptronServiceDep = Annotated[PerceptronService, Depends(get_perceptron_service)]
LmsServiceDep = Annotated[LmsService, Depends(get_lms_service)]
MlpServiceDep = Annotated[MlpService, Depends(get_mlp_service)]
GdServiceDep = Annotated[GdService, Depends(get_gd_service)]
BackpropServiceDep = Annotated[BackpropService, Depends(get_backprop_service)]
DeepServiceDep = Annotated[DeepService, Depends(get_deep_service)]
AlexNetServiceDep = Annotated[AlexNetService, Depends(get_alexnet_service)]
TransformerServiceDep = Annotated[TransformerService, Depends(get_transformer_service)]
ScaleServiceDep = Annotated[ScaleService, Depends(get_scale_service)]
ConvServiceDep = Annotated[ConvService, Depends(get_conv_service)]
DiagnosticsServiceDep = Annotated[DiagnosticsService, Depends(get_diagnostics_service)]
