"""Scale comparison service for Chapter 5."""

from __future__ import annotations

from typing import Any

# Historical model data for comparison
MODELS = [
    {
        "name": "LeNet-5",
        "year": 1998,
        "params": 60_000,
        "type": "CNN",
        "description": "First successful CNN for digit recognition",
        "input_size": "32×32",
        "notable": "Introduced convolutional layers",
    },
    {
        "name": "AlexNet",
        "year": 2012,
        "params": 61_000_000,
        "type": "CNN",
        "description": "ImageNet breakthrough, sparked deep learning revolution",
        "input_size": "224×224",
        "notable": "ReLU activation, dropout, GPU training",
    },
    {
        "name": "VGG-16",
        "year": 2014,
        "params": 138_000_000,
        "type": "CNN",
        "description": "Deeper network with 3×3 convolutions",
        "input_size": "224×224",
        "notable": "Showed depth matters",
    },
    {
        "name": "ResNet-50",
        "year": 2015,
        "params": 25_000_000,
        "type": "CNN",
        "description": "Introduced skip connections",
        "input_size": "224×224",
        "notable": "Enabled training of very deep networks",
    },
    {
        "name": "BERT-base",
        "year": 2018,
        "params": 110_000_000,
        "type": "Transformer",
        "description": "Bidirectional transformer for NLP",
        "input_size": "512 tokens",
        "notable": "Pre-training + fine-tuning paradigm",
    },
    {
        "name": "GPT-2",
        "year": 2019,
        "params": 1_500_000_000,
        "type": "Transformer",
        "description": "Large language model for text generation",
        "input_size": "1024 tokens",
        "notable": "Showed emergent abilities with scale",
    },
    {
        "name": "GPT-3",
        "year": 2020,
        "params": 175_000_000_000,
        "type": "Transformer",
        "description": "Few-shot learning, in-context learning",
        "input_size": "2048 tokens",
        "notable": "100× larger than GPT-2",
    },
    {
        "name": "GPT-4",
        "year": 2023,
        "params": 1_800_000_000_000,
        "type": "Transformer",
        "description": "Multimodal, most capable LLM",
        "input_size": "32k tokens",
        "notable": "10× larger than GPT-3 (estimated)",
    },
]


def format_params(params: int) -> str:
    """Format parameter count as human-readable string."""
    if params >= 1_000_000_000_000:
        return f"{params / 1_000_000_000_000:.1f}T"
    elif params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    elif params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    elif params >= 1_000:
        return f"{params / 1_000:.1f}K"
    return str(params)


class ScaleService:
    """Service for model scale comparisons."""

    def get_all_models(self) -> dict[str, Any]:
        """Get all models with formatted data."""
        return {
            "models": [
                {
                    **model,
                    "params_formatted": format_params(model["params"]),
                }
                for model in MODELS
            ],
            "total_count": len(MODELS),
        }

    def get_cnns(self) -> dict[str, Any]:
        """Get only CNN models."""
        cnn_models = [m for m in MODELS if m["type"] == "CNN"]
        return {
            "models": [
                {
                    **model,
                    "params_formatted": format_params(model["params"]),
                }
                for model in cnn_models
            ],
            "total_count": len(cnn_models),
        }

    def get_transformers(self) -> dict[str, Any]:
        """Get only Transformer models."""
        transformer_models = [m for m in MODELS if m["type"] == "Transformer"]
        return {
            "models": [
                {
                    **model,
                    "params_formatted": format_params(model["params"]),
                }
                for model in transformer_models
            ],
            "total_count": len(transformer_models),
        }

    def get_comparison(self, model1: str, model2: str) -> dict[str, Any]:
        """Compare two models by name."""
        m1 = next((m for m in MODELS if m["name"].lower() == model1.lower()), None)
        m2 = next((m for m in MODELS if m["name"].lower() == model2.lower()), None)

        if m1 is None:
            raise ValueError(f"Unknown model: {model1}")
        if m2 is None:
            raise ValueError(f"Unknown model: {model2}")

        ratio = m2["params"] / m1["params"] if m1["params"] > 0 else 0

        return {
            "model1": {**m1, "params_formatted": format_params(m1["params"])},
            "model2": {**m2, "params_formatted": format_params(m2["params"])},
            "ratio": ratio,
            "ratio_formatted": f"{ratio:.1f}×",
            "year_gap": m2["year"] - m1["year"],
            "explanation": (
                f"{m2['name']} has {ratio:.1f}× more parameters than {m1['name']}, "
                f"released {abs(m2['year'] - m1['year'])} years {'later' if m2['year'] > m1['year'] else 'earlier'}."
            ),
        }

    def get_growth_data(self) -> dict[str, Any]:
        """Get data showing exponential growth over time."""
        return {
            "data": [
                {
                    "name": m["name"],
                    "year": m["year"],
                    "params": m["params"],
                    "log_params": len(str(m["params"])),  # Order of magnitude
                    "type": m["type"],
                }
                for m in MODELS
            ],
            "insight": (
                "Model sizes have grown exponentially. "
                "From LeNet-5 (60K) to GPT-4 (1.8T) is a 30,000,000× increase over 25 years."
            ),
        }

