#!/usr/bin/env python3
"""Configuration-driven entry point for model validation."""

import sys
from pathlib import Path

from core.evaluation import (
    build_validation_inputs, evaluate_pde_chunked,
    evaluate_predictions_chunked, regional_metrics, sample_cases,
)
from core.validation import main
from core.validation_plots import save_validation_plots


CONFIG_PATH = Path("configs/validate_model.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH)
