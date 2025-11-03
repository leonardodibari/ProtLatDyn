"""
Core analysis functions for ProtLatDyn
"""

import torch
import numpy as np


def example_function():
    """
    Example function to verify PyTorch installation
    """
    x = torch.randn(3, 3)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return x


if __name__ == "__main__":
    example_function()
