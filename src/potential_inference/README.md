# Potential inference from PC-components (PyTorch)

This small module trains a multilayer perceptron (MLP) that maps PC components x -> scalar potential V(x). The model's predicted forces are obtained as the negative gradient of the potential w.r.t. the inputs (force = -∇_x V). Training minimizes the MSE between predicted forces and provided target forces.

Files added
- `train_potential.py`: single-file training CLI with dataset, model, autograd force computation, synthetic data generator, training/validation loop, checkpointing.

Quick start

1. Create a virtualenv and install requirements:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r ProtLatDyn/src/potential_inference/requirements.txt

2. Run synthetic quick experiment (fast):

    python ProtLatDyn/src/potential_inference/train_potential.py --synthetic --epochs 50

3. To use your data: save a NumPy `.npz` with two arrays named `pc` and `forces` (both shape `(M, N_pc)`) and run:

    python ProtLatDyn/src/potential_inference/train_potential.py --data-path path/to/data.npz --epochs 200

Notes
- `train_potential.py` saves the best model to `best_potential.pth` by default. You can change with `--save-path`.
- The script expects forces in the same coordinate space as the input `pc` vectors. By default predicted_force = -dV/dx; change sign in code if needed.

Next steps / extensions
- Add regularization on curvature (Hessian) if you want smoothness.
- Add learning-rate warmup, mixed precision, or dataset streaming for very large datasets.
- Wrap model training in a CI test, or add a small unit test that trains on synthetic data for a few epochs and checks loss decreases.

If you want, I can: add unit tests, split the file into modules, or integrate with your data pipeline.
