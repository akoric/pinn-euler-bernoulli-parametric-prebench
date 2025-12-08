# Physics-Informed Neural Network for Euler-Bernoulli Beam Deflection

A parametric Physics-Informed Neural Network (PINN) implementation for predicting deflection in clamped Euler-Bernoulli beams under varying material properties and loading conditions.

## Overview
This project implements a neural network that learns to solve the fourth-order beam deflection equation across a range of parameters:
- **Young's Modulus (E)**: 100 to 300 GPa
- **Moment of Inertia (I)**: 0.5×10⁻⁶ to 2×10⁻⁶ m⁴  
- **Distributed Load (q)**: 500 to 1500 N/m

The model enforces boundary conditions for a clamped-free beam configuration, where the left end is fully constrained (w=0, w'=0) and the right end is free (w''=0, w'''=0). Training combines physics-based loss (PDE residual), boundary condition enforcement, and data matching with analytical solutions.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Training the model:**
```bash
python train.py
```
Training runs for 6000 epochs and typically completes in approximately 75 seconds on modern hardware.

**Evaluating model performance:**
```bash
python test.py
```
Tests the trained model on 10 different parameter combinations including edge cases.

**Generating visualizations:**
```bash
python visualize.py
```
Creates comparison plots showing analytical solutions versus PINN predictions.

## Project Structure
- **models.py** — Neural network architecture and analytical solution implementation
- **train.py** — Training loop with importance sampling and validation 
- **test.py** — Evaluation suite for generalization testing
- **visualize.py** — Visualization tools for model comparison
- **requirements.txt** — Python package dependencies

## Performance
The model achieves approximately ~3-4% relative RMSE on the validation set. Generalization performance is strong within the training parameter ranges, with some degradation observed for out-of-distribution cases

## Implementation Details
- **Architecture**: 3 hidden layers with 64 neurons each, using Tanh activation
- **Training Strategy**: Importance sampling to improve coverage in high-gradient regions
- **Normalization**: Physics-based scaling factor s = qL⁴/(EI) for numerical stability
- **Loss Components**: Weighted combination of PDE residual, boundary conditions, and data fitting
- **Output**: Trained model saved as `parametric_pinn_model.pt`

## Technical Notes
The implementation uses a trial solution approach where boundary conditions at x=0 are automatically satisfied through the network architecture (multiplication by x²). Free-end conditions at x=L are enforced through the loss function. The normalized PDE residual (EI·w⁽⁴⁾/q - 1) improves training stability across different parameter ranges.

For improved performance on extreme parameter values, consider increasing the number of training samples in those regions or extending the training duration.

## Training Notes
- Uses mini-batches of 512 samples drawn from the 80% training split (with replacement); boundary conditions use 64 points at x=1 per step.
- Spatial coordinates are sampled with a Beta(0.5, 0.5) distribution to put more emphasis near the boundaries; parameters (E, I, q) use light importance sampling.
- Loss is a weighted mix: 30× PDE residual, 100× boundary conditions, 1× data fit; deflections are normalized by s = qL⁴/(EI) for stability.
- Reproducibility: fixed seed (42) and saved validation indices in the checkpoint; visualization reuses the exact 20% validation set.
- Typical run: 6000 steps in ~70–75 seconds on an M‑series Mac; the final model is saved to `parametric_pinn_model.pt`.

## Requirements
- Python 3.8 or higher
- PyTorch 2.2 or higher
- NumPy 1.24+
- Matplotlib 3.7+
- tqdm 4.66+

## Future Work
The next phase of this project will extend the forward PINN to an inverse formulation, estimating material stiffness (EI) and load intensity (q) from sparse FEM deflection data. This inverse approach will enable parameter identification from measurement data, which has practical applications in structural health monitoring and material characterization.
