# MECA0010 - Stochastic Sensitivity Analysis of Dynamic Input Conductances

## Repository Overview

This repository contains the implementation and analysis for the project **"Stochastic Sensitivity Analysis of Dynamical Input Conductances"** as part of the MECA0010 course at the University of Liège. The project explores the uncertainty quantification and sensitivity analysis of conductance-based neuron models, specifically focusing on Dynamic Input Conductances (DICs).

## Repository Contents

- **`main.py`**: The main script to run the numerical simulations, Monte Carlo analysis, and sensitivity calculations.
- **`utils.py`**: Helper functions for sampling, statistical calculations, and numerical computations.
- **`stg.py`**: Core functions implementing the Stomatogastric Ganglion (STG) neuron model and its dynamics.
- [**`report.pdf`**](report.pdf): Comprehensive documentation of the project, including methodology, results, and analysis.
- **Figures and Outputs**: Generated visualizations such as convergence graphs, distributions, and sensitivity indices.

## Features

1. **Monte Carlo Simulations**:
   - Propagate uncertainty in maximal conductances through the model.
   - Generate distributions for DICs using kernel density estimation.

2. **Stochastic Sensitivity Analysis**:
   - Compute Sobol sensitivity indices (first-order and total effect).
   - Identify key parameters influencing fast, slow, and ultra-slow DICs.

3. **Convergence Analysis**:
   - Validate the robustness of results with respect to population size and sample count.

4. **Visualization Tools**:
   - Generate distribution plots, sensitivity heatmaps, and convergence analysis graphs.

## Methodology

### Problem Definition
The study investigates how uncertainty in maximal conductances affects the DICs, which summarize neuronal behavior over different timescales. The STG neuron model is used as a black-box system, where maximal conductances are treated as input parameters with associated uncertainties modeled by Gamma distributions.

### Computational Steps
1. **Define Uncertainty**: Assign Gamma distributions to input parameters based on biological priors.
2. **Simulate Outputs**: Use Monte Carlo sampling to evaluate DICs for a population of neurons.
3. **Analyze Sensitivity**: Compute Sobol indices to determine the importance of each conductance.
4. **Convergence Validation**: Ensure the reliability of statistical estimators and sensitivity indices.
