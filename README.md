 ![](https://github.com/DongweiYe/Gaussian-Process-Learning/blob/main/github_figure.png)
### Gaussian process learning for nonlinear dynamics
This repository is associated with the publication of "[Gaussian process learning of nonlinear dynamics](https://www.sciencedirect.com/science/article/pii/S1007570424003691)". 

This repository presents the implementation of four scenarios, including identification and estimation with an affine parametrization, nonlinear parametric approximation without prior knowledge, and general parameter estimation for a given dynamical system. Each scenario is corresponding to the numerical examples presented in the paper: 

**Example 1 - Parameter estimation for Lotka–Volterra model (with analytical posterior)**
- `LotkaVolterra_model.py`: contain the function of Lotka Volterra model.
- `data_generation.py`: execute to generate the data to train a data-driven model.
- `Scene1.py`: execute to perform Gaussian process learning. It also includes the implementation of the comparison method FD+LinReg.
- `plot_compare.py`: visualize the comparison result.
      
**Example 2 - Sparse identification for Lotka-Volterra model** 
- `LotkaVolterra_model.py`: contain the function of Lotka Volterra model.
- `data_generation.py`: execute the generate the data to train a data-driven model.
- `Scene2.py`: execute to perform Gaussian process learning. It also includes the implementation of the comparison method SINDy.
- `plot_compare.py`: visualize the comparison result (corresponding to figure 6).
- `plot_error.py`: visualize the error under different noise and data density (corresponding to figure 5).
- `visualization.py`: contain the visualization function for the posterior distributions of candidate parameters.

**Example 3 - A neural network surrogate for 1D ODE system**
- `dynamical_system.py`: contain the function of 1D ODE system.
- `data_generation.py`: execute the generate the data to train a data-driven model.
- `Scene3.py`: execute to perform Gaussian process learning.
- `Scene3_IC.py`: execute to perform Gaussian process learning with data from multiple trajectories.

**Example 4 - Parameter estimation for binary black-hole system**
- `dynamical_system.py`: contain the function of 1D ODE system.
- `data_generation.py`: execute the generate the data to train a data-driven model.
- `Scene4.py`: execute to perform Gaussian process learning.

Note that some minor changes in hyperparameters/visualization are required for different data conditions. For those details, see the comments in the code.
