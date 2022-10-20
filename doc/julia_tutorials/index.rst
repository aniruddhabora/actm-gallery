:orphan:

Operator Learning Tools
=============

* **ACTM Performer:** Caltech team;
* **Author:** Costa Christopoulos (cchristo@caltech.edu) and Tapio Schneider (tapio@caltech.edu)


This section provides an overview of the CliMA code, including links to documentation containing tutorials.
The modeling and learning framework for this project is modular, and code for each component resides in unique repositories.
A hybrid machine-learning model (`TurbulenceConvection.jl`) is calibrated with Ensemble Kalman Processes in `CalibrateEDMF.jl`.
As training data [linked below] we employ a library of Large Eddy Simulations (LES), driven by conditions found in state of the art climate 
simulations at selected location on the globe.

 
Code repositories and data libraries associated with the project:

.. list-table:: CliMA Packages
   :widths: 25 25 25 25
   :header-rows: 1

   * - Package
     - Code
     - Docs
     - Purpose
   * - CalibrateEDMF.jl
     - `Link <https://github.com/CliMA/CalibrateEDMF.jl>`_
     - `Link <https://clima.github.io/CalibrateEDMF.jl/dev/>`_
     - Framework to learn about cloud processes from data
   * - EnsembleKalmanProcesses.jl
     - `Link <https://github.com/CliMA/EnsembleKalmanProcesses.jl>`_
     - `Link <https://clima.github.io/EnsembleKalmanProcesses.jl/dev/>`_
     - Implementation of gradient-free optimization techniques   
   * - TurbulenceConvection.jl
     - `Link <https://github.com/CliMA/TurbulenceConvection.jl>`_
     - `Link <https://clima.github.io/TurbulenceConvection.jl/dev/>`_
     - Implementation of EDMF scheme of turbulence, convection and clouds
   * - OperatorFlux.jl
     - `Link <https://github.com/CliMA/OperatorFlux.jl>`_
     -
     - A machine learning package for Fourier Neural Operators
   * - LES library
     - `Link <https://data.caltech.edu/records/20052>`_
     -
     - LES generated training data at current climate and 4K warming simulations
