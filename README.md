# Cairn

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cesmix-mit.github.io/Cairn.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cesmix-mit.github.io/Cairn.jl/dev/)
[![Build Status](https://github.com/cesmix-mit/Cairn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cesmix-mit/Cairn.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://travis-ci.com/cesmix-mit/Cairn.jl.svg?branch=main)](https://travis-ci.com/cesmix-mit/Cairn.jl)
<!-- [![Build Status](https://ci.appveyor.com/api/projects/status/github/cesmix-mit/Cairn.jl?svg=true)](https://ci.appveyor.com/project/cesmix-mit/Cairn-jl)
[![Coverage](https://codecov.io/gh/cesmix-mit/Cairn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cesmix-mit/Cairn.jl)
[![Coverage](https://coveralls.io/repos/github/cesmix-mit/Cairn.jl/badge.svg?branch=main)](https://coveralls.io/github/cesmix-mit/Cairn.jl?branch=main) -->

Cairn.jl is a toolkit of active learning algorithms for training machine learning interatomic potentials (ML-IPs) for molecular dynamics simulation. 

Cairn.jl is constructed as an extension to [Molly.jl](https://github.com/JuliaMolSim/Molly.jl), implementing enhanced MD samplers, and interfaces with other packages in the Julia ecosystem for molecular simulation, developed by [CESMIX](https://github.com/cesmix-mit) and [JuliaMolSim](https://github.com/JuliaMolSim). 

Active learning algorithms build efficient training datasets which maximally improve accuracy of a scientific machine learning model. These algorithms take an iterative structure, looping through the steps: 

1. **Data generation**. A system's potential energy landscape is sampled by generating trajectories of molecular configurations through the simulation of Newton's equation of motion or its modifications. Users have a choice between standard MD simulation, such as Langevin dynamics or Velocity-Verlet, or enhanced sampling algorithms, such as uncertainty driven dynamics ([UDD](https://www.nature.com/articles/s43588-023-00406-5)), Stein repulsive Langevin dynamics, or Stein variational molecular dynamics. These methods are specified under the abstract type `Simulator`. 

2. **Trigger for retraining.** Sampling is terminated and retraining is triggered when the trajectory has met a certain criteria. A "fixed trigger" calls on retraining after a fixed number of simulation steps. "Adaptive triggers" are based on metrics of uncertainty, from Gaussian process or ensemble-based estimates of variance; metrics of extrapolation, based on a MaxVol vector basis; or metrics of diversity, such as a DPP inclusion probability. These methods are specified under the abstract type `ActiveLearningTrigger`.

3. **Data subset selection and labelling.** A subset of the data from the simulated trajectory is selected for labelling using reference calculations and appending to the training set. The most basic selection is a random subset of the trajectory. "Adaptive" selections can be made based on data which exceeds a threshold or data which are chosen by a subset selection algorithm, such as MaxVol, k-means clustering, or DPPs. These methods are specified under the abstract type `SubsetSelector`. 

4. **Model updating.** The machine learning model is retrained on the augmented dataset according to the choice of objective function defined by the abstract type `LinearProblem`. These methods live in the package [PotentialLearning.jl](https://github.com/cesmix-mit/PotentialLearning.jl).


For a technical manual on the package, see the [docs](cesmix-mit.github.io/Cairn.jl/). For a demo, see the Jupyter notebooks in the `examples` folder. 


