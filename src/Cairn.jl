"""

Cairn.jl

Author: cesmix-mit
Version: 0.1.0
Year: 2024
Notes: Enhanced sampling algorithms for the active learning of machine learning interatomic potentials (ML-IPs), implemented in Julia.
"""

module Cairn

using LinearAlgebra 
using Statistics
using StatsBase
using StaticArrays
using Random
using Distributions
using Polynomials
using SpecialPolynomials
using AtomsBase
using Unitful
using UnitfulAtomic
using Molly
using AtomsCalculators
using InteratomicPotentials
using PotentialLearning
using AtomisticQoIs
using Maxvol
using ProgressBars

include("data/data.jl") # TODO: migrate to PotentialLearning

include("PotentialLearningExt.jl")

include("interactions/doublewell.jl")
include("interactions/himmelblau.jl")
include("interactions/sinusoid.jl")
include("interactions/pce.jl")
include("interactions/muller_brown_rotated.jl")
include("interactions/interatomicpotential.jl")

include("simulators/simulators.jl")

include("triggers/triggers.jl")

include("training/training.jl")

include("loggers/loggers.jl")

include("subsetselection/subsetselector.jl") # TODO: migrate to PotentialLearning

include("activelearning/activelearning.jl")



end
