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
using InteratomicPotentials
using PotentialLearning
using AtomisticQoIs
using Maxvol

include("types.jl")

include("interactions/doublewell.jl")
include("interactions/himmelblau.jl")
include("interactions/sinusoid.jl")
include("interactions/pce.jl")
include("interactions/muller_brown_rotated.jl")

include("simulators/overdampedlangevin.jl")
include("simulators/stochasticsvgd.jl")
include("simulators/srld.jl")

include("activelearning/triggers.jl")
include("activelearning/distributions.jl")
include("activelearning/kernels.jl")
include("activelearning/ensembles.jl")
include("activelearning/training.jl")
include("activelearning/activelearning.jl")

include("loggers/triggerlogger.jl")
include("loggers/traininglogger.jl")
include("loggers/stepcomponentlogger.jl")
include("loggers/get_values.jl")




end
