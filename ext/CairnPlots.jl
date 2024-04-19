"""

CairnPlots.jl

Author: cesmix-mit
Version: 0.1.0
Year: 2024
Notes: Julia package extension with plotting functions complementing Cairn.jl.
"""
module CairnPlots

using Cairn
using Molly
using CairoMakie

# TODO: remove this once everything works!
__init__() = println("Extension was loaded!")

include("makie/plot_surface.jl")
include("makie/plot_contours.jl")
include("makie/plot_md.jl")
include("makie/plot_step.jl")
include("makie/plot_trigger.jl")

end