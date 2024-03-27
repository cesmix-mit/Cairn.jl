import Molly: potential_energy, forces # GeneralInteraction, 

export Sinusoid

@doc raw"""
    Sinusoid(; force_units, energy_units)

A sinusoidal potential energy surface with multiple basins.

The potential energy is defined as
```math
V(x,y) = x^2/20 + y^2/20 + sin(x) + sin(y)
```

This potential is only compatible with 2D systems.

"""
struct Sinusoid{F, E} <: GeneralInteraction
    force_units::F
    energy_units::E
end

function Sinusoid(;
                    force_units = u"kJ * mol^-1 * nm^-1",
                    energy_units = u"kJ * mol^-1")

    return Sinusoid{typeof(force_units), typeof(energy_units)}(
        force_units, energy_units)
end 

@inline function potential_energy(inter::Sinusoid, sys, neighbors=nothing;
                                            n_threads::Integer=Threads.nthreads())
    return sum(potential_sinusoid.(Ref(inter), sys.coords))
end

@inline function potential_sinusoid(inter::Sinusoid, coord::SVector{2})
    x, y = ustrip.(coord)

    res = x^2/20 + y^2/20 + sin(x) + sin(y)

    return res * inter.energy_units
end

@inline function forces(inter::Sinusoid, sys, neighbors=nothing;
                                  n_threads::Integer=Threads.nthreads())
    return force_sinusoid.(Ref(inter),sys.coords)
end

@inline function force_sinusoid(inter::Sinusoid, coord::SVector{2})
    x, y = ustrip.(coord)

    res_x = x/10 + cos(x)
    res_y = y/10 + cos(y)

    res = [-res_x, -res_y] .* inter.force_units
    return SVector{2}(res)
end