import Molly: potential_energy, forces

export Himmelblau

@doc raw"""
    Himmelblau(; force_units, energy_units)

The Himmelblau potential energy surface with 4 minima.

The potential energy is defined as
```math
V(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

This potential is only compatible with 2D systems.

"""
struct Himmelblau{F, E} <: GeneralInteraction
    force_units::F
    energy_units::E
end

function Himmelblau(;
                    force_units = u"kJ * mol^-1 * nm^-1",
                    energy_units = u"kJ * mol^-1")

    return Himmelblau{typeof(force_units), typeof(energy_units)}(
        force_units, energy_units)
end 

@inline function potential_energy(inter::Himmelblau, sys, neighbors=nothing;
                                            n_threads::Integer=Threads.nthreads())
    return sum(potential_himmelblau.(Ref(inter), sys.coords))
end

@inline function potential_himmelblau(inter::Himmelblau, coord::SVector{2})
    x, y = ustrip.(coord)

    res = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    return res * inter.energy_units
end

@inline function forces(inter::Himmelblau, sys, neighbors=nothing;
                                  n_threads::Integer=Threads.nthreads())
    return force_himmelblau.(Ref(inter),sys.coords)
end

@inline function force_himmelblau(inter::Himmelblau, coord::SVector{2})
    x, y = ustrip.(coord)

    res_x = 4*x*(x^2 + y - 11) + 2*(x + y^2 - 7)
    res_y = 2*(x^2 + y - 11) + 4*y*(x + y^2 - 7)

    res = [-res_x, -res_y] .* inter.force_units
    return SVector{2}(res)
end