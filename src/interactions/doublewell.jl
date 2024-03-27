import Molly: potential_energy, forces # GeneralInteraction, 

export DoubleWell

@doc raw"""
    DoubleWell(; force_units, energy_units)

The double well potential energy surface with 2 minima.

The potential energy is defined as
```math
V(x,y) = (1/6)*(4*(1-x^2-y^2)^2 + 2*(x^2-2)^2 + ((x+y)^2-1)^2 + ((x-y)^2-1)^2)
```

This potential is only compatible with 2D systems.

"""
struct DoubleWell{F, E} <: GeneralInteraction
    force_units::F
    energy_units::E
end

function DoubleWell(;
                    force_units = u"kJ * mol^-1 * nm^-1",
                    energy_units = u"kJ * mol^-1")

    return DoubleWell{typeof(force_units), typeof(energy_units)}(
        force_units, energy_units)
end 

@inline function potential_energy(inter::DoubleWell, sys, neighbors=nothing;
                                            n_threads::Integer=Threads.nthreads())
    return sum(potential_double_well.(Ref(inter), sys.coords))
end

@inline function potential_double_well(inter::DoubleWell, coord::SVector{2})
    x, y = ustrip.(coord)

    res = 1/6 * (4*(1-x^2-y^2)^2
        + 2*(x^2-2)^2
        + ((x+y)^2-1)^2
        + ((x-y)^2-1)^2)

    return res * inter.energy_units
end

@inline function forces(inter::DoubleWell, sys, neighbors=nothing;
                                  n_threads::Integer=Threads.nthreads())
    return force_double_well.(Ref(inter),sys.coords)
end

@inline function force_double_well(inter::DoubleWell, coord::SVector{2})
    x, y = ustrip.(coord)

    res_x = 4/3 * x *(4*x^2 + 5*y^2 - 5)
    res_y = 4/3 * y *(5*x^2 + 3*y^2 - 3)

    res = [-res_x, -res_y] .* inter.force_units
    return SVector{2}(res)
end