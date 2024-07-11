import Molly: accelerations, Atom, System
export Ensemble, get_atoms, get_coords, define_ens, remove_loggers



# function convert_to_molly_sys(
#     sys::FlexibleSystem;
#     dist_units=u"nm",
#     mass_units=u"g/mol",
#     )
# end
"""
function System(
    inter,
    coords::Vector{<:Real};
    dist_units=u"nm",
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)

Defines the System struct for a single-atom system.
"""
# single atom system
function System(
    inter,
    coord::Vector{<:Real};
    dist_units=u"nm",
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)
    d = length(coord)
    atom = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ)]
    coord = [SVector{d}(coord) .* dist_units] # initial position
    if loggers == false
        sys = System(
            atoms=atom,
            coords=coord,
            boundary=boundary,
            general_inters=(inter,),
            data=deepcopy(data),
        )
    else 
        sys = System(
            atoms=atom,
            coords=coord,
            boundary=boundary,
            general_inters=(inter,),
            loggers=deepcopy(loggers),
            data=deepcopy(data),
        )
    end

    return sys
end
function System(
    inter,
    coord::SVector; # with units
    dist_units=unit(coord[1]),
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)
    return System(
        inter,
        get_values(coord);
        dist_units=dist_units,
        atom_mass=atom_mass,
        σ=σ,
        ϵ=ϵ,
        boundary=boundary,
        loggers=loggers,
        data=data,
    )
end


# multi-atom system
function System(
    inter,
    coords::Vector{<:Vector{<:Real}};
    dist_units=u"nm",
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)
    n = length(coords) # number of atoms
    d = length(coords[1]) # state dimension
    atoms = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ) for i=1:n]
    coords = [SVector{d}(coord) .* dist_units for coord in coords] # initial position
    if loggers == false
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            general_inters=(inter,),
            data=deepcopy(data),
        )
    else 
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            general_inters=(inter,),
            loggers=deepcopy(loggers),
            data=deepcopy(data),
        )
    end

    return sys
end

function System(
    inter,
    coords::Vector{<:SVector};
    dist_units=unit(coords[1][1]),
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)
    return System(
        inter,
        [get_values(coord) for coord in coords];
        dist_units=dist_units,
        atom_mass=atom_mass,
        σ=σ,
        ϵ=ϵ,
        boundary=boundary,
        loggers=loggers,
        data=data,
    )
end



"""
function Ensemble(
    inter,
    coords::Vector;
    dist_units=u"nm",
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)

Defines a Vector{<:System} for an ensemble of systems.
"""
function Ensemble(
    inter,
    coords::Vector;
    dist_units=u"nm",
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*dist_units),
    loggers=false,
    data=nothing,
)
    ens = [System(
            inter,
            coord;
            dist_units=dist_units,
            atom_mass=atom_mass,
            σ=σ,
            ϵ=ϵ,
            boundary=boundary,
            loggers=loggers,
            data=data,
        ) for coord in coords
    ]

    return ens
end


"""
remove_loggers(sys::System)
remove_loggers(ens::Vector{<:System})

A function which refines the system, `sys`, or ensemble of systems, `ens`, without loggers.
"""
# redefine the system without loggers
function remove_loggers(
    sys::System
)
    return System(
        atoms=sys.atoms,
        coords=sys.coords,
        boundary=sys.boundary,
        general_inters=sys.general_inters,
        data=sys.data,
    )
end
# redefine the system without loggers
function remove_loggers(
    ens::Vector{<:System},
)
    return [System(
        atoms=sys.atoms,
        coords=sys.coords,
        boundary=sys.boundary,
        general_inters=sys.general_inters,
        data=sys.data,
    ) for sys in ens]
end


# computes accelerations for each system in the ensemble
function accelerations(
            ens::Vector{<:System};
            neighbor_ens = nothing,
            n_threads::Integer=Threads.nthreads()
)
    if neighbor_ens != nothing # neighbors provided
        return [accelerations(
                sys,
                nb;
                n_threads=n_threads
                ) for (sys, nb) in zip(ens, neighbor_ens)]
    else # calculate neighbors using sys.neighbor_finder
        return [accelerations(
                sys,
                find_neighbors(sys; n_threads=n_threads), # check current_neighbors
                n_threads=n_threads
                ) for sys in ens]
    end
end

