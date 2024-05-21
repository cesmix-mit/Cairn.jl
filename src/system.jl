import Molly: accelerations
import PotentialLearning: get_local_descriptors, get_force_descriptors, compute_local_descriptors, compute_force_descriptors
export get_atoms, get_coords, define_sys, define_ens, remove_loggers



"""
define_sys(
    inter,
    coords::Vector{<:Real};
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*u"nm"),
    loggers=false,
    data=nothing,
)

Defines the System struct for a single-atom system.
"""
function define_sys(
    inter,
    coords::Vector{<:Real};
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*u"nm"),
    loggers=false,
    data=nothing,
)
    d = length(coords)
    atoms = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ)]
    coords = [SVector{d}(coords) .* u"nm"] # initial position
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


"""
define_sys(
    inter,
    coords::Vector{<:Real};
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*u"nm"),
    loggers=false,
    data=nothing,
)

Defines a Vector{<:System} for an ensemble of single-atom system.
"""
function define_ens(
    inter,
    coords::Vector{<:Union{Vector,SVector}};
    atom_mass=1.0u"g/mol",
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
    boundary=RectangularBoundary(Inf*u"nm"),
    loggers=false,
    data=nothing,
)
    n = length(coords)
    d = length(coords[1])
    atoms = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ) for i in 1:n]

    if loggers == false
        sys = [System(
            atoms=[atoms_i],
            coords=[coords_i],
            boundary=boundary,
            general_inters=(inter,),
            data=deepcopy(data),
        ) for (atoms_i, coords_i) in zip(atoms, coords)]
    else
        sys = [System(
            atoms=[atoms_i],
            coords=[coords_i],
            boundary=boundary,
            general_inters=(inter,),
            loggers=deepcopy(loggers),
            data=deepcopy(data),
        ) for (atoms_i, coords_i) in zip(atoms, coords)]
    end

    return sys
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
    ) for sys in ens]
end


"""
    get_atoms(ens::Vector{<:System})

Returns a vector of atoms contained in all systems of the ensemble `ens`.
"""
get_atoms(ens::Vector{<:System}) = [sys.atoms for sys in ens]


"""
    get_coords(ens::Vector{<:System})

Returns a vector of atomic coordinates contained in all systems of the ensemble `ens`.
"""
get_coords(ens::Vector{<:System}) = [sys.coords for sys in ens]
get_coords(ens::Vector{<:System}, t::Integer) = [sys.loggers.coords.history[t] for sys in ens]


"""
    get_local_descriptors(sys::System)
    get_local_descriptors(ens::Vector{<:System})

Returns the local energy descriptors of the system, `sys`, or ensemble of systems, `ens`.
"""
get_local_descriptors(sys::System) = sys.data["energy_descriptors"]
get_local_descriptors(ens::Vector{<:System}) = [sys.data["energy_descriptors"] for sys in ens]


"""
    get_force_descriptors(sys::System)
    get_force_descriptors(ens::Vector{<:System})

Returns the per-atom force descriptors of the system, `sys`, or ensemble of systems, `ens`.
"""
get_force_descriptors(sys::System) = sys.data["force_descriptors"]
get_force_descriptors(ens::Vector{<:System}) = [sys.data["force_descriptors"] for sys in ens]


"""
    compute_local_descriptors(ens::Vector{<:System}, inter)

Computes the local energy descriptors each system in the ensemble.
"""
compute_local_descriptors(ens::Vector{<:System}, inter) = [compute_local_descriptors(sys, inter) for sys in ens]


"""
    compute_force_descriptors(ens::Vector{<:System}, inter)

Computes the per-atom force descriptors each system in the ensemble.
"""
compute_force_descriptors(ens::Vector{<:System}, inter) = [compute_force_descriptors(sys, inter) for sys in ens]


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

