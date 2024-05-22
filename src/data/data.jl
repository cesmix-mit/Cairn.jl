import InteratomicPotentials: compute_local_descriptors, compute_force_descriptors
import PotentialLearning: get_local_descriptors, get_force_descriptors, get_values, Energy, Force, Forces, Configuration
import Unitful: Quantity

export get_atoms, get_coords, TrainConfiguration, TrainDataSet
include("system.jl")
include("types.jl")



get_values(qt::Vector{<:Quantity}) = ustrip.(values(qt))
get_values(qt::Union{SVector{2, <:Quantity}, SVector{3, <:Quantity}}) = Vector(ustrip.(values(qt)))
get_values(qt::Quantity) = ustrip(qt)


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
function compute_local_descriptors(
    ens::Vector{<:System},
    inter;
    pbar = true
)

Computes the local energy descriptors each system in the ensemble using threads.
"""
function compute_local_descriptors(
    ens::Vector{<:System},
    inter;
    pbar = true
)
    iter = collect(enumerate(ens))
    if pbar
        iter = ProgressBar(iter)
    end
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        e_des[j] = LocalDescriptors(compute_local_descriptors(sys, inter))
    end
    return e_des
end


"""
function compute_force_descriptors(
    ens::Vector{<:System},
    inter;
    pbar = true
)

Compute force descriptors of a basis system and dataset using threads.
"""
function compute_force_descriptors(
    ens::Vector{<:System},
    inter;
    pbar = true
)
    iter = collect(enumerate(ens))
    if pbar
        iter = ProgressBar(iter)
    end
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    Threads.@threads for (j, sys) in iter
        f_des[j] = ForceDescriptors([fi for fi in compute_force_descriptors(sys, inter)])
    end
    return f_des
end


Energy(e::Quantity) = Energy(get_values(e), unit(e))


Force(f::Union{SVector{2, <:Quantity}, SVector{3, <:Quantity}}) = Force(get_values(f), unit.(f)[1])


Forces(f::Vector{<:SVector}) = Forces(Force.(f))


function TrainConfiguration(
    sys::System,
    ref,
    mlip=sys.general_inters[1],
)
    e = Energy(potential_energy(sys, ref))
    f = Forces(forces(sys, ref))
    ed = try 
        LocalDescriptors(sys.data["energy_descriptors"])
    catch 
        LocalDescriptors(compute_local_descriptors(sys, mlip))
    end
    if length(ed) == 0
        ed = LocalDescriptors(compute_local_descriptors(sys, mlip))
    end
    fd = try
        ForceDescriptors(sys.data["force_descriptors"])
    catch
        ForceDescriptors(compute_force_descriptors(sys, mlip))
    end
    if length(fd) == 0
        fd = ForceDescriptors(compute_force_descriptors(sys, mlip))
    end
    return Configuration(e, f, ed, fd)
end

function TrainDataSet(
    ens::Vector{<:System},
    ref,
    mlip=ens[1].general_inters[1],
)
    configs = [TrainConfiguration(sys, ref, mlip) for sys in ens]
    return DataSet(configs)
end