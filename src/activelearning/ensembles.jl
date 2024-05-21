import Molly: accelerations
import PotentialLearning: get_local_descriptors, get_force_descriptors
export get_atoms, get_coords

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

get_local_descriptors(sys::System) = sys.data["energy_descriptors"]
get_force_descriptors(sys::System) = sys.data["force_descriptors"]

get_local_descriptors(ens::Vector{<:System}) = [sys.data["energy_descriptors"] for sys in ens]
get_force_descriptors(ens::Vector{<:System}) = [sys.data["force_descriptors"] for sys in ens]

compute_local_descriptors(ens::Vector{<:System}, inter) = [compute_local_descriptors(sys, inter) for sys in ens]
compute_force_descriptors(ens::Vector{<:System}, inter) = [compute_force_descriptors(sys, inter) for sys in ens]


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

