import Molly: accelerations
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

