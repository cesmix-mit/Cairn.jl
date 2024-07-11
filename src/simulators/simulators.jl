## initialize simulation
# function initialize_sim!(
#     sys::System,
#     sim;
#     n_threads::Integer=Threads.nthreads(),
#     run_loggers=true,
#     )

#     sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
#     !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
#     nb = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
#     run_loggers!(sys, nb, 0, run_loggers; n_threads=n_threads)

#     return sys, nb
# end


include("overdampedlangevin.jl")
include("stochasticsvgd.jl")
include("srld.jl")