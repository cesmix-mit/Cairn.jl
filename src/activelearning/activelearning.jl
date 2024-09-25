include("alroutine.jl")
include("aldata.jl")
# include("distributions.jl")
include("kernels.jl")



export
    active_learn!




"""
    active_learn!(sys::Union{System, Vector{<:System}}, sim::Simulator, n_steps::Integer, sys_train::Vector{<:System}, ref::Union{GeneralInteraction, PairwiseInteraction}, trigger::Union{Bool, ActiveLearningTrigger}; n_threads::Integer=Threads.nthreads(), run_loggers=true)

Performs online active learning by molecular dynamics simulation defined in `sim`, using the retraining criterion defined in `trigger`.

# Arguments
- `sys::Union{System, Vector{<:System}}` : a system or ensemble of systems to simulate
- `sim::Simulator` : simulator of the equations of motion
- `n_steps::Integer` : number of simulation time steps
- `sys_train::Vector{<:System}` : ensemble of systems representing the training data
- `ref::Union{GeneralInteraction, PairwiseInteraction}` : interaction for computing reference values
- `trigger::Union{Bool, ActiveLearningTrigger}` : trigger which instantiates retraining
- `n_threads::Integer=Threads.nthreads()` : number of threads
- `run_loggers=true` : Bool for running loggers
    
"""
function active_learn!(sys::System,
    sim,
    n_steps::Integer,
    al::ActiveLearnRoutine;
    n_threads::Integer=Threads.nthreads(),
    run_loggers=true,
    rng=Random.GLOBAL_RNG,
)
    sys, neighbors = initialize_sim!(sys; n_threads=n_threads, run_loggers=run_loggers)
    compute_error_metrics!(al)


    for step_n in 1:n_steps
        neighbors, ksd = simulation_step!(sys,
                            sim,
                            step_n,
                            n_threads=n_threads,
                            neighbors=neighbors,
                            run_loggers=run_loggers,
                            rng=rng,
        )

        sys_new = remove_loggers(sys)   
        sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new])

        # online active learning 
        if trigger_activated(sys, al.triggers; sys_train=sys_train, step_n=step_n)
            al.sys_train = add_train_data()
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end



# single trajectory MD
function active_learn!(sys::System,
            sim::SteinRepulsiveLangevin,
            n_steps::Integer,
            al::ActiveLearnRoutine;
            n_threads::Integer=Threads.nthreads(),
            run_loggers=true,
            rng=Random.GLOBAL_RNG,
)
    sys, neighbors = initialize_sim!(sys; n_threads=n_threads, run_loggers=run_loggers)
    compute_error_metrics!(al)
    

    for step_n in 1:n_steps
        neighbors, ksd = simulation_step!(sys,
                            sim,
                            step_n,
                            n_threads=n_threads,
                            neighbors=neighbors,
                            run_loggers=run_loggers,
                            rng=rng,
        )

        sys_new = remove_loggers(sys)   
        sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new])

        # online active learning 
        if trigger_activated(al.trigger; step_n=step_n, ksd=ksd)
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end

# ensemble (multi-trajectory) MD
function active_learn!(ens::Vector{<:System},
            sim::StochasticSVGD,
            n_steps::Integer,
            al::ActiveLearnRoutine;
            n_threads::Integer=Threads.nthreads(),
            run_loggers=true,
            rng=Random.GLOBAL_RNG,
)

    N = length(ens)
    T = typeof(find_neighbors(ens[1], n_threads=n_threads))
    nb_ens = Vector{T}(undef, N)
    bwd = zeros(n_steps)
    compute_error_metrics!(al)

    # initialize
    for (sys, nb) in zip(ens, nb_ens)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
        nb = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        run_loggers!(sys, nb, 0, run_loggers; n_threads=n_threads)
    end
    
    for step_n in 1:n_steps
        nb_ens, ksd, bwd[step_n] = simulation_step!(ens,
                    nb_ens,
                    sim,
                    step_n,
                    n_threads=n_threads,
                    run_loggers=run_loggers,
                    rng=rng,
        )

        # online active learning 
        if trigger_activated(al.trigger; step_n=step_n, ens_old=al.sys_train, ens_new=ens, ksd=ksd)
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, ens, al.sys_train) 
            al.train_func(ens, al.sys_train, al.ref) # retrain potential
            al.mlip = ens[1].general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [ens[1].general_inters[1].params])
            compute_error_metrics!(al)
        end
    end

    return al, bwd
end


function active_learn!(sys::System,
    sim::OverdampedLangevin,
    n_steps::Integer,
    al::ActiveLearnRoutine;
    n_threads::Integer=Threads.nthreads(),
    run_loggers=true,
    rng=Random.GLOBAL_RNG,
)
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    compute_error_metrics!(al)

    for step_n in 1:n_steps
        neighbors = simulation_step!(sys,
                        sim,
                        step_n,
                        n_threads=n_threads,
                        neighbors=neighbors,
                        run_loggers=run_loggers,
                        rng=rng,
        )

        # online active learning 
        if trigger_activated(al.trigger; ens_old=al.sys_train, sys_new=sys, step_n=step_n)
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end