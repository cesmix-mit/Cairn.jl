export
    ActiveLearnRoutine,
    active_learn!,
    update_sys,
    remove_loggers,
    compute_error_metrics!


"""
    ActiveLearnRoutine

Struct containing all parameters required to trigger retraining during simulation.

# Arguments
- `ref :: Union{GeneralInteraction, PairwiseInteraction}` : reference (ground truth) potential 
- `mlip :: MLInteraction` : machine learning (surrogate) potential)
- `sys_train :: Vector{<:System}` : ensemble of systems representing the training data
- `eval_int` : Integrator specifying mode of integration
- `trigger :: ActiveLearningTrigger` : defines trigger function for active learning
- `error_hist :: Dict` : dictionary of error metrics to record over simulation

# Keyword Arguments
- `burnin :: Integer = 0` : number of burn-in simulation steps before triggering training 
- `update_func :: Function = update_sys` : function for updating the the training and simulation systems
- `train_func :: Function - train_potential_e!` : function defining training objective for `mlip`
- `train_steps :: Vector{<:Integer} = [1]` : vector of simulation steps at which training is trigger_activated
- `param_hist :: Vector{<:Vector} = [mlip.params]` : vector of the parameter history
- `kwargs...` : additional keyword arguments for `update_func` and `train_func`

""" 

mutable struct ActiveLearnRoutine
    ref :: Union{GeneralInteraction, PairwiseInteraction}
    mlip :: MLInteraction
    sys_train :: Vector{<:System}
    eval_int :: Integrator
    trigger :: ActiveLearningTrigger
    error_hist :: Dict
    burnin :: Integer
    update_func :: Function
    train_func :: Function
    train_steps :: Vector{<:Integer}
    param_hist :: Vector{<:Vector}
    
end

function ActiveLearnRoutine(
        ref::Union{GeneralInteraction, PairwiseInteraction},
        mlip::MLInteraction,
        sys_train::Vector{<:System},
        eval_int::Integrator,
        trigger::ActiveLearningTrigger,
        error_hist::Dict; 
        burnin::Integer = 0,
        update_func::Function = update_sys,
        train_func::Function = train_potential_e!,
        kwargs...
)
        
        train_steps = [1]
        param_hist = [mlip.params]
        
        update(sim, sys, ref) = update_func(sim, sys, ref; kwargs...)
        train(sys, sys_train, ref) = train_func(sys, sys_train, ref; kwargs...)
        return ActiveLearnRoutine(ref, mlip, sys_train, eval_int, trigger, error_hist, burnin, update, train, train_steps, param_hist)
end



"""
    active_learn!(sys::Union{System, Vector{<:System}}, sim::Simulator, n_steps::Integer, sys_train::Vector{<:System}, ref::Union{GeneralInteraction, PairwiseInteraction}, trigger::Union{Bool, ActiveLearningTrigger}; n_threads::Integer=Threads.nthreads(), burnin::Integer=100, run_loggers=true)

Performs online active learning by molecular dynamics simulation defined in `sim`, using the retraining criterion defined in `trigger`.

# Arguments
- `sys::Union{System, Vector{<:System}}` : a system or ensemble of systems to simulate
- `sim::Simulator` : simulator of the equations of motion
- `n_steps::Integer` : number of simulation time steps
- `sys_train::Vector{<:System}` : ensemble of systems representing the training data
- `ref::Union{GeneralInteraction, PairwiseInteraction}` : interaction for computing reference values
- `trigger::Union{Bool, ActiveLearningTrigger}` : trigger which instantiates retraining
- `n_threads::Integer=Threads.nthreads()` : number of threads
- `burnin::Integer=100` : number of burn-in steps before active learning routine
- `run_loggers=true` : Bool for running loggers
    
"""
# single trajectory MD
function active_learn!(sys::System,
            sim::SteinRepulsiveLangevin,
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
    
    ct = 0
    for step_n in 1:n_steps
        ct += 1
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
        if trigger_activated(al.trigger; step_n=step_n, ksd=ksd) && ct >= al.burnin
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
            ct = 0 # reset counter
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
    
    ct = 0
    for step_n in 1:n_steps
        ct += 1
        nb_ens, ksd, bwd[step_n] = simulation_step!(ens,
                    nb_ens,
                    sim,
                    step_n,
                    n_threads=n_threads,
                    run_loggers=run_loggers,
                    rng=rng,
        )

        # online active learning 
        if trigger_activated(al.trigger; step_n=step_n, ens_old=al.sys_train, ens_new=ens, ksd=ksd) && ct >= al.burnin
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, ens, al.sys_train) 
            al.train_func(ens, al.sys_train, al.ref) # retrain potential
            al.mlip = ens[1].general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [ens[1].general_inters[1].params])
            compute_error_metrics!(al)
            ct = 0 # reset counter
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

    ct = 0
    for step_n in 1:n_steps
        ct += 1
        neighbors = simulation_step!(sys,
                        sim,
                        step_n,
                        n_threads=n_threads,
                        neighbors=neighbors,
                        run_loggers=run_loggers,
                        rng=rng,
        )

        # online active learning 
        if trigger_activated(al.trigger; ens_old=al.sys_train, sys_new=sys, step_n=step_n) && ct >= al.burnin
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
            ct = 0 # reset counter
        end
    end
    return al
end

"""
    update_sys(sim::Simulator)

Performs online active learning by molecular dynamics simulation defined in `sim`, using the retraining criterion defined in `trigger`.

# Arguments
- `sim::Simulator` : simulator of the equations of motion
- `sys::Union{System,Vector{<:System}}` : simulation system
- `sys_train::Vector{<:System}`: Vector of systems in the training data

# Keyword Arguments
- `n_add::Integer=10` : number of samples to add to `sys_train`
- `steps::Integer=1000` : number of steps in the sample path over which to draw new training data
    
"""
function update_sys(sim::OverdampedLangevin,
    sys::System,
    sys_train::Vector{<:System};
    n_add=10,    
    steps=1000,         
)
    coords = sys.loggers.coords.history[end:-1:(end-steps)]
    coords_new = StatsBase.sample(coords, n_add; replace=false)

    sys_new = [System(
        atoms=sys.atoms,
        coords=coords_i,
        boundary=sys.boundary,
        general_inters=sys.general_inters,
    ) for coords_i in coords_new]

    sys_train = reduce(vcat, [sys_train, sys_new])

    return sys_train
end


function update_sys(sim::SteinRepulsiveLangevin,
    sys::System,
    sys_train::Vector{<:System};    
    kwargs...           
)
    sys_new = remove_loggers(sys)   
    # sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new]) # update repulsive set
    sys_train = reduce(vcat, [sys_train, sys_new]) # update training data

    return sys_train
end


function update_sys(sim::StochasticSVGD,
    ens::Vector{<:System}, 
    sys_train::Vector{<:System};    
    kwargs...          
)
    ens_new = [remove_loggers(sys) for sys in ens]
    sys_train = reduce(vcat, [sys_train, ens_new]) # update training data
    sim.sys_fix = ens_new # sys_train
    return sys_train
end


"""
remove_loggers(sys::System)

A function which refines the system `sys` without loggers.
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
    compute_error_metrics!(al::ActiveLearnRoutine)

Appends error metric calculations (RMSE in energies and forces, Fisher divergence in probability measures) to `al.error_hist`.

"""
function compute_error_metrics!(al::ActiveLearnRoutine)
    fd = compute_fisher_div(al.ref, al.mlip, al.eval_int)
    r_e, r_f = compute_rmse(al.ref, al.mlip, al.eval_int)
    append!(al.error_hist["fd"], fd)
    append!(al.error_hist["rmse_e"], r_e)
    append!(al.error_hist["rmse_f"], r_f)
end
