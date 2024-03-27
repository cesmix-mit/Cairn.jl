import Molly: simulate!
export SteinRepulsiveLangevin

"""
    SteinRepulsiveLangevin(; <keyword arguments>)

Simulates a hybrid simulator combining OverdampedLangevin and a Stein repulsive term, after Ye et al. 2020 (https://arxiv.org/abs/2002.09070).

# Arguments
- `dt::S`               : the time step of the simulation.
- `kernel::K`           : kernel used for computing kernelized forces.
- `sys_fix::Vector{X}`  : Vector of systems of fixed atoms for computing kernelized forces.
- `temperature::T`      : the equilibrium temperature of the simulation.
- `friction::F`         : the friction coefficient of the simulation.
- `remove_CM_motion=1`  : remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
mutable struct SteinRepulsiveLangevin{S, K, X, T, F} <: Simulator
    dt::S
    kernel::K
    sys_fix::Vector{X}
    temperature::T
    friction::F
    remove_CM_motion::Int
end

function SteinRepulsiveLangevin(; dt, kernel, sys_fix, temperature, friction, remove_CM_motion=1)
    return SteinRepulsiveLangevin(dt, kernel, sys_fix, temperature, friction, Int(remove_CM_motion))
end


function simulation_step!(sys::System,
                    sim::SteinRepulsiveLangevin,
                    step_n::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)

    accels_t = accelerations(sys, neighbors; n_threads=n_threads)

    # interaction terms
    N = length(sim.sys_fix)
    accels_fix = [accelerations(sys_i, nothing; n_threads=n_threads) for sys_i in sim.sys_fix]
    Kt, ∇Kt = compute_kernelized_forces(sys, sim.sys_fix, sim.kernel)

    old_coords = copy(sys.coords)
    noise = random_velocities(sys, sim.temperature; rng=rng)

    drift_t = (accels_t ./ sim.friction) .* sim.dt
    knl_t = (sum(Kt .* accels_fix) ./ sim.friction) ./ N .* sim.dt
    gradknl_t = ([sum(∇Kt.*u"kJ * g^-1 * nm^-1")] ./ sim.friction) ./ N .* sim.dt
    ksd_t = knl_t + gradknl_t
    noise_t = sqrt((2 / sim.friction) * sim.dt) .* noise
    sys.coords +=  drift_t .+ knl_t .+ gradknl_t .+ noise_t

    apply_constraints!(sys, old_coords, sim.dt)
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
        remove_CM_motion!(sys)
    end

    neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                n_threads=n_threads)

    # add force components to loggers
    if has_step_property(sys)
        sys.loggers.drift.observable = drift_t
        sys.loggers.knl.observable = knl_t
        sys.loggers.gradknl.observable = gradknl_t
        sys.loggers.noise.observable = noise_t
        sys.loggers.ksd.observable = ksd_t
    end
    run_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, kernel=Kt)

    return neighbors, Kt
end



function simulate!(sys::System,
                    sim::SteinRepulsiveLangevin,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)
                
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)

    for step_n in 1:n_steps
        neighbors, _ = simulation_step!(sys,
                                    sim,
                                    step_n,
                                    n_threads=n_threads,
                                    neighbors=neighbors,
                                    run_loggers=run_loggers,
                                    rng=rng,
        )
        sys_new = remove_loggers(sys)   
        sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new])

    end

end



