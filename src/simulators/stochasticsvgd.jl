import Molly: simulate!
export StochasticSVGD

"""
    StochasticSVGD(; <keyword arguments>)

Simulates a stochastic variant of Stein variational gradient descent (SVGD).

# Arguments
- `dt::S`               : the time step of the simulation.
- `kernel::K`           : kernel used for computing kernelized forces.
- `kernel_bandwidth::Function`  : function which computes the kernel bandwidth.
- `sys_fix::Vector{X}`  : Vector of systems of fixed atoms for computing kernelized forces.
- `temperature::T`      : the equilibrium temperature of the simulation.
- `friction::F`         : the friction coefficient of the simulation.
- `remove_CM_motion=1`  : remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
mutable struct StochasticSVGD{S, K, X, T, F} <: Simulator
    dt::S
    kernel::K
    kernel_bandwidth::Function
    sys_fix::Vector{X}
    temperature::T
    friction::F
    remove_CM_motion::Int
end

function StochasticSVGD(; dt, kernel, kernel_bandwidth=const_kernel_bandwidth, sys_fix, temperature, friction, remove_CM_motion=1)
    return StochasticSVGD(dt, kernel, kernel_bandwidth, sys_fix, temperature, friction, Int(remove_CM_motion))
end


function simulation_step!(ens::Vector{<:System},
                    neighbor_ens::Vector,
                    sim::StochasticSVGD,
                    step_n::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)

    N = length(ens)
    M = N + length(sim.sys_fix)
    ksd_ens = Vector{Vector}(undef, N)
    ens_all = reduce(vcat, [ens, sim.sys_fix])
    accels_t = accelerations(ens_all; n_threads=n_threads)
    # accels_ens = accelerations(ens; neighbor_ens=neighbor_ens, n_threads=n_threads)
    # accels_fix = accelerations(sim.sys_fix; n_threads=n_threads)
    # accels_t = reduce(vcat, [accels_ens, accels_fix])

    # update kernel bandwidth
    sim.kernel.ℓ = sim.kernel_bandwidth(ens_all, sim.kernel)
    # Kt_all, ∇Kt_all = compute_kernelized_forces(ens, sim.kernel)
    Kt_all, ∇Kt_all = compute_kernelized_forces(ens, sim.sys_fix, sim.kernel)

    # interaction terms
    for (i,sys) in enumerate(ens)
        # Kt, ∇Kt = compute_kernelized_forces(sys, ens, sim.kernel)
        Kt = Kt_all[:,i]
        ∇Kt = ∇Kt_all[:,i]
        old_coords = copy(sys.coords)
        noise = random_velocities(sys, sim.temperature; rng=rng)

        knl_t = sum(Kt .* accels_t) ./ sim.friction
        gradknl_t = [sum(∇Kt.*u"kJ * g^-1 * nm^-1")] ./ sim.friction
        ksd_t = (knl_t + gradknl_t) ./ M
        noise_t = sqrt((2 / sim.friction) * sim.dt) .* noise
        sys.coords += ksd_t * sim.dt .+ noise_t

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbor_ens[i] = find_neighbors(sys, sys.neighbor_finder, neighbor_ens[i], step_n; n_threads=n_threads)
        
        # add force components to loggers
        if has_step_property(sys)
            # sys.loggers.knl.observable = knl_t / M * sim.dt
            # sys.loggers.gradknl.observable = gradknl_t / M * sim.dt
            # sys.loggers.noise.observable = noise_t
            sys.loggers.ksd.observable = ksd_t * sim.dt
        end
        run_loggers!(sys, neighbor_ens[i], step_n, run_loggers; n_threads=n_threads, ens_old=sim.sys_fix, ens_new=ksd=ksd_t)

        ksd_ens[i] = ksd_t
    end

    return neighbor_ens, ksd_ens, sim.kernel.ℓ
end



function simulate!(ens::Vector{<:System},
                    sim::StochasticSVGD,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)

    N = length(ens)
    T = typeof(find_neighbors(ens[1], n_threads=n_threads))
    nb_ens = Vector{T}(undef, N)
    bwd = zeros(n_steps)

    # initialize
    for (sys, nb) in zip(ens, nb_ens)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
        nb = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        run_loggers!(sys, nb, 0, run_loggers; n_threads=n_threads)
    end

    # run simulation
    for step_n in 1:n_steps
        nb_ens, ksd_ens, bwd[step_n] = simulation_step!(ens, 
                                nb_ens,
                                sim,
                                step_n,
                                n_threads=n_threads,
                                run_loggers=run_loggers,
                                rng=rng,
        )
    end

    return bwd

end


