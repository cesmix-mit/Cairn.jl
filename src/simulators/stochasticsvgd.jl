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
mutable struct StochasticSVGD{S, K, T, F} <: Simulator
    dt::S
    kernel::K
    kernel_bandwidth::Function
    sys_fix::Union{Nothing, Vector}
    temperature::T
    friction::F
    remove_CM_motion::Int
end

function StochasticSVGD(; dt, kernel, kernel_bandwidth=const_kernel_bandwidth, sys_fix=nothing, temperature, friction, remove_CM_motion=1)
    return StochasticSVGD(dt, kernel, kernel_bandwidth, sys_fix, temperature, friction, Int(remove_CM_motion))
end


function simulation_step!(ens::Vector{<:System},
                    neighbor_ens::Vector,
                    sim::StochasticSVGD,
                    step_n::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)

    if sim.sys_fix == nothing
        M = length(ens)
        accels_t = accelerations(ens; neighbor_ens=neighbor_ens, n_threads=n_threads)
        force_units = unit(accels_t[1][1][1])

        # update kernel bandwidth
        sim.kernel.ℓ = sim.kernel_bandwidth(ens, sim.kernel)
        Kt, ∇Kt = compute_kernelized_forces(ens, sim.kernel)

    else
        ens_all = [ens; sim.sys_fix]
        M = length(ens_all)
        accels_t = accelerations(ens_all; n_threads=n_threads)
        force_units = unit(accels_t[1][1][1])

        # update kernel bandwidth
        sim.kernel.ℓ = sim.kernel_bandwidth(ens_all, sim.kernel)
        Kt, ∇Kt = compute_kernelized_forces(ens, sim.sys_fix, sim.kernel)
    end

    # interaction terms
    for (i,sys) in enumerate(ens)
        Kti = Kt[:,i]
        ∇Kti = ∇Kt[:,i]
        old_coords = copy(sys.coords)
        noise = random_velocities(sys, sim.temperature; rng=rng)

        knl_t = sum(Kti .* accels_t) ./ sim.friction ./ M * sim.dt
        gradknl_t = [sum(∇Kti.*force_units)] ./ sim.friction ./ M * sim.dt
        noise_t = sqrt((2 / sim.friction) * sim.dt) .* noise
        sys.coords += knl_t .+ gradknl_t .+ noise_t

        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        neighbor_ens[i] = find_neighbors(sys, sys.neighbor_finder, neighbor_ens[i], step_n; n_threads=n_threads)
        
        run_loggers!(
            sys,
            neighbor_ens[i],
            step_n,
            run_loggers;
            n_threads=n_threads,
            stepcomp=[knl_t, gradknl_t, noise_t],
        )
    end

    return neighbor_ens, sim.kernel.ℓ
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
        run_loggers!(sys, nb_ens, 0, run_loggers; n_threads=n_threads)
    end

    # run simulation
    for step_n in 1:n_steps
        nb_ens, bwd[step_n] = simulation_step!(ens, 
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


