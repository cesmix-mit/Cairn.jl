import PotentialLearning: get_metrics
function get_metrics(
    ref,
    mli::MLInteraction,
    sys_eval::Vector{<:System},
)
    e_true = Molly.potential_energy.((ref,), sys_eval)
    e_pred = SteinMD.potential_energy.((mli,), sys_eval)
    f_true = reduce(vcat, [reduce(vcat, Molly.forces(ref, sys)) for sys in sys_eval])
    f_pred = reduce(vcat, [reduce(vcat, SteinMD.forces(mli, sys)) for sys in sys_eval])
    return get_metrics(e_pred, e_true, f_pred, f_true)
end



function pce_training_trial(
    ref::Union{GeneralInteraction, PairwiseInteraction},
    basisfam::Vector,
    order::Vector{<:Integer},
    dim::Integer,
    int::QuadIntegrator,
    train_func!::Function,
    train_sys::Vector,
    eval_sys::Vector,
    lim::Vector,
)

    mae = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)
    rmse = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)
    rsq = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)
    fd = Dict(bf => zeros(length(order)) for bf in basisfam)

    for bf in basisfam
        for (i,p) in enumerate(order)
            println(bf, p)
            pce = PolynomialChaos(p, dim, bf, xscl=lim)
            nsamp = length(pce.basis)*20
            println("training points: $nsamp")
            train_func!(train_sys[1:nsamp], ref, pce)

            # compute pointwise error metrics
            metrics = get_metrics(ref, pce, eval_sys)
            mae[bf]["e"][i] = ustrip(metrics["e_train_mae"])
            mae[bf]["f"][i] = ustrip(metrics["e_test_mae"])
            rmse[bf]["e"][i] = ustrip(metrics["e_train_rmse"])
            rmse[bf]["f"][i] = ustrip(metrics["e_test_rmse"])
            rsq[bf]["e"][i] = ustrip(metrics["e_train_rsq"])
            rsq[bf]["f"][i] = ustrip(metrics["e_test_rsq"])

            # compute discrepancy metrics
            p = define_gibbs_dist(ref, θ=[1])
            q = define_gibbs_dist(pce, θ=pce.params)
            f = FisherDivergence(int)
            fd[bf][i] = compute_discrepancy(p, q, f)
        end
    end

    return mae, rmse, rsq, fd
end


function pce_training_trial(
    ref::Union{GeneralInteraction, PairwiseInteraction},
    basisfam,
    order::Integer,
    dim::Integer,
    int::QuadIntegrator,
    train_func!::Function,
    train_sys::Vector,
    eval_sys::Vector,
    lim::Vector,
)
    pce = PolynomialChaos(order, dim, basisfam, xscl=lim)
    train_func!(train_sys, ref, pce)

    # compute pointwise error metrics
    metrics = get_metrics(ref, pce, eval_sys)
    mae_e = ustrip(metrics["e_train_mae"])
    mae_f = ustrip(metrics["e_test_mae"])
    rmse_e = ustrip(metrics["e_train_rmse"])
    rmse_f = ustrip(metrics["e_test_rmse"])
    rsq_e = ustrip(metrics["e_train_rsq"])
    rsq_f = ustrip(metrics["e_test_rsq"])

    # compute discrepancy metrics
    p = define_gibbs_dist(ref, θ=[1])
    q = define_gibbs_dist(pce, θ=pce.params)
    f = FisherDivergence(int)
    fd = compute_discrepancy(p, q, f)

    return [mae_e, mae_f], [rmse_e, rmse_f], [rsq_e, rsq_f], fd
end



# function pce_training_trial(
#     ref::Union{GeneralInteraction, PairwiseInteraction},
#     basisfam::Vector,
#     order::Vector{<:Integer},
#     dim::Integer,
#     train_func!::Function,
#     train_sys::Vector{<:Vector},
#     train_wts::Vector{<:Vector},
#     eval_sys::Vector,
#     lim::Vector,
# )

#     mae = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)
#     rmse = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)
#     rsq = Dict(bf => Dict("e" => zeros(length(order)), "f" => zeros(length(order))) for bf in basisfam)

#     for (j,bf) in enumerate(basisfam)
#         for (i,p) in enumerate(order)
#             println(bf, p)
#             pce = PolynomialChaos(p, dim, bf, xscl=lim)
#             train_func!(train_sys[j], ref, pce, wts=train_wts[j])
#             metrics = get_metrics(ref, pce, eval_sys)
#             mae[bf]["e"][i] = ustrip(metrics["e_train_mae"])
#             mae[bf]["f"][i] = ustrip(metrics["e_test_mae"])
#             rmse[bf]["e"][i] = ustrip(metrics["e_train_rmse"])
#             rmse[bf]["f"][i] = ustrip(metrics["e_test_rmse"])
#             rsq[bf]["e"][i] = ustrip(metrics["e_train_rsq"])
#             rsq[bf]["f"][i] = ustrip(metrics["e_test_rsq"])
#         end
#     end

#     return mae, rmse, rsq
# end



function define_sys(
    iap::Union{GeneralInteraction, PairwiseInteraction, MLInteraction},
    coords::Vector,
    boundary;
    σ=0.3u"nm",
    ϵ=0.2u"kJ * mol^-1",
)
    n = length(coords)
    atoms = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ) for i in 1:n]
    sys = [System(
        atoms=[atoms_i],
        coords=[coords_i],
        boundary=boundary,
        general_inters=(iap,),
        # k = 1.0u"kJ * K^-1 * mol^-1",
    ) for (atoms_i, coords_i) in zip(atoms, coords)]

    return sys
end



function init_trajectory(
    iap,
    x0::Vector;
    atom_mass=1.0u"g/mol",
    σ=1.0u"nm",
    ϵ=1.0u"kJ * mol^-1",
    logstep=1000,
)
    atoms = [Atom(mass=atom_mass, σ=σ, ϵ=ϵ)]
    coords = [SVector{2}(x0) .* u"nm"] # initial position
    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        general_inters=(iap,),
        loggers=(
            coords=CoordinateLogger(logstep; dims=2),
            # params=TrainingLogger(),
        )
    )
    return sys
end



function plot_error_metric(
    met_dicts::Vector{<:Dict},
    met_type::String,
    orders::Vector{<:Integer},
    labels::Vector{<:String};
    cols=[:skyblue, :orange, :goldenrod],
    lines=[:solid, :dash, :dot, :dashdot],
)

    keynames = collect(keys(met_dicts[1]))

    if typeof(met_dicts[1][keynames[1]]) <: Dict
        # energy error
        fig = Figure(resolution = (1200,600))

        ax1 = Axis(fig[1, 1], 
                xlabel="poly. order (p)",
                ylabel=met_type,
                title="Error in V",
                yscale=log10,
                )

        for (j,met) in enumerate(met_dicts)
            for (i,key) in enumerate(keynames)
                scatterlines!(ax1, orders, met[key]["e"], color=cols[j], linestyle=lines[i], label=string(key)*", "*labels[j])
            end
        end
        axislegend(ax1, position=:lb)

        ax2 = Axis(fig[1, 2], 
                xlabel="poly. order (p)",
                ylabel=met_type,
                title="Error in ∇xV",
                yscale=log10,
                )

        for (j,met) in enumerate(met_dicts)
            for (i,key) in enumerate(keynames)
                scatterlines!(ax2, orders, met[key]["f"], color=cols[j], linestyle=lines[i], label=string(key)*", "*labels[j])
            end
        end
        # axislegend(ax2)
    else
        fig = Figure(resolution = (600,600))

        ax1 = Axis(fig[1, 1], 
                xlabel="poly. order (p)",
                ylabel=met_type,
                title="Discrepancy",
                limits=(5, 25, 10^3, 10^5),
                yscale=log10,
                )

        for (j,met) in enumerate(met_dicts)
            for (i,key) in enumerate(keynames)
                scatterlines!(ax1, orders, met[key], color=cols[j], linestyle=lines[i], label=string(key)*", "*labels[j])
            end
        end
        axislegend(ax1, position=:lt)
    end

    return fig
end


    
function plot_error_metric(
    met::Dict,
    xlabel::String,
    ylabel::String,
    ttl::String,
    cols=[:skyblue, :gold, :darkorange2],
)
    trainobj = collect(keys(met)) # training objective
    keynames = collect(keys(met[trainobj[1]])) # numerical iterate

    fig = Figure(resolution = (600,600))

    ax1 = Axis(fig[1, 1], 
            xlabel=xlabel,
            ylabel=ylabel,
            title=ttl,
            xscale=log2,
            yscale=log10,
            # xticks = (order, string.(order)),
            # limits=(4, 31, 10^2.5, 10^4.5),
            xticks=(nsamp_arr, string.(nsamp_arr)),
            # limits=(180, 15000, 10^2.5, 10^5)
            )

    for (j,to) in enumerate(trainobj)
        err_med = [median(met[to][key]) for key in keynames]
        err_lqr = [quantile(met[to][key], 0.25) for key in keynames]
        err_uqr = [quantile(met[to][key], 0.75) for key in keynames]
        scatter!(ax1, keynames, err_med, color=cols[j], linestyle=:dash, label=to)
        rangebars!(ax1, keynames, err_lqr, err_uqr, color=cols[j], linewidth=2)
    end
    axislegend(ax1, position=:lt)

    return fig
end


# function compute_fisher_div(ref, mlip, integrator)
#     p = define_gibbs_dist(ref)
#     q = define_gibbs_dist(mlip, θ=mlip.params)
#     f = FisherDivergence(integrator)
#     return compute_discrepancy(p, q, f)
# end

# function compute_rmse(ref, mlip, integrator)
#     p = define_gibbs_dist(ref)
#     q = define_gibbs_dist(mlip, θ=mlip.params)

#     x_eval = integrator.ξ
#     e_true = p.V.(x_eval)
#     e_pred = q.V.(x_eval)
#     f_true = p.∇xV.(x_eval)
#     f_pred = q.∇xV.(x_eval)
    
#     Zp = normconst(p, integrator)
#     prob(x) = updf(p, x) / Zp
#     wts = prob.(x_eval) ./ length(x_eval)

#     rmse_e = wts' * rmse.(e_pred, e_true)
#     rmse_f = wts' * rmse.(f_pred, f_true)

#     return rmse_e, rmse_f
# end

# function compute_error_metrics!(al::ActiveLearnRoutine)
#     fd = compute_fisher_div(al.ref, al.mlip, al.eval_int)
#     r_e, r_f = compute_rmse(al.ref, al.mlip, al.eval_int)
#     append!(al.error_hist["fd"], fd)
#     append!(al.error_hist["rmse_e"], r_e)
#     append!(al.error_hist["rmse_f"], r_f)
# end

