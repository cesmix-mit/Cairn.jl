using Cairn
using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using PotentialLearning
using Molly, AtomsCalculators
using AtomisticQoIs
using SpecialPolynomials, SpecialFunctions

include("./src/makie/makie.jl")
include("./examples/utils.jl")



## define models ------------------------------------------------------------------
# choose reference model
ref = Himmelblau()

# define main support
limits = [[-6.5,6.5],[-6,6]]
# limits = [[-3.5,1.5],[-1.5,3.5]]
coord_grid = coord_grid_2d(limits, 0.1)
ctr_lvls = 0:25:400

# PCE properties
basisfam = Jacobi{0.5,0.5}
order = 5
pce0 = PolynomialChaos(order, 2, basisfam, xscl=limits)

# grid over main support
coords_eval = potential_grid_2d(ref, limits, 0.1, cutoff = 400)
sys_eval = define_ens(ref, coords_eval)

# use grid to define uniform quadrature points
ξ = [ustrip.(Vector(coords)) for coords in coords_eval]
GQint = GaussQuadrature(ξ, ones(length(ξ))./length(ξ))

# plot
f0, ax0 = plot_contours_2d(ref, coord_grid; fill=true, lvls=ctr_lvls)
coordmat = reduce(hcat, get_values(coords_eval))'
scatter!(ax0, coordmat[:,1], coordmat[:,2], color=:red, markersize=5, label="test points")
axislegend(ax0)
f0

# plot density 
f, _ = plot_density(ref, coord_grid, GQint)


# reference: train to test set
# pce = deepcopy(pce0)
# lp = learn!(sys_eval, ref, pce, [1000,1], false; e_flag=true, f_flag=true)
# p = define_gibbs_dist(ref)
# q = define_gibbs_dist(pce, θ=lp.β)
# fish = FisherDivergence(GQint)
# fd_best = compute_divergence(p, q, fish)


## training set 1: grid over main support ---------------------------------------
# sample from grid
coords1 = potential_grid_2d(ref, limits, 0.2, cutoff = 400)
trainset1 = define_ens(deepcopy(pce0), coords1)

# plot
f0, ax0 = plot_contours_2d(ref, coord_grid; fill=true, lvls=ctr_lvls)
coordmat = reduce(hcat, get_values(coords1))'
scatter!(ax0, coordmat[:,1], coordmat[:,2], color=:red, markersize=5, label="train set 1")
axislegend(ax0)
f0



## training set 2: samples from Langevin MD -------------------------------------
# Langevin simulator
sim_langevin = OverdampedLangevin(
            dt=0.002u"ps",
            temperature=500.0u"K",
            friction=4.0u"ps^-1",
)

x0arr = [[4.5, -2], [-3.5,3], [-3.5,-3]]
sys_langevin = Vector(undef, 3)
for (i,x0) in enumerate(x0arr)
    sys0 = define_sys(
                ref,
                x0,
                loggers=(coords=CoordinateLogger(100; dims=2),),
    )
    # simulate
    sys2 = deepcopy(sys0)
    simulate!(sys2, sim_langevin, 1_000_000)
    sys_langevin[i] = sys2
end


# subselect train data from the trajectory
n = [1335, 669, 669]
coords2 = [[sys_langevin[j].loggers.coords.history[i][1] for i=1:n[j]] for j=1:3]
coords2 = reduce(vcat, coords2)
trainset2 = define_ens(deepcopy(pce0), coords2)

# plot
f, ax = plot_contours_2d(ref, coord_grid; fill=true, lvls=ctr_lvls)
coordmat = reduce(hcat, get_values(coords2))'
scatter!(ax, coordmat[:,1], coordmat[:,2], color=:red, markersize=5, label="train set 2")
axislegend(ax)
f



## training set 3: samples from high-T MD -------------------------------------
# high-temp Langevin simulator
sim_highT = OverdampedLangevin(
            dt=0.002u"ps",
            temperature=2000.0u"K",
            friction=4.0u"ps^-1",
)
# simulate
sys3 = deepcopy(sys0)
simulate!(sys3, sim_highT, 2_000_000)
# f = plot_md_trajectory(sys3, coord_grid, fill=false, lvls=ctr_lvls, showpath=false)

# subselect train data from the trajectory
id = StatsBase.sample(1:length(sys3.loggers.coords.history), length(coords1), replace=false)
coords3 = [sys3.loggers.coords.history[i][1] for i in id] 
trainset3 = define_ens(deepcopy(pce0), coords3)

# plot
f, ax = plot_contours_2d(ref, coord_grid; fill=true, lvls=ctr_lvls)
coordmat = reduce(hcat, get_values(coords3))'
scatter!(ax, coordmat[:,1], coordmat[:,2], color=:red, markersize=5, label="train set 3")
axislegend(ax)
f


# train with changing weight λ --------------------------------------------------------------
λarr = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
trainsets = [trainset1, trainset2, trainset3]
p = define_gibbs_dist(ref)
fish = FisherDivergence(GQint)


# store results
param_dict = Dict( "ts$j" => Dict(
    "E" => zeros(length(pce.basis)),
    "F" => zeros(length(pce.basis)), 
    "EF" => Vector{Vector}(undef, length(λarr)),
    ) for j = 1:length(trainsets)
)

err_dict = Dict( "ts$j" => Dict(
    "E" => 0.0,
    "F" => 0.0, 
    "EF" => zeros(length(λarr)),
    ) for j = 1:length(trainsets)
)

fd_dict = Dict( "ts$j" => Dict(
    "E" => 0.0,
    "F" => 0.0, 
    "EF" => zeros(length(λarr)),
    ) for j = 1:length(trainsets)
)


# train on E or F only (UnivariateLinearProblem)
for (j,ts) in enumerate(trainsets)
    # E objective
    println("train set $j, E only")
    pce = deepcopy(pce0)
    lpe = learn!(ts, ref, pce; e_flag=true, f_flag=false)
    q = define_gibbs_dist(pce, θ=lpe.β)
    err_dict = 
    fd_dict["ts$j"]["E"] = compute_divergence(p, q, fish)
    param_dict["ts$j"]["E"] = lpe.β

    # F objective
    println("train set $j, F only")
    pce = deepcopy(pce0)
    lpf = learn!(ts, ref, pce; e_flag=false, f_flag=true)
    q = define_gibbs_dist(pce, θ=lpf.β)
    fd_dict["ts$j"]["F"] = compute_divergence(p, q, fish)
    param_dict["ts$j"]["F"] = lpf.β
end

# train on EF (CovariateLinearProblem)
for (i,λ) in enumerate(λarr)
    for (j,ts) in enumerate(trainsets)
        
        # EF objective
        println("train set $j, EF (λ=$λ)")
        pce = deepcopy(pce0)
        lpef = learn!(ts, ref, pce, [λ, 1], false; e_flag=true, f_flag=true)
        q = define_gibbs_dist(pce, θ=lpef.β)
        fd_dict["ts$j"]["EF"][i] = compute_divergence(p, q, fish)
        param_dict["ts$j"]["EF"][i] = lpef.β
    end
end



# plot results
λlab = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
f = Figure(resolution=(550,450))
ax = Axis(f[1,1],
    xlabel="λ",
    ylabel="Fisher divergence",
    title="Model Error vs. Weight λ",
    xscale=log10,
    yscale=log10,
    xticks=(λlab, ["F", "1e-4", "1e-3", "1e-2", "1e-1", "1", "1e1", "1e2", "1e3", "1e4", "E"]))

for j = 1:3
    fd_all = reduce(vcat, [[fd_dict["ts$j"]["F"]], fd_dict["ts$j"]["EF"], [fd_dict["ts$j"]["E"]]])
    scatterlines!(ax, λlab, fd_all, label="train set $j")
end
axislegend(ax, position=:lt)
f

pce.params = param_dict["ts2"]["E"]
ctr_lvls2 = -20:5:50 # for forces
f, _ = plot_contours_2d(pce, coord_grid, fill=true, lvls=ctr_lvls)
