using Cairn
using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using PotentialLearning
using Molly, AtomsCalculators
using AtomisticQoIs
using SpecialPolynomials, SpecialFunctions

include("./src/makie/makie.jl")
# include("./examples/utils.jl")



## initialize -------------------------------------------------------------------------------------------

ref = MullerBrownRot()

# define initial system
temp = 100.0u"K"
x0 = [-1.5, 1.0]
sys0 = define_sys(
    ref,
    x0,
    loggers=(coords=CoordinateLogger(1000; dims=2),),
)

# define main support
limits = [[-4.4,1.5],[-2,2]]
coord_grid = coord_grid_2d(limits, 0.05)
ctr_lvls = -150:20:400



## evaluation set ---------------------------------------------------------------------------------------
# grid over main support
coords_eval = potential_grid_2d(ref, limits, 0.05, cutoff = 1000)
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



## draw initial training set from OverdampedLangevin ----------------------------------------------------

# define simulator
sim_langevin = OverdampedLangevin(
            dt=0.001u"ps",
            temperature=1200.0u"K",
            friction=4.0u"ps^-1")

sys = deepcopy(sys0)
simulate!(sys, sim_langevin, 1_000_000)
f = plot_md_trajectory(sys, coord_grid, fill=false, lvls=ctr_lvls, showpath=false)


# define initial ensemble
nens = 100
id = StatsBase.sample(1:length(sys.loggers.coords.history), nens, replace=false)
coords0 = [sys.loggers.coords.history[i][1] for i in id] 

ens0 = define_ens(ref, coords0,
    loggers=(
        coords=CoordinateLogger(10; dims=2),
    ),
)

# define svgd simulator
rbf = RBF(Euclidean(2), β=1.0, ℓ=1.0)
sim_svgd = StochasticSVGD(
            dt=0.001u"ps",
            kernel=rbf,
            kernel_bandwidth=median_kernel_bandwidth,
            temperature=20.0u"K",
            friction=1.0u"ps^-1")

simulate!(ens0, sim_svgd, 10_000)
    

# plot trajectory
f = plot_md_trajectory(ens0, coord_grid, fill=false, lvls=ctr_lvls, showpath=false)


# training data - fixed
# coords_train = [sys.coords[1] for sys in ens0]
coords_train = coords0
sys_train = define_ens(ref, coords_train,
    data=Dict(
        "energy_descriptors" => [],
        "force_descriptors" => [],
    )
)



## set up model ------------------------------------------------------------------------------

# define model
basisfam = Jacobi{0.5,0.5} # ChebyshevU
order = 8 # [5,10,15,20,25,30] 
pce = PolynomialChaos(order, 2, basisfam, xscl=limits)

# train initial model on initial dataset
pce0 = deepcopy(pce)

# ensemble - active
nens = 10
ens0 = define_ens(pce0, coords_train[1:nens],
    loggers=(
        coords=CoordinateLogger(10; dims=2),
        steps=StepComponentLogger(10; dims=2),
    ),
    data=Dict(
        "energy_descriptors" => Float64[],
        "force_descriptors" => Vector[],
        "kernel" => 1.0,
        "ksd" => 1.0,
    )
)


# train model
train!(ens0, sys_train, ref)
pce0.params = ens0[1].general_inters[1].params


# plot
ctr_lvls0 = -300:20:100
f0, ax0 = plot_contours_2d(pce0, coord_grid; fill=true, lvls=ctr_lvls0)
coordmat = reduce(hcat, get_values(coords_train))'
scatter!(ax0, coordmat[:,1], coordmat[:,2], color=:red, label="training points")
axislegend(ax0)
f0



## active learning with StochasticSVGD ------------------------------------------------------------------------------

# define kernel
rbf = RBF(Euclidean(2), β=0.2)


# define simulator
sim_svgd = StochasticSVGD(
            dt=0.001u"ps",
            kernel=rbf,
            kernel_bandwidth=median_kernel_bandwidth,
            sys_fix=sys_train,
            temperature=temp,
            friction=1.0u"ps^-1")

# simulate
ens = deepcopy(ens0)
simulate!(ens, sim_svgd, 1_000)


# define triggers
trig1 = TimeInterval(interval=100)
trig2 = MaxVol(thresh=1.22)

# define active learning routine 
al = ActiveLearnRoutine(
    ref=ref,
    mlip=pce0,
    trainset=sys_train,
    triggers=(trig1, trig2),
    dataselector=RandomSelector,
    trainobj=LinearLeastSquares,
)

# run active learning loop
ens = deepcopy(ens0)
al, bwd = active_learn!(ens, sim_svgd, 5_000, al)