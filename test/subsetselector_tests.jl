using Cairn, Molly
using LinearAlgebra
using PotentialLearning
using SpecialPolynomials
using Unitful
using Distributions
using Test

@testset "Subselector Tests" begin


    # define data distributions and sample
    dist = MvNormal(zeros(2), 0.3.*I(2))
    xtrain = [rand(dist1) for i=1:100]

    # define evaluation data
    xrng = -2:0.1:2
    N = length(xrng)
    xpred = [[xrng[i], xrng[j]] for i=1:N, j=1:N]

    # define output function
    y(x) = sin(2*x[1])*cos(2*x[2])
    y(x::Configuration) = DFT(x), LennardJones(x)

    # define training set
    ytrain = Energy.(y.(xtrain))
    dtrain = LocalDescriptors.([[x] for x in xtrain])
    trainset = DataSet(Configuration.(ytrain, dtrain))

    # define prediction set
    dpred = LocalDescriptors.([[x] for x in xpred[:]])
    dataset = DataSet(Configuration.(dpred))

    # define kernel
    rbf = RBF(Euclidean(2))

    # define subset selector
    gp = GPVariance(dataset, trainset, GlobalMean(), rbf; batch_size=15)
    id = get_subset(gp)


    # plot
    yeval = y.(xpred)
    fig = Figure()
    ax = Axis(fig[1,1][1,1], 
            xlabel="x1", ylabel="x2",
            title="true function")
    
    ct = contourf!(ax, xrng, xrng, yeval)
    Colorbar(fig[1, 1][1, 2], ct, label="y(x)")
    fig

    fig = Figure()
    ax = Axis(fig[1,1][1,1], 
            xlabel="x1", ylabel="x2",
            title="GP prediction")
    
    xmat = reduce(hcat, xtrain)'
    ct = contourf!(ax, xrng, xrng, reshape(gp.mean, (41,41)))
    scatter!(ax, xmat[:,1], xmat[:,2], color=:red, label="train. data")
    Colorbar(fig[1, 1][1, 2], ct, label="μ_GP(x)")
    axislegend(ax)
    fig

    fig = Figure()
    ax = Axis(fig[1,1][1,1], 
            xlabel="x1", ylabel="x2",
            title="GP prediction error")
    
    xmat = reduce(hcat, xtrain)'
    ct = contourf!(ax, xrng, xrng, yeval .- reshape(gp.mean, (41,41)))
    scatter!(ax, xmat[:,1], xmat[:,2], color=:red, label="train. data")
    Colorbar(fig[1, 1][1, 2], ct, label="y(x) - μ_GP(x)")
    axislegend(ax)
    fig

    fig = Figure()
    ax = Axis(fig[1,1][1,1], 
            xlabel="x1", ylabel="x2",
            title="GP prediction variance")
    
    xmat = reduce(hcat, xtrain)'
    ct = contourf!(ax, xrng, xrng, reshape(diag(gp.cov), (41,41)))
    scatter!(ax, xmat[:,1], xmat[:,2], color=:red, label="train. data")
    Colorbar(fig[1, 1][1, 2], ct, label="σ_GP(x)")
    axislegend(ax)
    fig







    ## using descriptors

    # define potentials
    ref = MullerBrown()
    pce = PolynomialChaos(3, 2, Jacobi{0.5,0.5})
    n = length(pce.basis)
    pce.params = ones(n)

    systrain = define_ens(pce, xtrain)
    dtrain =  compute_local_descriptors(systrain, pce)
    trainset = DataSet(Configuration.(ytrain, dtrain))
    
    syspred = define_ens(pce, xpred)
    dpred = compute_local_descriptors(syspred, pce)
    dataset = DataSet(Configuration.(dpred))

    rbf = RBF(Euclidean(n))
end