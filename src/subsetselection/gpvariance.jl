export GPVariance, get_subset

struct GPVariance <: SubsetSelector
    mean :: Vector
    cov :: Matrix
    batch_size :: Int
end


function GPVariance(
    ds::DataSet,
    trainset::DataSet,
    f::Feature,
    k::Kernel;
    batch_size=length(ds) ÷ 4,
    dt = LocalDescriptors,
)
    Σ11 = KernelMatrix(trainset, f, k; dt = dt)
    Σ12 = KernelMatrix(ds, trainset, f, k; dt = dt)
    Σ22 = KernelMatrix(ds, f, k; dt = dt)

    μ1 = get_all_energies(trainset)
    μ2 = (pinv(Σ11) * Σ12)' * μ1 # cond. mean
    Σ2 = Σ22 - (pinv(Σ11) * Σ12)' * Σ12 # cond. cov.

    return GPVariance(μ2, Σ2)
end



function get_subset(
    gp::GPVariance,
    batch_size::Int = gp.batch_size
)
    σ = diag(gp.cov) # variance
    indices = partialsortperm(σ, 1:batch_size, rev=true)
    return indices
end
