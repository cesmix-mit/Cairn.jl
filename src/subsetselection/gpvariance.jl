export GPVariance, get_subset

struct GPVariance <: SubsetSelector
    var :: Vector
    batch_size :: Int
end


function GPVariance(
    ds::DataSet,
    trainset::DataSet,
    f::Feature,
    k::Kernel;
    batch_size = length(ds) ÷ 4,
    dt = LocalDescriptors,
)
    Σ11 = KernelMatrix(trainset, f, k; dt = dt)
    Σ12 = KernelMatrix(trainset, ds, f, k; dt = dt)
    Σ22 = KernelMatrix(ds, f, k; dt = dt)

    Σ2 = Σ22 - Σ12' * pinv(Σ11, 1e-4) * Σ12 # cond. cov.
    σ = diag(Σ2)
    return GPVariance(σ, batch_size)
end



function get_subset(
    gp::GPVariance,
    batch_size::Int = gp.batch_size
)
    σ = diag(gp.cov) # variance
    indices = partialsortperm(σ, 1:batch_size, rev=true)
    return indices
end
