import PotentialLearning: SubsetSelector

include("gpvariance.jl")


struct CurrentEnsemble <: SubsetSelector
end


struct CommitteeVariance <: SubsetSelector
    batch_size::Int
end

struct EntropyMaximization <: SubsetSelector 
    batch_size::Int
end

