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


function add_train_data(
    sys::System,
    sys_train::Vector{<:System},
    ss::RandomSelector;
    t0=1,
)
    # extract data from sys
    coords = values(sys.loggers.coords)[t0:end]
    inter = sys.general_inters[1]

    # select new training data
    ids = get_random_subset(ss, num_configs=length(coords))
    coords_new = coords[ids]
    sys_new = Ensemble(inter, coords_new; data=deepcopy(sys.data))

    # populate with descriptors
    compute_local_descriptors(sys_new, inter)
    compute_force_descriptors(sys_new, inter)

    # append to training set
    return reduce(vcat, [sys_train, sys_new])
end


function add_train_data(
    sys::System,
    sys_train::Vector{<:System},
    ss::kDPP;
    t0=1,
)
    # extract data from sys
    coords = values(sys.loggers.coords)[t0:end]
    inter = sys.general_inters[1]

    # select new training data
    kDPP()
    ids = get_random_subset(ss, num_configs=length(coords))
    coords_new = coords[ids]
    sys_new = Ensemble(inter, coords_new; data=deepcopy(sys.data))

    # populate with descriptors
    compute_local_descriptors(sys_new, inter)
    compute_force_descriptors(sys_new, inter)

    # append to training set
    return reduce(vcat, [sys_train, sys_new])
end


