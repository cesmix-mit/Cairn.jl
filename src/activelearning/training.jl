export 
    train_potential_e!,
    train_potential_f!,
    train_potential_ef!,
    compute_importance_weights


"""
train_potential!

Solves for parameters of the machine learning interatomic potential (MLIP) using training data in `sys_train` and a reference potential `ref`.

# Arguments
- `sys::System` : system whose interactions are described by the MLIP
- `sys_train::Vector{<:System}` : ensemble of systems representing the training data
- `ref::Union{GeneralInteraction, PairwiseInteraction}` : interaction for computing reference values
- `mlip::MLInteraction = sys.general_inters[1]` : MLIP
- `α::Real = 1e-8` : conditioning parameter

"""
# single trajectory MD
function train_potential_e!(
    sys::System,
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = sys.general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = ones(length(sys_train)), # compute_importance_weights(sys_train, ref, mlip, kBT=kBT),
    α::Real = 1e-8,
    kwargs...
)
    
    # fit MLIP
    train_potential_e!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    sys.general_inters = (mlip,)
end

# ensemble (multi-trajectory) MD
function train_potential_e!(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = ens[1].general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = ones(length(sys_train)), # compute_importance_weights(sys_train, ref, mlip, kBT=kBT),
    α::Real = 1e-8,
    kwargs...
)
    train_potential_e!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    for sys in ens
        sys.general_inters = (mlip,)
    end
end

function train_potential_f!(
    sys::System,
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = sys.general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = ones(length(sys_train)), # compute_importance_weights(sys_train, ref, mlip, kBT=kBT),
    α::Real = 1e-8,
    kwargs...
)
    train_potential_f!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    sys.general_inters = (mlip,)
end

# ensemble (multi-trajectory) MD
function train_potential_f!(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = ens[1].general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = ones(length(sys_train)), # compute_importance_weights(sys_train, ref, mlip, kBT=kBT),
    α::Real = 1e-8,
    kwargs...
)
    train_potential_f!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    for sys in ens
        sys.general_inters = (mlip,)
    end
end

function train_potential_ef!(
    sys::System,
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = sys.general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = [1000,1],
    α::Real = 1e-8,
    kwargs...
)
    train_potential_ef!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    sys.general_inters = (mlip,)
end

# ensemble (multi-trajectory) MD
function train_potential_ef!(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    ref; # ::Union{GeneralInteraction, PairwiseInteraction}
    mlip::MLInteraction = ens[1].general_inters[1],
    kBT = 1.0,
    wts::Vector{<:Real} = [1000,1],
    α::Real = 1e-8,
    kwargs...
)
    train_potential_ef!(sys_train, ref, mlip, kBT=kBT, wts=wts, α=α)
    for sys in ens
        sys.general_inters = (mlip,)
    end
end


function compute_importance_weights(
    sys_train::Vector{<:System},
    ref,
    mlip::MLInteraction,
    kBT = 1.0,
)
    # compute importance weights
    p = define_gibbs_dist(ref, β=ustrip(1/kBT))
    q = define_gibbs_dist(mlip, θ=mlip.params)
    w(x) = updf(q, x) / updf(p, x)
    coords_train = [ustrip(coord[1]) for coord in get_coords(sys_train)]
    imp_wts = w.(coords_train) 
    return imp_wts / sum(imp_wts)
end