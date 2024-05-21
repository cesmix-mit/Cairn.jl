export MaxVol

struct MaxVol <: ActiveLearningTrigger
    eval::Function
    thresh::Real
end

"""
    MaxVol(feature_func; thresh::Real=1.0)

An active learning trigger activated after the D-optimality based extrapolation grade exceeds a threshold `thresh`.

"""
function MaxVol(; thresh::Real=1.0)
    return MaxVol(extrap_grade, thresh)
end

# extrapolation grade from single trajectory
function extrap_grade(
    sys::System;
    sys_train::Vector{<:System},
    kwargs...
)
    A = Matrix(reduce(hcat, sum.(get_local_descriptors(sys_train)))')
    B = sum(get_local_descriptors(sys))

    # select D-optimal subset using MaxVol
    # need long rectangular matrix, e. g. nrows(A) > ncol(A)
    rows, _ = maxvol!(A)

    # compute extrapolation grade
    γ = maximum(abs.(B*pinv(A[rows,:])), dims=2)
    return γ
end


# max extrapolation grade among ensemble trajectories
function extrap_grade(
    ens::Vector{<:System};
    sys_train::Vector{<:System},
    kwargs...
)
    A = Matrix(reduce(hcat, sum.(get_local_descriptors(sys_train)))')
    B = Matrix(reduce(hcat, sum.(get_local_descriptors(ens)))')

    # select D-optimal subset using MaxVol
    # need long rectangular matrix, e. g. nrows(A) > ncol(A)
    rows, _ = maxvol!(A)

    # compute extrapolation grade
    γ = maximum(abs.(B*pinv(A[rows,:])), dims=2)
    return maximum(γ)
end


function trigger_activated(
    sys,
    trigger::MaxVol;
    sys_train::Vector{<:System},
    kwargs...
)
    return trigger.eval(sys; sys_train=sys_train) > trigger.thresh
end

