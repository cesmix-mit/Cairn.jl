export MaxVol

struct MaxVol <: ActiveLearningTrigger
    feature_func::Function
    eval::Function
    thresh::Real
end

"""
    MaxVol(feature_func; thresh::Real=1.0)

An active learning trigger activated after the D-optimality based extrapolation grade exceeds a threshold `thresh`.

"""
function MaxVol(feature_func; thresh::Real=1.0)
    return MaxVol(feature_func, extrap_grade, thresh)
end

function extrap_grade(
    trigger::MaxVol;
    ens_old::Vector{<:System},
    ens_new::Vector{<:System},
    kwargs...
)
    N = length(ens_old)
    ens_all = reduce(vcat, [ens_old, ens_new])
    A = Matrix(trigger.feature_func(ens_old))
    A_all = Matrix(trigger.feature_func(ens_all))
    rows, _ = maxvol!(A)
    γ = maximum(abs.(A_all*A[rows,:]), dims=2)
    γ_new = γ[N+1:end]
    return γ_new
end

# function extrap_grade(
#     trigger::MaxVol;
#     ens_old::Vector{<:System},
#     ens_new::System,
#     kwargs...
# )
#     N = length(ens_old)
#     ens_all = reduce(vcat, [ens_old, ens_new])
#     A = Matrix(trigger.feature_func(ens_old))
#     A_all = Matrix(trigger.feature_func(ens_all))
#     rows, _ = maxvol!(A)
#     γ = maximum(abs.(A_all*A[rows,:]), dims=2)
#     return γ[end]
# end


function trigger_activated(
    trigger::MaxVol;
    ens_old::Vector{<:System},
    ens_new::Vector{<:System},
    kwargs...
)
    return maximum(trigger.eval(trigger; ens_old=ens_old, ens_new=ens_new)) > trigger.thresh
end

# function trigger_activated(
#     trigger::MaxVol;
#     ens_old::Vector{<:System},
#     ens_new::System,
#     kwargs...
# )
#     return maximum(trigger.eval(trigger; ens_old=ens_old, ens_new=ens_new)) > trigger.thresh
# end
