export
    ActiveLearningTrigger,
    UpperThreshold,
    LowerThreshold,
    TimeInterval,
    MaxKernelEval,
    MaxVol,
    MeanKSD,
    trigger_activated

"""
Abstract type for defining criteria triggering the active learning step during simulation.

"""
abstract type ActiveLearningTrigger end


struct UpperThreshold <: ActiveLearningTrigger
    eval::Function
    thresh::Real
end

struct LowerThreshold <: ActiveLearningTrigger
    eval::Function
    thresh::Real
end

struct TimeInterval <: ActiveLearningTrigger
    interval::Real
end

struct MaxVol <: ActiveLearningTrigger
    feature_func::Function
    eval::Function
    thresh::Real
end


"""
    MaxKernelEval(; thresh::Real=0.1)

An active learning trigger activated when the maximum kernel evaluation falls below a threshold `thresh`.

"""
function MaxKernelEval(; thresh::Real=0.1)
    return UpperThreshold(maxkerneval, thresh)
end

maxkerneval(; kernel=1.0, kwargs...) = Base.maximum(kernel)

"""
    MeanKSD(; thresh::Real=0.1)

An active learning trigger activated when the mean magnitude of kernel Stein discrepancy (KSD) metric falls below a threshold `thresh`.

"""
function MeanKSD(; thresh::Real=0.1)
    return UpperThreshold(meanksd, thresh)
end

meanksd(; ksd=1.0, kwargs...) = mean(norm.(get_values(ksd)))

"""
    TimeInterval(; interval::Real=1)

An active learning trigger activated after a fixed number of simulation steps specified by `interval`.

"""
function TimeInterval(; interval::Real=1)
    return TimeInterval(interval)
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


"""
    trigger_activated(trigger::ActiveLearningTrigger, kwargs...)
    trigger_activated(trigger::Bool, kwargs...)

A function which returns a Bool of whether or not the trigger for active learning is activated.
"""
function trigger_activated(
    trigger::UpperThreshold;
    kwargs...
)
    return trigger.eval(; kwargs...) < trigger.thresh
end

function trigger_activated(
    trigger::LowerThreshold;
    kwargs...
)
    return trigger.eval(; kwargs...) > trigger.thresh
end

function trigger_activated(
    trigger::TimeInterval;
    step_n::Integer = 1,
    kwargs...
)
    return (step_n % trigger.interval) == 0
end

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

trigger_activated(trigger::Bool; kwargs...) = trigger