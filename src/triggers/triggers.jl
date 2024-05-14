export
    ActiveLearningTrigger,
    UpperThreshold,
    LowerThreshold,
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


trigger_activated(trigger::Bool; kwargs...) = trigger