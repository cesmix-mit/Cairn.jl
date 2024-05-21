export
    ActiveLearningTrigger,
    trigger_activated

"""
Abstract type for defining criteria triggering the active learning step during simulation.

"""
abstract type ActiveLearningTrigger end



"""
    trigger_activated(trigger::ActiveLearningTrigger, kwargs...)
    trigger_activated(trigger::Bool, kwargs...)

A function which returns a Bool of whether or not the trigger for active learning is activated.
"""
trigger_activated(sys, trigger::Bool; kwargs...) = trigger


## evaluate multiple triggers in order
function trigger_activated(
    sys::Union{System, Vector{<:System}},
    triggers::Tuple{<:ActiveLearningTrigger};
    sys_train=nothing,
    step_n::Integer=1,
)
    for trigger in triggers
        if trigger_activated(sys, trigger; sys_train=sys_train, step_n=step_n)
            return true
        end
    end
end


include("timeinterval.jl")
include("maxkernel.jl")
include("meanksd.jl")
include("maxvol.jl")