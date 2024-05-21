import Molly: GeneralObservableLogger, log_property!

export TriggerLogger

"""
    TriggerLogger(trigger::ActiveLearningTrigger, nsteps::Int, history::Vector{T})

A logger which holds a record of evaluations of the trigger function for active learning. 

# Arguments
- `trigger::ActiveLearningTrigger`      : trigger function.
- `observable::T`                       : value of the trigger function of type `T`. 
- `n_steps::Int`                        : time step interval at which the trigger function is evaluated.
- `history::Vector{T}`                  : record of the trigger function evaluation.
"""
mutable struct TriggerLogger{A, T}
    trigger::A
    observable::T
    n_steps::Int
    history::Vector{T}
end


function TriggerLogger(trigger::ActiveLearningTrigger, T::DataType, n_steps::Integer)
    return TriggerLogger{typeof(trigger), T}(trigger, T[], n_steps, T[])
end
TriggerLogger(trigger::ActiveLearningTrigger, n_steps::Integer) = TriggerLogger(trigger, Float64, n_steps)


Base.values(logger::TriggerLogger) = logger.history


function log_property!(logger::TriggerLogger, s::System, neighbors=nothing,
    step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
    
    obs = logger.trigger.eval(s)
    logger.observable = obs

    if (step_n % logger.n_steps) == 0
        if typeof(logger.trigger) <: Union{Bool, TimeInterval} 
            return
        else
            push!(logger.history, obs)
        end
    end
end


function Base.show(io::IO, fl::TriggerLogger)
    print(io, "TriggerLogger{", eltype(fl.trigger), ", ", eltype(eltype(values(fl))), "} with n_steps ",
            fl.n_steps, ", ", length(values(fl)), " frames recorded for ",
            length(values(fl)) > 0 ? length(first(values(fl))) : "?", " atoms")
end
