import PotentialLearning: get_values

include("triggerlogger.jl")
include("stepcomponentlogger.jl")

"""
    get_values(logger::Logger)
    get_values(qt::Vector)
    get_values(qt::Real)

Returns a Vector of unitless values of the logger history, vector, or scalar quantity. 
"""
function get_values(logger::GeneralObservableLogger)
    return [ustrip.(values(logger)[i][1]) for i = 1:length(values(logger))]
end

function get_values(logger::StepComponentLogger)
    return [ustrip.(values(logger)[i][1]) for i = 2:length(values(logger))]
end

function get_values(logger::TriggerLogger)
    return [ustrip.(values(logger)[i][1]) for i = 2:length(values(logger))]
end

function get_values(qt::Vector)
    return [ustrip.(values(qt)[i]) for i in 1:length(values(qt))]
end

function get_values(qt::Real)
    return ustrip(qt)
end



"""
    has_step_property(sys::System)
    has_step_property(logger::Logger)

Returns a Bool evaluating whether the system `sys` contains any loggers which are of type `StepComponentLogger`.
"""
function has_step_property(sys::System)
    return any([typeof(logger) <: StepComponentLogger for logger in sys.loggers])
end

function has_step_property(loggers::NamedTuple)
    return any([typeof(logger) <: StepComponentLogger for logger in loggers])
end

has_step_property(logger::StepComponentLogger) = true
has_step_property(logger::TriggerLogger) = false
has_step_property(logger::GeneralObservableLogger) = false



