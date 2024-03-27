import PotentialLearning: get_values

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