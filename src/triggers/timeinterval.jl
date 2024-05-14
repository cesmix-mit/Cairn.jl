export TimeInterval

struct TimeInterval <: ActiveLearningTrigger
    interval::Real
end


"""
    TimeInterval(; interval::Real=1)

An active learning trigger activated after a fixed number of simulation steps specified by `interval`.

"""
function TimeInterval(; interval::Real=1)
    return TimeInterval(interval)
end


function trigger_activated(
    trigger::TimeInterval;
    step_n::Integer = 1,
    kwargs...
)
    return (step_n % trigger.interval) == 0
end