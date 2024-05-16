export TimeInterval


"""
    TimeInterval(; interval::Real=1)

An active learning trigger activated after a fixed number of simulation steps specified by `interval`.

"""
struct TimeInterval <: ActiveLearningTrigger
    eval::Function
    interval::Real
end
function TimeInterval(; interval::Real=1)
    return TimeInterval(timeint, interval)
end


timeint(; step_n, interval) = (step_n % interval)


function trigger_activated(
    sys,
    trigger::TimeInterval;
    step_n::Integer,
    kwargs...
)
    return trigger.eval(; step_n=step_n, interval=trigger.interval) == 0
end

