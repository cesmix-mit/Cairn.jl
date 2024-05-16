export MaxKernelEval



"""
    MaxKernelEval(; thresh::Real=0.1)

An active learning trigger activated when the maximum kernel evaluation falls below a threshold `thresh`.

"""
struct MaxKernelEval <: ActiveLearningTrigger
    eval::Function
    thresh::Real
end
function MaxKernelEval(; thresh::Real=0.1)
    return MaxKernelEval(maxkerneval, thresh)
end

maxkerneval(; kernel) = Base.maximum(kernel)


function trigger_activated(
    sys,
    trigger::MaxKernelEval;
    kwargs...
)
    return trigger.eval(; kernel=sys.kernel) < trigger.thresh
end