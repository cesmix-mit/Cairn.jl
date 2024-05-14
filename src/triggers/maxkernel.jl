export MaxKernelEval

"""
    MaxKernelEval(; thresh::Real=0.1)

An active learning trigger activated when the maximum kernel evaluation falls below a threshold `thresh`.

"""
function MaxKernelEval(; thresh::Real=0.1)
    return UpperThreshold(maxkerneval, thresh)
end

maxkerneval(; kernel=1.0, kwargs...) = Base.maximum(kernel)
