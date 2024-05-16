export MeanKSD



"""
    MeanKSD(; thresh::Real=0.1)

An active learning trigger activated when the mean magnitude of kernel Stein discrepancy (KSD) metric falls below a threshold `thresh`.

"""
struct MeanKSD <: ActiveLearningTrigger
    eval::Function
    thresh::Real
end
function MeanKSD(; thresh::Real=0.1)
    return MeanKSD(meanksd, thresh)
end


meanksd(; ksd) = mean(norm.(get_values(ksd)))


function trigger_activated(
    sys::Vector{<:System},
    trigger::MeanKSD;
    kwargs...
)
    return trigger.eval(sys) < trigger.thresh
end

