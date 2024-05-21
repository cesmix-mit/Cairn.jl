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


function meanksd(ens::Vector{<:System})
    ksd = [sys.data["ksd"] for sys in ens]
    return mean(norm.(get_values(ksd)))
end


function trigger_activated(
    sys::Vector{<:System},
    trigger::MeanKSD;
    kwargs...
)
    return trigger.eval(sys) < trigger.thresh
end

