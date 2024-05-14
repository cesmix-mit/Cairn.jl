export MeanKSD

"""
    MeanKSD(; thresh::Real=0.1)

An active learning trigger activated when the mean magnitude of kernel Stein discrepancy (KSD) metric falls below a threshold `thresh`.

"""
function MeanKSD(; thresh::Real=0.1)
    return UpperThreshold(meanksd, thresh)
end

meanksd(; ksd=1.0, kwargs...) = mean(norm.(get_values(ksd)))