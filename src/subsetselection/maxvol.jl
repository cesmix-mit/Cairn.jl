export MaxVolSubset

struct MaxVolSubset <: SubsetSelector
    desc::Vector
end


function get_subset(
    mv::MaxVolSubset,
)
    A = Matrix(reduce(hcat, mv.desc)')

    # select D-optimal subset using MaxVol
    # need long rectangular matrix, e. g. nrows(A) > ncol(A)
    indices, _ = maxvol!(A)

    return indices
end
