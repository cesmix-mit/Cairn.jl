import PotentialLearning: get_random_subset
export KMeans

struct KMeans <: SubsetSelector
    assign :: Vector
    c :: Vector
    batch_size :: Int
end
# function KMeans(assign::Vector, c::Vector, batch_size::Int)
#     return KMeans(assign, c, batch_size)
# end
# function KMeans(desc::Vector, k::Int, batch_size::Int)
#     A = Matrix(reduce(hcat, desc))
#     res = kmeans(A, k)
#     return KMeans(assignments(res), counts(res), batch_size)
# end
# function KMeans(dist::Union{Symmetric{T, Matrix{T}}, Matrix{T}}, k::Int, batch_size::Int) where {T}
#     res = kmedoids(dist, k)
#     return KMeans(assignments(res), counts(res), batch_size)
# end

function get_random_subset(
    km::KMeans,
)
    prop = km.c ./ length(km.assign)
    nsamp = Int.(ceil.(prop .* km.batch_size))
    indices = Int64[]
    for (i,n) in enumerate(nsamp)
        id = StatsBase.sample(findall(x -> x == i, km.assign), n, replace=false)
        append!(indices, id)
    end
    return StatsBase.sample(indices, km.batch_size, replace=false)
end
