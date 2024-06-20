export KMeans

struct KMeans <: SubsetSelector
    desc :: Vector
    k :: Int
    batch_size :: Int
end


function get_random_subset(
    km::KMeans,
)
    A = Matrix(reduce(hcat, km.desc))
    res = kmeans(A, km.k)

    a = assignments(res) # get the assignments of points to clusters
    c = counts(res) # get the cluster sizes

    prop = c ./ length(e_desc)
    nsamp = Int.(ceil.(prop .* km.batch_size))
    indices = Int64[]
    for (i,n) in enumerate(nsamp)
        id = StatsBase.sample(findall(x -> x == i, a), n, replace=false)
        append!(indices, id)
    end
    return indices
end
