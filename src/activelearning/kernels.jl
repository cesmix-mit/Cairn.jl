import PotentialLearning: compute_kernel, compute_gradx_kernel, compute_grady_kernel, compute_gradxy_kernel
export
    compute_kernelized_forces,
    median_kernel_bandwidth,
    min_kernel_bandwidth,
    const_kernel_bandwidth


"""
compute_kernelized_forces(sys::System, ens_fix::Vector{<:System}, knl::Kernel; feature::Union{Nothing,Feature} = nothing)

A function which computes the kernel and gradient kernel terms between a single atom in `sys` and fixed atoms in `ens_fix`, using kernel function `knl`.

"""
function compute_kernelized_forces(
    sys::System,
    ens_fix::Vector{<:System},
    knl::Kernel;
    feature::Union{Nothing,Feature} = nothing
)
    x0 = values(sys.coords)
    x_fix = get_coords(ens_fix)
    dim = length(x0[1])

    K = [compute_kernel(xi, x0, knl, feature=feature) for xi in x_fix]
    ∇K = [SVector{dim}(compute_gradx_kernel(xi, x0, knl)) for xi in x_fix]

    return K, ∇K
end

"""
compute_kernelized_forces(ens::Vector{<:System}, knl::Kernel)

A function which computes the kernel and gradient kernel terms all atoms in `ens` using kernel function `knl`.

"""
function compute_kernelized_forces(
    ens::Vector{<:System},
    knl::Kernel;
    feature::Union{Nothing,Feature} = nothing
)
    n = length(ens)
    dim = length(ens[1].coords[1])
    K = Matrix{Float64}(undef, (n,n))
    ∇K = Matrix{SVector{dim, Float64}}(undef, (n,n))

    for i = 1:n
        for j = i:n
            if i != j
                xi = ens[i].coords 
                xj = ens[j].coords 
                K[i,j] = K[j,i] = compute_kernel(xi, xj, knl, feature=feature)
                ∇Kij = SVector{dim}(compute_gradx_kernel(xi, xj, knl))
                ∇K[i,j] = ∇Kij
                ∇K[j,i] = -∇Kij
            else
                xi = ens[i].coords 
                K[i,j] = compute_kernel(xi, xi, knl, feature=feature)
                ∇K[i,j] = SVector{dim}(compute_gradx_kernel(xi, xi, knl))
            end
        end
    end

    return K, ∇K
end


function compute_kernelized_forces(
    ens::Vector{<:System},
    ens_fix::Vector{<:System},
    knl::Kernel;
    feature::Union{Nothing,Feature} = nothing
)
    n = length(ens)
    m = length(ens_fix)
    dim = length(ens[1].coords[1])
    K = Matrix{Float64}(undef, (n+m,n))
    ∇K = Matrix{SVector{dim, Float64}}(undef, (n+m,n))

    # top part of matrix: symmetric matrix with respect to ens
    Ktop, ∇Ktop = compute_kernelized_forces(ens, knl; feature=feature)
    K[1:n,1:n] = Ktop
    ∇K[1:n,1:n] = ∇Ktop

    # bottom part of matrix: compute all pairwise terms with respect to ens_fix
    for i = 1:m
        for j = 1:n
            xi = ens_fix[i].coords 
            xj = ens[j].coords 
            K[i+n,j] = compute_kernel(xi, xj, knl, feature=feature)
            ∇K[i+n,j] = SVector{dim}(compute_gradx_kernel(xi, xj, knl))
        end
    end

    return K, ∇K
end


function median_kernel_bandwidth(ens::Vector{<:System}, knl::Kernel)
    n = length(ens)
    dist = zeros(n*(n-1) ÷ 2)
    count = 1
    for i = 1:n
        for j = 1:(i-1) 
            qi = Vector(ustrip.(ens[i].coords[1])) 
            qj = Vector(ustrip.(ens[j].coords[1])) 
            dist[count] = compute_distance(qi, qj, knl.d)
            count +=1
        end
    end
    ℓ = sqrt(median(dist) / 2)
    return ℓ
end

function min_kernel_bandwidth(ens::Vector{<:System}, knl::Kernel)
    n = length(ens)
    dist = zeros(n*(n-1) ÷ 2)
    count = 1
    for i = 1:n
        for j = 1:(i-1) 
            qi = Vector(ustrip.(ens[i].coords[1])) 
            qj = Vector(ustrip.(ens[j].coords[1])) 
            dist[count] = compute_distance(qi, qj, knl.d)
            count +=1
        end
    end
    ℓ = sqrt(minimum(dist) / 2)
    return ℓ
end

const_kernel_bandwidth(ens::Vector{<:System}, knl::Kernel) = knl.ℓ



## computes the kernel with arguments with units
function compute_kernel(
    coords1::Vector{T},
    coords2::Vector{T},
    k::Kernel;
    feature::Union{Nothing,Feature} = nothing
) where T <: SVector
    if feature == nothing # single atom case: use coordinates directly
        x1 = Vector(ustrip.(coords1[1]))
        x2 = Vector(ustrip.(coords2[1]))
        
        return PotentialLearning.compute_kernel(x1, x2, k) # unitless
    # else # multi-atom case: use descriptor feature
    end
end

function compute_gradx_kernel(
    coords1::Vector{T},
    coords2::Vector{T},
    k::Kernel;
    feature::Union{Nothing,Feature} = nothing
) where T <: SVector
    if feature == nothing # single atom case: use coordinates directly
        dist_units = unit(coords1[1][1])
        x1 = Vector(ustrip.(coords1[1]))
        x2 = Vector(ustrip.(coords2[1]))
        
        return PotentialLearning.compute_gradx_kernel(x1, x2, k)
    # else # multi-atom case: use descriptor feature
    end
end

function compute_grady_kernel(
    coords1::Vector{T},
    coords2::Vector{T},
    k::Kernel;
    feature::Union{Nothing,Feature} = nothing
) where T <: SVector
    if feature == nothing # single atom case: use coordinates directly
        dist_units = unit(coords1[1][1])
        x1 = Vector(ustrip.(coords1[1]))
        x2 = Vector(ustrip.(coords2[1]))
        
        return PotentialLearning.compute_grady_kernel(x1, x2, k)
    # else # multi-atom case: use descriptor feature
    end
end

function compute_gradxy_kernel(
    coords1::Vector{T},
    coords2::Vector{T},
    k::Kernel;
    feature::Union{Nothing,Feature} = nothing
) where T <: SVector
    if feature == nothing # single atom case: use coordinates directly
        dist_units = unit(coords1[1][1])
        x1 = Vector(ustrip.(coords1[1]))
        x2 = Vector(ustrip.(coords2[1]))
        
        return PotentialLearning.compute_gradxy_kernel(x1, x2, k)
    # else # multi-atom case: use descriptor feature
    end
end

