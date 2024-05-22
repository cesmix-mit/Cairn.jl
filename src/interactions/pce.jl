# polynomial chaos expansion
import AtomsCalculators: potential_energy, forces
import PotentialLearning: compute_local_descriptors, compute_force_descriptors
export
    PolynomialChaos,
    eval_basis,
    eval_grad_basis

@doc raw"""
    PolynomialChaos{P, F, E} <: MLInteraction

Total-degree polynomial chaos expansion (PCE) model defining the potential energy function for a single-atom system.

# Arguments
- `p :: Integer`                : total polynomial degree
- `d :: Integer`                : dimension
- `BasisFamily :: P`            : family of basis functions from SpecialPolynomials
- `basis :: Vector{Vector}`     : set of basis functions
- `gbasis :: Vector{Vector}`    : set of gradient basis functions
- `force_units :: F`            : units of force
- `energy_units :: E`           : units of energy


"""
mutable struct PolynomialChaos{P, F, E} <: MLInteraction
    p :: Int
    d :: Int
    params :: Union{Nothing,Vector{<:Real}}
    BasisFamily :: P
    basis :: Vector{Vector}
    gbasis :: Vector{Vector}
    xscl :: Vector{<:Union{Real, Vector}}
    force_units :: F
    energy_units :: E
end

# outer constructor
function PolynomialChaos(
    p::Int,
    d::Int,
    BasisFamily;
    params::Union{Nothing,Vector{<:Real}} = nothing,
    xscl = nothing,
    force_units = u"kJ * mol^-1 * nm^-1",
    energy_units = u"kJ * mol^-1",
)
    bas = construct_basis(p, d, BasisFamily)
    gbas = construct_grad_basis(p, d, BasisFamily)
    if xscl == nothing && d > 1
        xscl = domain(bas[1][1])
    elseif xscl == nothing && d == 1
        xscl = domain(bas[1])
    end

    return PolynomialChaos{Type{BasisFamily}, typeof(force_units), typeof(energy_units)}(
        p, d, params, BasisFamily,
        bas, gbas, xscl,
        force_units, energy_units,
    )    

end


## functions for computing potential energy and forces
@inline function potential_energy(sys, inter::PolynomialChaos; neighbors=nothing,
    n_threads::Integer=Threads.nthreads())
    e_descr = try
        sys.data["energy_descriptors"]
    catch
        compute_local_descriptors(sys, inter)
    end
    if length(e_descr) == 0 # at initialization
        e_descr = compute_local_descriptors(sys, inter)
    end
    return sum(potential_pce.(Ref(inter), sys.coords, e_descr))
end
@inline function potential_pce(inter::PolynomialChaos, coord::SVector{2},
    descr::Vector=eval_basis(Vector(get_values(coord)), inter))
    energy = dot(inter.params, descr)
    return energy .* inter.energy_units
end

@inline function forces(sys, inter::PolynomialChaos; neighbors=nothing,
    n_threads::Integer=Threads.nthreads())
    f_descr = try
        sys.data["force_descriptors"]
    catch
        compute_force_descriptors(sys, inter)
    end
    if length(f_descr) == 0 # at initialization
        f_descr = compute_force_descriptors(sys, inter)
    end
    return force_pce.(Ref(inter), sys.coords, f_descr)
end
@inline function force_pce(inter::PolynomialChaos, coord::SVector{2},
    descr::Vector=eval_grad_basis(Vector(get_values(coord)), inter))
    forces = -dot.((inter.params,), descr)
    return SVector{2}(forces .* inter.force_units)
end



## functions for constructing the polynomial basis
function construct_basis(p::Int, d::Int, BasisFamily)
    Mset = TotalDegreeMset(p, d)
    N = length(Mset)
    bas = Vector{Vector}(undef, N)
    for n = 1:N
        m = Mset[n]
        bas[n] = [basis(BasisFamily, m[j]) for j = 1:d]
    end
    return bas
end
function construct_grad_basis(p::Int, d::Int, BasisFamily)
    basis = construct_basis(p, d, BasisFamily)
    N = length(basis)

    gbas = [deepcopy(basis) for j = 1:d]
    for j = 1:d
        for n = 1:N
            gbas[j][n][j] = SpecialPolynomials.derivative(basis[n][j])
        end
    end

    return gbas
end

function eval_basis(x::Vector, bas::Vector)
    N = length(bas)
    Φ = Vector{Float64}(undef, N)
    d = length(bas[1])
    for n = 1:N
        ϕn = prod([bas[n][j](x[j]) for j = 1:d])
        # ϕn_norm = prod([factorial(m[j]) for j = 1:d])
        Φ[n] = ϕn # / ϕn_norm # make orthonormal
    end
    return Φ
end
function eval_basis(x::Vector, inter::PolynomialChaos)
    xs = rescale(x, inter.xscl, domain(inter))
    return eval_basis(xs, inter.basis)
end

function eval_grad_basis(x::Vector, gbas::Vector)
    d = length(gbas)
    gΦ = Vector{Vector{Float64}}(undef, d)
    for j = 1:d
        gΦ[j] = eval_basis(x, gbas[j])
    end
    return gΦ
end
function eval_grad_basis(x::Vector, inter::PolynomialChaos)
    xs = rescale(x, inter.xscl, domain(inter))
    return eval_grad_basis(xs, inter.gbasis)
end

function compute_local_descriptors(
    sys,
    inter::PolynomialChaos=sys.general_inters[1],
)
    x = Vector.(get_values.(sys.coords))
    edescr = eval_basis.(x, Ref(inter))
    try sys.data["energy_descriptors"] = edescr catch; end
    return edescr
end
function compute_force_descriptors(
    sys,
    inter::PolynomialChaos=sys.general_inters[1],
)
    x = Vector.(get_values.(sys.coords))
    fdescr = eval_grad_basis.(x, Ref(inter))
    try sys.data["force_descriptors"] = fdescr catch; end
    return fdescr
end


## functions for constructing total-order multi-index sets
"""
`TotalDegreeMset(p::Int, d::Int)`

Return a matrix of indices, where the columns are the multi-indices in dimension d with the total degree less than or equal to p.

# Arguments
- `p :: Int`        : The total degree of the polynomials
- `d :: Int`        : The dimension of the problem
"""
function TotalDegreeMset(p::Int, d::Int)
    # Initialize the multi-index set
    sz = TotalMsetSize(p, d)
    Mset = zeros(Int, d, sz)
    # Call the recursive helper function
    TotalDegreeMsetHelper!(Mset, p)
    return [Mset[:,i] for i = 1:size(Mset,2)] # Mset'
end

# Calculate the size of a total order multi-index set of order p in d dimensions
function TotalMsetSize(p::Int, d::Int)
    sum(binomial(i+d-1, i) for i in 0:p; init=0)
end

# Recursively construct the total order multi-index set
function TotalDegreeMsetHelper!(A, p)
    # Find dimension of problem
    d = size(A, 1)
    # If there are no more degrees of freedom, everything is zero
    if p == 0
        fill!(A, 0)
        return
    end
    # If there is only one dimension, then we can store everything from 0 to p
    if d == 1
        A' .= 0:p
        return
    end
    # Initialize loop vars
    col_st = 0
    col_end = 0
    # Loop over all possible numbers we can put at the top of this view
    # Start with p, so everything is zero, then p-1, etc.
    for p_top in p:-1:0
        # The start of this "chunk" is after the end of the last one
        col_st = col_end + 1
        # The end of this "chunk" is the size of the subproblem,
        # where the limit is p-p_top (what's left after putting p_top at the top)
        # and the dimension is d-1
        col_end += TotalMsetSize(p-p_top, d-1)
        # Fill in the top row with p_top
        A[1, col_st:col_end] .= p_top
        # Fill in the rest of the matrix with the subproblem
        TotalDegreeMsetHelper!(@view(A[2:d, col_st:col_end]), p-p_top)
    end
end



function assign_params(
    pce::PolynomialChaos,
    params::Vector,
)
    return pce.params = params
end

domain(bas::Legendre) = [-1,1]
domain(bas::ChebyshevU) = [-1,1]
domain(bas::Laguerre) = [0,20]
domain(bas::Hermite) = [-5,5]
domain(bas::Gegenbauer) = [-1,1]
domain(bas::Jacobi) = [-1,1]
function domain(pce::PolynomialChaos)
    if pce.d > 1
        return domain(pce.basis[1][1])
    elseif pce.d == 1
        return domain(pce.basis[1])
    end
end

# same scale for each element of x
function rescale(x, scl_old::Vector{<:Real}, scl_new::Vector{<:Real})
    xmin, xmax = scl_old
    a, b = scl_new
    z = [(b-a)*(xi - xmin) / (xmax - xmin) + a for xi in x]
    return z
end
# differential scales for each (input) element of x
function rescale(x, scl_old::Vector{<:Vector}, scl_new::Vector{<:Real})
    a, b = scl_new
    N = length(x)
    z = zeros(N)
    for n = 1:N
        xmin, xmax = scl_old[n]
        z[n] = (b-a)*(x[n] - xmin) / (xmax - xmin) + a
    end
    return z
end
# differential scales for each (output) element of x
function rescale(x, scl_old::Vector{<:Real}, scl_new::Vector{<:Vector})
    xmin, xmax = scl_old
    N = length(x)
    z = zeros(N)
    for n = 1:N
        a, b = scl_new[n]
        z[n] = (b-a)*(x[n] - xmin) / (xmax - xmin) + a
    end
    return z
end
# scale for each element of x
function rescale(x, scl_old::Vector{<:Vector}, scl_new::Vector{<:Vector})
    N = length(x)
    z = zeros(N)
    for n = 1:N
        xmin, xmax = scl_old[n]
        a, b = scl_new[n]
        z[n] = (b-a)*(x[n] - xmin) / (xmax - xmin) + a
    end
    return z
end

