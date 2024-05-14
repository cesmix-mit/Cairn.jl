# polynomial chaos expansion
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
@inline function potential_energy(inter::PolynomialChaos, sys, neighbors=nothing;
    n_threads::Integer=Threads.nthreads())
    return sum(potential_pce.(Ref(inter), sys.coords))
end

@inline function potential_pce(inter::PolynomialChaos, coord::SVector{2})
    coord = Vector(get_values(coord)) # single-atom
    energy = dot(inter.params, eval_basis(coord, inter))
    return energy .* inter.energy_units
end

@inline function forces(inter::PolynomialChaos, sys, neighbors=nothing;
    n_threads::Integer=Threads.nthreads())
    return force_pce.(Ref(inter), sys.coords)
end
@inline function force_pce(inter::PolynomialChaos, coord::SVector{2})
    coord = Vector(get_values(coord))
    forces = -dot.((inter.params,), eval_grad_basis(coord, inter))
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


function eval_basis(x, inter::PolynomialChaos)
    xs = rescale(x, inter.xscl, domain(inter))
    return eval_basis(xs, inter.basis)
end

function eval_basis(x, bas::Vector)
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



function eval_grad_basis(x, inter::PolynomialChaos)
    xs = rescale(x, inter.xscl, domain(inter))
    return eval_grad_basis(xs, inter.gbasis)
end

function eval_grad_basis(x, gbas::Vector)
    d = length(gbas)
    gΦ = Vector{Vector}(undef, d)
    for j = 1:d
        gΦ[j] = eval_basis(x, gbas[j])
    end
    return gΦ
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


## function for training ML potential
function train_potential_e!(
    sys_train::Vector{<:System},
    ref, # ::Union{GeneralInteraction, PairwiseInteraction}
    pce::PolynomialChaos;
    wts::Vector{<:Real} = ones(length(sys_train)),
    kBT = 1.0,
    α::Real = 1e-8,
)
    β = 1/kBT
    coords = get_coords(sys_train)
    xtrain = [ustrip.(coord[1]) for coord in coords]
    b = [β * ustrip(potential_energy(ref, sys_i)) for sys_i in sys_train]
    A = reduce(hcat, eval_basis.(xtrain, (pce,)))'
    W = Diagonal(wts)
    params_fit = pinv(A' * W * A, α) * (A' * W * b)

    pce.params = params_fit

    return pce
end

function train_potential_f!(
    sys_train::Vector{<:System},
    ref, # ::Union{GeneralInteraction, PairwiseInteraction}
    pce::PolynomialChaos;
    wts::Vector{<:Real} = ones(length(sys_train)),
    kBT = 1.0,
    α::Real = 1e-8,
)
    β = 1/kBT
    coords = get_coords(sys_train)
    xtrain = [ustrip.(coord[1]) for coord in coords]
    b = reduce(vcat, β .* [reduce(vcat, ustrip.(forces(ref, sys_i))) for sys_i in sys_train])
    A = reduce(vcat, reduce(hcat, eval_grad_basis.(xtrain, (pce,)))')
    
    W = Diagonal(reduce(vcat, [w*ones(pce.d) for w in wts]))
    params_fit = pinv(A' * W * A, α) * (A' * W * b)

    pce.params = params_fit

    return pce
end

function train_potential_ef!(
    sys_train::Vector{<:System},
    ref, # ::Union{GeneralInteraction, PairwiseInteraction}
    pce::PolynomialChaos,
    wts::Vector{<:Vector};
    kBT = 1.0,
    α::Real = 1e-8,
)
    β = 1/kBT
    coords = get_coords(sys_train)
    xtrain = [ustrip.(coord[1]) for coord in coords]
    N = length(xtrain)

    b_e = [β * ustrip(potential_energy(ref, sys_i)) for sys_i in sys_train]
    b_f = reduce(vcat, β .* [reduce(vcat, ustrip.(forces(ref, sys_i))) for sys_i in sys_train])
    b = reduce(vcat, [b_e, b_f])

    A_e = reduce(hcat, eval_basis.(xtrain, (pce,)))'
    force_descr = eval_grad_basis.(xtrain, (pce,))
    force_descr = [reduce(hcat, fd) for fd in force_descr]
    A_f = reduce(hcat, force_descr)'
    A = reduce(vcat, [A_e, A_f])

    W_e = wts[1]
    W_f = reduce(vcat, [wts[2][i]*ones(pce.d) for i = 1:N])
    W = Diagonal(reduce(vcat, [W_e, W_f]))
    params_fit = pinv(A' * W * A, α) * (A' * W * b)

    pce.params = params_fit

    return pce
end

function train_potential_ef!(
    sys_train::Vector{<:System},
    ref, # ::Union{GeneralInteraction, PairwiseInteraction}
    pce::PolynomialChaos;
    wts::Vector{<:Real} = [100,1],
    kBT = 1.0,
    α::Real = 1e-8,
)
    N = length(sys_train)
    wt_vec = [wts[1]*ones(N), wts[2]*ones(N)]

    return train_potential_ef!(sys_train, ref, pce, wt_vec, kBT=kBT, α=α)
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
# differential scales for each element of x
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

