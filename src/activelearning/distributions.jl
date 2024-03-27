export
    define_gibbs_dist,
    compute_fisher_div,
    compute_rmse

# potential and gradient functions for PolynomialChaos
function V(x, θ, inter::PolynomialChaos; dist_units = u"nm")
    inter.params = θ
    coord = SVector{2}(x) .* dist_units
    return ustrip(SteinMD.potential_pce(inter, coord))
end
function ∇xV(x, θ, inter::PolynomialChaos; dist_units = u"nm")
    inter.params = θ
    coord = SVector{2}(x) .* dist_units
    return Vector(ustrip.(SteinMD.force_pce(inter, coord)))
end
∇θV(x, θ, pce::PolynomialChaos; dist_units = u"nm") = eval_basis(x, pce)

# potential and gradient functions for MullerBrown
function V(x, θ, inter::MullerBrownRot; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return ustrip(Molly.potential_muller_brown(inter, coord))
end

function ∇xV(x, θ, inter::MullerBrownRot; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return Vector(ustrip.(Molly.force_muller_brown(inter, coord)))
end

∇θV(x, θ, inter::MullerBrownRot; dist_units = u"nm") = 0

# potential and gradient functions for Himmelblau
function V(x, θ, inter::Himmelblau; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return ustrip(SteinMD.potential_himmelblau(inter, coord))
end

function ∇xV(x, θ, inter::Himmelblau; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return Vector(ustrip.(SteinMD.force_himmelblau(inter, coord)))
end

∇θV(x, θ, inter::Himmelblau; dist_units = u"nm") = 0

# potential and gradient functions for DoubleWell
function V(x, θ, inter::DoubleWell; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return ustrip(SteinMD.potential_double_well(inter, coord))
end

function ∇xV(x, θ, inter::DoubleWell; dist_units = u"nm")
    coord = SVector{2}(x) .* dist_units
    return Vector(ustrip.(SteinMD.force_double_well(inter, coord)))
end

∇θV(x, θ, inter::DoubleWell; dist_units = u"nm") = 0


# function which defines Gibbs object
function define_gibbs_dist(
    inter;
    β::Real = 1.0,
    θ::Union{Real, Vector{<:Real}, Nothing} = [1],
)
    V0 = (x, θ) -> V(x, θ, inter)
    ∇xV0 = (x, θ) -> ∇xV(x, θ, inter)
    ∇θV0 = (x, θ) -> ∇θV(x, θ, inter)

    return Gibbs(V=V0, ∇xV=∇xV0, ∇θV=∇θV0, β=β, θ=θ)
end

function compute_fisher_div(ref, mlip, integrator)
    p = define_gibbs_dist(ref)
    q = define_gibbs_dist(mlip, θ=mlip.params)
    f = FisherDivergence(integrator)
    return compute_discrepancy(p, q, f)
end

function compute_rmse(ref, mlip, integrator)
    p = define_gibbs_dist(ref)
    q = define_gibbs_dist(mlip, θ=mlip.params)

    x_eval = integrator.ξ
    e_true = p.V.(x_eval)
    e_pred = q.V.(x_eval)
    f_true = reduce(vcat, p.∇xV.(x_eval))
    f_pred = reduce(vcat, q.∇xV.(x_eval))
    
    # Zp = normconst(p, integrator)
    # prob(x) = updf(p, x) / Zp
    # wts = prob.(x_eval) ./ length(x_eval)

    rmse_e = rmse(e_pred, e_true) # wts' * 
    rmse_f = rmse(f_pred, f_true) # wts' * 

    return rmse_e, rmse_f
end