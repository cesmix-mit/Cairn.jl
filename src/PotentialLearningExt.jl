import PotentialLearning: compute_divergence, Divergence
"""
    FisherDivergence <: Divergence
"""
struct FisherDivergence <: Divergence
    int::Integrator
end

function compute_divergence(
    p::Gibbs,
    q::Gibbs,
    d::FisherDivergence,
)
    return fisher_divergence(p, q, d.int)
end

function fisher_divergence(
    p::Gibbs,
    q::Gibbs,
    int::QuadIntegrator,
)
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)

    Zp = normconst(p, int)
    h(x) = updf(p, x)/Zp .* norm(sp(x) - sq(x))^2
    return sum(int.w .* h.(int.ξ))
end 

function fisher_divergence(
    p::Gibbs,
    q::Gibbs,
    int::MCMC,
)
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)

    xsamp = rand(p, int.n, int.sampler, int.ρ0) 
    h = x -> norm(sp(x) - sq(x))^2
    return sum(h.(xsamp)) / length(xsamp)
end 

function fisher_divergence(
    p::Gibbs,
    q::Gibbs,
    int::MCSamples,
)
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)

    h = x -> norm(sp(x) - sq(x))^2
    return sum(h.(int.xsamp)) / length(int.xsamp)
end 


export FisherDivergence, fisher_divergence