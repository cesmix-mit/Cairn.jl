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
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)

    if typeof(d.int) <: QuadIntegrator
        Zp = normconst(p, d.int)
        h(x) = updf(p, x)/Zp .* norm(sp(x) - sq(x))^2
        return sum(d.int.w .* h.(d.int.ξ))
    
    elseif typeof(d.int) <: MCMC
        xsamp = rand(p, d.int.n, d.int.sampler, d.int.ρ0) 
        h = x -> norm(sp(x) - sq(x))^2
        return sum(h.(xsamp)) / length(xsamp)

    elseif typeof(d.int) <: MCSamples
        h = x -> norm(sp(x) - sq(x))^2
        return sum(h.(d.int.xsamp)) / length(d.int.xsamp)
        
    end
end

export FisherDivergence