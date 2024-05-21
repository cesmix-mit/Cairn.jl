export 
    compute_importance_weights


function compute_importance_weights(
    sys_train::Vector{<:System},
    ref,
    mlip::MLInteraction,
    kBT = 1.0,
)
    # compute importance weights
    p = define_gibbs_dist(ref, β=ustrip(1/kBT))
    q = define_gibbs_dist(mlip, θ=mlip.params)
    w(x) = updf(q, x) / updf(p, x)
    coords_train = [ustrip(coord[1]) for coord in get_coords(sys_train)]
    imp_wts = w.(coords_train) 
    return imp_wts / sum(imp_wts)
end