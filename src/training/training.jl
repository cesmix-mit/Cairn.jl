export train!, compute_importance_weights


function train!(
    sys::System,
    sys_train::Vector{<:System},
    ref,
    args...;
    mlip::MLInteraction = sys.general_inters[1],
    e_flag = true,
    f_flag = true
)
    lp = learn!(sys_train, ref, mlip, args...; e_flag=e_flag, f_flag=f_flag)
    mlip.params = lp.β
    sys.general_inters = (mlip,)
end


function train!(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    ref,
    args...;
    mlip::MLInteraction = ens[1].general_inters[1],
    e_flag = true,
    f_flag = true
)
    lp = learn!(sys_train, ref, mlip, args...; e_flag=e_flag, f_flag=f_flag)
    mlip.params = lp.β
    for sys in ens
        sys.general_inters = (mlip,)
    end
end


include("linear-learn.jl")




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