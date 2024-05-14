export train!

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