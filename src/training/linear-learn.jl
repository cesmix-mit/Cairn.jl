import PotentialLearning: learn!, LinearProblem, UnivariateLinearProblem, CovariateLinearProblem

function LinearProblem(
    sys_train::Vector{<:System},
    ref,
    mlip::PolynomialChaos;
    e_flag = true,
    f_flag = true
)

    if e_flag
        # TODO: try find_descriptors catch; end
        descriptors = sum.(compute_local_descriptors.(sys_train, Ref(mlip)))
        energies = AtomsCalculators.potential_energy.(sys_train, Ref(ref))
    end
    if f_flag 
        force_descriptors = reduce(vcat, compute_force_descriptors.(sys_train, Ref(mlip)))
        force = AtomsCalculators.forces.(sys_train, Ref(ref))
    end


    if e_flag & ~f_flag
        dim = length(descriptors[1])
        β = zeros(dim)
        β0 = zeros(1)

        p = UnivariateLinearProblem(
            descriptors,
            ustrip.(energies),
            β,
            β0,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    elseif ~e_flag & f_flag
        dim = length(force_descriptors[1][1])
        β = zeros(dim)
        β0 = zeros(1)

        force = [reduce(vcat, ustrip.(fi)) for fi in force]

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(
            force_descriptors,
            reduce(vcat, force),
            β,
            β0,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    elseif e_flag & f_flag
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])
        if (dim_d != dim_fd)
            error("Descriptors and Force Descriptors have different dimension!")
        else
            dim = dim_d
        end

        β = zeros(dim)
        β0 = zeros(1)
        force = [reduce(vcat, ustrip.(fi)) for fi in force]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = CovariateLinearProblem(
            ustrip.(energies),
            force,
            descriptors,
            force_descriptors,
            β,
            β0,
            [1.0],
            [1.0],
            Symmetric(zeros(dim, dim)),
        )
    end
    p
end


function learn!(
    sys_train::Vector{<:System},
    ref,
    mlip,
    args...;
    e_flag = true,
    f_flag = true
)
    lp = LinearProblem(
        sys_train,
        ref,
        mlip;
        e_flag = e_flag,
        f_flag = f_flag
    )

    learn!(lp, args...)

    return lp
end



