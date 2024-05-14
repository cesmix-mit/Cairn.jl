import PotentialLearning: learn!, LinearProblem, UnivariateLinearProblem, CovariateLinearProblem

function LinearProblem(
    sys_train::Vector{<:System},
    ref,
    mlip::PolynomialChaos;
    e_flag = true,
    f_flag = true
)

    coords = get_coords(sys_train)
    xtrain = [ustrip.(coord[1]) for coord in coords]

    if e_flag
        descriptors = eval_basis.(xtrain, (mlip,))
        energies = [ustrip(potential_energy(ref, sys)) for sys in sys_train]
    end
    if f_flag 
        force_descriptors = eval_grad_basis.(xtrain, (mlip,))
        force = [ustrip.(Molly.forces(ref, sys)) for sys in sys_train]
    end


    if e_flag & ~f_flag
        dim = length(descriptors[1])
        β = zeros(dim)
        β0 = zeros(1)

        p = UnivariateLinearProblem(
            descriptors,
            energies,
            β,
            β0,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    elseif ~e_flag & f_flag
        dim = length(force_descriptors[1][1])
        β = zeros(dim)
        β0 = zeros(1)

        force = [reduce(vcat, fi) for fi in force]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(
            force_descriptors,
            force,
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
        force = [reduce(vcat, fi) for fi in force]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = CovariateLinearProblem(
            energies,
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



