export plot_contours_2D

function potential_grid_points(
    potential_func::Function,
    limits::Vector{<:Vector},
    step::Real;
    cutoff = nothing,
    dist_units = u"nm",
)
    rng1 = Vector(limits[1][1]:step:limits[1][2]) .* dist_units
    rng2 = Vector(limits[2][1]:step:limits[2][2]) .* dist_units 

    coords = []

    for i = 1:length(rng1)
        for j = 1:length(rng2)
            coord = SVector{2}([rng1[i],rng2[j]])
            Vij = ustrip(potential_func(coord))
            if typeof(cutoff) <: Real && Vij <= cutoff
                append!(coords, [coord])
            elseif typeof(cutoff) <: Vector && cutoff[1] <= Vij <= cutoff[2]
                append!(coords, [coord])
            end
        end
    end

    return coords
end


# general function for plotting contours in 2D
function plot_contours_2D(
    eval_function::Function,
    xcoords::Vector,
    ycoords::Vector,
    label::String;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700, 600),
    ttl::String="",
)

    mx = length(xcoords) 
    my = length(ycoords)
    V_surf = Matrix{Float64}(undef, (mx,my))
    for i = 1:mx
        for j = 1:my
            coord = SVector{2}([xcoords[i], ycoords[j]])
            Vij = ustrip(eval_function(coord))
            if cutoffs[1] <= Vij <= cutoffs[2]
                V_surf[i,j] = Vij
            elseif Vij < cutoffs[1]
                V_surf[i,j] = -Inf
            elseif Vij > cutoffs[2]
                V_surf[i,j] = Inf
            end
        end
    end

    fig = Figure(resolution = res)
    ax = Axis(fig[1, 1][1, 1], 
            xlabel="x1", ylabel="x2",
            limits=(
            ustrip(xcoords[1]),
            ustrip(xcoords[end]),
            ustrip(ycoords[1]),
            ustrip(ycoords[end]),
            ),
            xgridvisible=false,
            ygridvisible=false,
            title=ttl,
        )
    # tune contour levels
    if lvls == "linear" && cutoffs == (-Inf,Inf)
        lvls = LinRange(minimum(V_surf), maximum(V_surf), 20)
    elseif lvls == "linear"
        lvls = LinRange(cutoffs[1], cutoffs[2], 20)
    elseif lvls == "log10" && cutoffs == (-Inf,Inf)
        lvls = exp.(LinRange(log(minimum(V_surf) .+ 1e-16), log(maximum(V_surf)), 20))
    elseif lvls == "log10"
        lvls = exp.(LinRange(log(cutoffs[1]), log(cutoffs[2]), 20))
    end
    
    if fill==true && cutoffs != [-Inf, Inf]
        ct = contourf!(ax, ustrip.(xcoords), ustrip.(ycoords), V_surf .+ 1e-16, levels=lvls, extendlow=:auto)
        # tightlimits!(ax)
        Colorbar(fig[1, 1][1, 2], ct, label=label)
    elseif fill==true
        ct = contourf!(ax, ustrip.(xcoords), ustrip.(ycoords), V_surf .+ 1e-16, levels=lvls)
        Colorbar(fig[1, 1][1, 2], ct, label=label)
    else
        ct = contour!(ax, ustrip.(xcoords), ustrip.(ycoords), V_surf .+ 1e-16, levels=lvls)
    end

    fig, ax
end


# plot contours for DoubleWell
function plot_contours_2D(
    inter::DoubleWell,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
)
    potential_func(coords) = SteinMD.potential_double_well(inter, coords)
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end

# plot contours for Himmelblau
function plot_contours_2D(
    inter::Himmelblau,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
    ttl::String = "",
)
    potential_func(coords) = SteinMD.potential_himmelblau(inter, coords)
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res, ttl=ttl)
end

# plot contours for Sinusoid
function plot_contours_2D(
    inter::Sinusoid,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
)
    potential_func(coords) = SteinMD.potential_sinusoid(inter, coords)
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end

# plot contours for MullerBrown
function plot_contours_2D(
    inter::MullerBrown,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
)
    potential_func(coords) = Molly.potential_muller_brown(inter, coords)
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end

# plot contours for MullerBrown
function plot_contours_2D(
    inter::MullerBrownRot,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (900,600),
)
    potential_func(coords) = Molly.potential_muller_brown(inter, coords) 
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end

# plot contours for PolynomialChaos
function plot_contours_2D(
    inter::PolynomialChaos,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
    ttl::String="",
)
    potential_func(coords) = SteinMD.potential_pce(inter, coords)
    return plot_contours_2D(potential_func, xcoords, ycoords, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res, ttl=ttl)
end


function plot_density(
    inter::PolynomialChaos,
    xcoords::Vector,
    ycoords::Vector,
    normint::Integrator;
    kB=1.0u"kJ * K^-1 * mol^-1", # Maxwell-Boltzmann constant
    temp=1.0u"K", # temperature
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
)
    β = ustrip(1 / (kB * temp))
    p = define_gibbs_dist(inter, β=β, θ=inter.params)
    density_func(coord) = updf(p, get_values(coord)) / normconst(p, normint)
    return plot_contours_2D(density_func, xcoords, ycoords, "p(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end


function plot_density(
    inter::GeneralInteraction,
    xcoords::Vector,
    ycoords::Vector,
    normint::Integrator;
    kB=1.0u"kJ * K^-1 * mol^-1", # Maxwell-Boltzmann constant
    temp=1.0u"K", # temperature
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (900,600),
)
    β = ustrip(1 / (kB * temp))
    p = define_gibbs_dist(inter, β=β)
    Zp = normconst(p, normint)
    density_func(coord) = updf(p, get_values(coord)) / Zp
    return plot_contours_2D(density_func, xcoords, ycoords, "p(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end



function plot_basis(
    pce::PolynomialChaos,
    xcoords::Vector,
    ycoords::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    )

    mx = length(xcoords) 
    my = length(ycoords)
    N = length(pce.basis)
    Mset = SteinMD.TotalDegreeMset(pce.p, pce.d)

    bas_mat = Matrix{Vector}(undef, (mx,my))
    for i = 1:mx
        for j = 1:my
            coord = ustrip.([xcoords[i], ycoords[j]])
            bas_mat[i,j] =  SteinMD.eval_basis(coord, pce.basis)
        end
    end

    figs = Vector{Figure}(undef, N)
    axs = Vector{Axis}(undef, N)
    for n = 1:N
        V_surf = [bas_mat[i,j][n] for i = 1:mx, j=1:my]
        M = Mset[n]

        figs[n] = Figure(resolution = (700, 600))
        axs[n] = Axis(figs[n][1, 1][1, 1], 
                    xlabel="x1", ylabel="x2",
                    limits=(
                        ustrip(xcoords[1]),
                        ustrip(xcoords[end]),
                        ustrip(ycoords[1]),
                        ustrip(ycoords[end]),
                    ),
                    xgridvisible=false,
                    ygridvisible=false,
                    title="Product basis $n (degrees $M)")

        # tune contour levels
        if lvls == "linear"
            lvls = LinRange(minimum(V_surf), maximum(V_surf), 20)
        elseif lvls == "log10"
            lvls = exp.(LinRange(log(minimum(V_surf)), log(maximum(V_surf)), 20))
        end

        if fill==true
            ct = contourf!(axs[n], ustrip.(xcoords), ustrip.(ycoords), V_surf, levels=lvls)
            # Colorbar(figs[n][1, 1][1, 2], ct, label="V(x)")
        else
            ct = contour!(axs[n], ustrip.(xcoords), ustrip.(ycoords), V_surf, levels=lvls)
        end

    end
    return figs
end
        
