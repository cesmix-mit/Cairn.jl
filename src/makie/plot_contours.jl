import Molly: potential_energy
export coord_grid_2d, potential_grid_2d, plot_contours_2d


## grid on 2d domain
function coord_grid_2d(
    limits::Vector{<:Vector},
    step::Real;
    dist_units = u"nm"
)
    xcoord = Vector(limits[1][1]:step:limits[1][2]) .* dist_units
    ycoord = Vector(limits[2][1]:step:limits[2][2]) .* dist_units 
    return [xcoord, ycoord]
end


## generic potential energy function with coord as argument
function potential(inter, coord::SVector{2})
    sys = let coords=[coord]; () -> [SVector{2}(coords)]; end # pseudo-struct
    return potential_energy(inter, sys)
end


## grid across potential energy surface below cutoff
function potential_grid_2d(
    inter,
    limits::Vector{<:Vector},
    step::Real;
    cutoff = nothing,
    dist_units = u"nm",
)
    rng1, rng2 = coord_grid_2d(limits, step; dist_units=dist_units)
    coords = SVector[]

    for i = 1:length(rng1)
        for j = 1:length(rng2)
            coord = SVector{2}([rng1[i],rng2[j]])
            Vij = ustrip(potential(inter, coord))
            if typeof(cutoff) <: Real && Vij <= cutoff
                append!(coords, [coord])
            elseif typeof(cutoff) <: Vector && cutoff[1] <= Vij <= cutoff[2]
                append!(coords, [coord])
            end
        end
    end

    return coords
end


# general function for plotting contours in 2d
function plot_contours_2d(
    eval_function::Function,
    coord_grid::Vector,
    label::String;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700, 600),
    ttl::String="",
)

    xcoords, ycoords = coord_grid
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


## plot contours for 2d interatomic potential
function plot_contours_2d(
    inter,
    coord_grid::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    cutoffs::Tuple = (-Inf, Inf),
    res::Tuple = (700,600),
)
    potential_func = coords -> potential(inter, coords)
    return plot_contours_2d(potential_func, coord_grid, "V(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end


## plot density function for 2d interatomic potential
function plot_density(
    inter::GeneralInteraction,
    coord_grid::Vector,
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
    return plot_contours_2d(density_func, coord_grid, "p(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end

function plot_density(
    inter::PolynomialChaos,
    coord_grid::Vector,
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
    Zp = normconst(p, normint)
    density_func(coord) = updf(p, get_values(coord)) / Zp
    return plot_contours_2d(density_func, coord_grid, "p(x)", fill=fill, lvls=lvls, cutoffs=cutoffs, res=res)
end


## plot basis functions of PolynomialChaos
function plot_basis(
    pce::PolynomialChaos,
    coord_grid::Vector;
    fill::Bool=false,
    lvls::Union{Integer, String, Vector, StepRange, StepRangeLen}="linear",
    )

    xcoords, ycoords = coord_grid
    mx = length(xcoords) 
    my = length(ycoords)
    N = length(pce.basis)
    Mset = Cairn.TotalDegreeMset(pce.p, pce.d)

    bas_mat = Matrix{Vector}(undef, (mx,my))
    for i = 1:mx
        for j = 1:my
            coord = ustrip.([xcoords[i], ycoords[j]])
            bas_mat[i,j] =  Cairn.eval_basis(coord, pce.basis)
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
        
