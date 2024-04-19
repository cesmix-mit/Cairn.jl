using Molly
export plot_surface

# general function for plotting contours in 2D
function plot_surface(
    inter,
    potential_function::Function,
    xcoords::Vector,
    ycoords::Vector;
    cutoff = nothing,
    plotangle = nothing,
)

    mx = length(xcoords) 
    my = length(ycoords)
    V_surf = Matrix{Float64}(undef, (mx,my))

    if cutoff == nothing
        for i = 1:mx
            for j = 1:my
                coord = SVector{2}([xcoords[i], ycoords[j]])
                V_surf[i,j] = ustrip(potential_function(inter, coord))
            end
        end
    else
        for i = 1:mx
            for j = 1:my
                coord = SVector{2}([xcoords[i], ycoords[j]])
                V_ij = ustrip(potential_function(inter, coord))
                if V_ij <= cutoff
                    V_surf[i,j] = V_ij
                else
                    V_surf[i,j] = NaN
                end
            end
        end
    end

    fig = Figure(size = (700, 600))
    if plotangle == nothing
        ax = Axis3(fig[1, 1], 
                aspect = (1,1,1),
                xlabel="x1",
                ylabel="x2",
                zlabel="V(x1,x2)",
                # limits=(
                # ustrip(xcoords[1]),
                # ustrip(xcoords[end]),
                # ustrip(ycoords[1]),
                # ustrip(ycoords[end]),
                # ),
                # xgridvisible=false,
                # ygridvisible=false
                )
    else
        ax = Axis3(fig[1, 1], 
                aspect = (1,1,1),
                azimuth = plotangle[1]*pi,
                elevation = plotangle[2]*pi,
                xlabel="x1",
                ylabel="x2",
                zlabel="V(x1,x2)",
                # limits=(
                # ustrip(xcoords[1]),
                # ustrip(xcoords[end]),
                # ustrip(ycoords[1]),
                # ustrip(ycoords[end]),
                # ),
                # xgridvisible=false,
                # ygridvisible=false
                )
    end
    
    surface!(ax, ustrip.(xcoords), ustrip.(ycoords), V_surf, colormap=:deep)
    fig, ax
end


# plot contours for DoubleWell
function plot_surface(
    inter::DoubleWell,
    xcoords::Vector,
    ycoords::Vector;
    cutoff = nothing,
    plotangle = nothing,
)
    return plot_surface(inter, Cairn.potential_double_well, xcoords, ycoords, cutoff=cutoff, plotangle=plotangle)

end

# plot contours for MullerBrown
function plot_surface(
    inter::Molly.MullerBrown,
    xcoords::Vector,
    ycoords::Vector;
    cutoff = nothing,
    plotangle = nothing,
)
    return plot_surface(inter, Molly.potential_muller_brown, xcoords, ycoords, cutoff=cutoff, plotangle=plotangle)

end

# plot contours for PolynomialChaos
function plot_surface(
    inter::PolynomialChaos,
    xcoords::Vector,
    ycoords::Vector;
    cutoff = nothing,
    plotangle = nothing,
)
    return plot_surface(inter, Cairn.potential_pce, xcoords, ycoords, cutoff=cutoff, plotangle=plotangle)
end

