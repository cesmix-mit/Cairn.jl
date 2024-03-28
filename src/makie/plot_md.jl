export plot_md_trajectory, plot_md_trajectory_hist


# plot single-trajectory MD
function plot_md_trajectory(
    sys::System,
    contourgrid::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
    showpath::Bool=false,
)
    # preprocess
    xcoords, ycoords = contourgrid
    inter = sys.general_inters[1]

    if fill == true
        c_traj = :red 
        c_train = :white 
    else
        c_traj = :Green
        c_train = :black 
    end 

    # create figure
    f, ax = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls)
    # plot trajectory
    coordvals = reduce(hcat, get_values(sys.loggers.coords))'
    if showpath == true
        scatterlines!(ax, coordvals[:,1], coordvals[:,2], markersize=5, color=c_traj)
    else
        scatter!(ax, coordvals[:,1], coordvals[:,2], markersize=5, color=c_traj)
    end
    return f
end


# plot single-trajectory MD and training points
function plot_md_trajectory(
    sys::System,
    sys_train::Vector{<:System},
    contourgrid::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
    showpath::Bool=false,
)
    # preprocess
    xcoords, ycoords = contourgrid
    inter = sys.general_inters[1]

    if fill == true
        c_traj = :red 
        c_train = :white 
    else
        c_traj = :Green
        c_train = :black 
    end 

    # create figure
    f, ax = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls)
    # plot trajectory
    coordvals = reduce(hcat, get_values(sys.loggers.coords))'
    if showpath == true
        scatterlines!(ax, coordvals[:,1], coordvals[:,2], markersize=5, color=c_traj, label="trajectory")
    else
        scatter!(ax, coordvals[:,1], coordvals[:,2], markersize=5, color=c_traj, label="trajectory")
    end
    # plot fixed points
    coordfix = reduce(hcat, [ustrip.(sys_i.coords)[1] for sys_i in sys_train])'
    scatter!(ax, coordfix[:,1], coordfix[:,2], markersize=7, color=c_train, label="fixed points")
    axislegend(ax)
    return f
end


# plot first and last position of ensemble MD
function plot_md_trajectory(
    ens::Vector{<:System},
    contourgrid::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
    showpath::Bool=false,
)
    # preprocess
    xcoords, ycoords = contourgrid
    inter = ens[1].general_inters[1]

    if fill == true
        c_start = :white
        c_end = :red 
    else
        c_start = :black
        c_end = :red 
    end 

    # create figure
    f, ax = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls)
    
    for (i,sys) in enumerate(ens)
        coordvals = reduce(hcat, get_values(sys.loggers.coords))'
        # plot trajectory
        if showpath == true
            scatterlines!(ax, coordvals[:,1], coordvals[:,2], markersize=5)
        else
            scatter!(ax, coordvals[:,1], coordvals[:,2], markersize=5)
        end
        # plot start and end points
        if i == 1
            scatter!(ax, coordvals[1,1], coordvals[1,2], markersize=10, color=c_start, label="start")
            scatter!(ax, coordvals[end,1], coordvals[end,2], markersize=10, color=c_end, label="end")
        else
            scatter!(ax, coordvals[1,1], coordvals[1,2], markersize=10, color=c_start)
            scatter!(ax, coordvals[end,1], coordvals[end,2], markersize=10, color=c_end)
        end
    end
    axislegend(ax)
    return f
end


# plot first and last position of ensemble MD
function plot_md_trajectory(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    contourgrid::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
    showpath::Bool=false,
)
    # preprocess
    xcoords, ycoords = contourgrid
    inter = ens[1].general_inters[1]

    # create figure
    f, ax = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls)
    
    for (i,sys) in enumerate(ens)
        coordvals = reduce(hcat, get_values(sys.loggers.coords))'
        # plot trajectory
        if showpath == true
            scatterlines!(ax, coordvals[:,1], coordvals[:,2], markersize=5)
        else
            scatter!(ax, coordvals[:,1], coordvals[:,2], markersize=5)
        end
    end
    # plot training points
    coordfix = reduce(hcat, [ustrip.(sys_i.coords)[1] for sys_i in sys_train])'
    scatter!(ax, coordfix[:,1], coordfix[:,2], markersize=7, color=:black, label="train. points")

    axislegend(ax)
    return f
end


# plot snapshots of single-trajectory MD
function plot_md_trajectory_hist(
    sys::System,
    sys_train::Vector{<:System},
    contourgrid::Vector,
    intervals::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
)

    # if length(intervals) != length(sys.loggers.params.history)
    #     throw(ArgumentError("ERROR: `intervals` must be the same length as the parameter history in `sys.loggers.params.history`"))
    # end

    # preprocess 
    intervals = vcat(1, intervals) # add initial condition
    T = length(intervals)
    inter = sys.general_inters[1]
    params_hist = reduce(vcat, [[sys.loggers.params.history[1]], sys.loggers.params.history])
    xcoords, ycoords = contourgrid
    x_md = reduce(hcat, get_values(sys.loggers.coords))
    x_train = reduce(hcat, reduce(vcat, get_values(get_coords(sys_train))))
    start = length(sys_train) - T+1

    # create figure
    figs = Vector{Figure}(undef, T)
    axs = Vector{Axis}(undef, T)
    for i = 1:T
        # plot contours of potential
        inter.params = params_hist[i]
        figs[i], axs[i] = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls)
        
        if i != 1
            int_old = 1:intervals[i-1]
            int_new = intervals[i-1]:intervals[i]
            scatterlines!(axs[i], x_md[1,int_old], x_md[2,int_old], linestyle=:dash, markersize=10, color=(:red,0.4))
            scatterlines!(axs[i], x_md[1,int_new], x_md[2,int_new], linestyle=:dash, markersize=10, color=:red, label="MD trajectory")
            scatter!(axs[i], x_train[1,1:start+i-1], x_train[2,1:start+i-1], markersize=10, color=:white, label="train. data")
            scatter!(axs[i], x_md[1,int_new[end]], x_md[2,int_new[end]], markersize=20, color=:cyan, label="new train. datum")
            x_train = hcat(x_train, x_md[:,int_new[end]])
        else
            scatter!(axs[i], x_train[1,1:start], x_train[2,1:start], markersize=10, color=:white, label="train. data")
        end
        
        axislegend(axs[i])
        
    end
    return figs

end


# plot snapshots of ensemble MD
function plot_md_trajectory_hist(
    ens::Vector{<:System},
    sys_train::Vector{<:System},
    contourgrid::Vector,
    intervals::Vector;
    fill::Bool=false,
    lvls::Union{String, Vector, StepRange, StepRangeLen}="linear",
)
    sys = ens[1]
    # if length(intervals) != length(sys.loggers.params.history)
    #     throw(ArgumentError("ERROR: `intervals` must be the same length as the parameter history in `sys.loggers.params.history`"))
    # end

    # preprocess 
    # intervals = vcat(intervals, 20_000) # add initial condition
    T = length(intervals)
    inter = sys.general_inters[1]
    params_hist = reduce(vcat, [[sys.loggers.params.history[1]], sys.loggers.params.history, ])
    xcoords, ycoords = contourgrid
    x_train = reduce(hcat, reduce(vcat, get_values(get_coords(sys_train))))
    nens = length(ens)
    nstart = length(sys_train) - (T-1)*nens

    # create figure
    figs = Vector{Figure}(undef, T)
    axs = Vector{Axis}(undef, T)
    for i = 1:T
        # plot contours of potential
        inter.params = params_hist[i]
        figs[i], axs[i] = plot_contours_2D(inter, xcoords, ycoords, fill=fill, lvls=lvls, ttl="Training iteration $i")
        # title!(axs[i], "Training iteration $i")
        if i != 1
            int_old = 1:intervals[i-1]
            int_new = intervals[i-1]:intervals[i]
            # if length(int_new) > 500; int_new = int_new[1:500]; end
            ens_old = x_train[:, 1:(nstart+nens*(i-2))]
            ens_new = x_train[:, (nstart+nens*(i-2)+1):(nstart+nens*(i-1))]

            for sys in ens
                x_md = reduce(hcat, get_values(sys.loggers.coords))
                # scatterlines!(axs[i], x_md[1,int_old], x_md[2,int_old], linestyle=:dash, markersize=5, color=(:gold,0.05))
                scatterlines!(axs[i], x_md[1,int_new], x_md[2,int_new], linestyle=:dash, markersize=5, color=:tan1)
            end
            scatter!(axs[i], ens_old[1,:], ens_old[2,:], markersize=10, color=:lightgoldenrod1, label="train. set")
            scatter!(axs[i], ens_new[1,:], ens_new[2,:], markersize=14, color=:red3, label="active set")
        else
            scatter!(axs[i], x_train[1,1:nstart], x_train[2,1:nstart], markersize=10, color=:red3, label="train. set")
        end
        
        axislegend(axs[i], position=:rb)
        
    end
    return figs

end