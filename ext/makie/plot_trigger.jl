export plot_trigger, plot_trigger_hist

# plot full history of trigger evaluation for single-trajectory MD
function plot_trigger(
    logger::TriggerLogger;
    logscl::Bool = false,
    nsteps::Integer = length(values(logger))-1
)

    fig = Figure(size = (800, 400))
    if logscl == true
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            yscale=log10)
    else
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false)
    end
    lines!(ax, 1:nsteps, get_values(logger)[1:nsteps] .+ 1e-16, label="trigger function")
    hlines!(ax, logger.trigger.thresh, color=:goldenrod, label="threshold")
    axislegend(ax, position=:rb)
    return fig, ax
end


# plot full history of trigger evaluation for ensemble MD
function plot_trigger(
    ens::Vector{<:System};
    logscl::Bool = false,
    nsteps::Integer = length(values(ens[1].loggers.trigger))-1
)

    fig = Figure(size = (800, 400))
    if logscl == true
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            yscale=log10)
    else
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false)
    end

    for sys in ens
        for logger in sys.loggers
            if typeof(logger) <: TriggerLogger
                lines!(ax, 1:nsteps, get_values(logger)[1:nsteps] .+ 1e-16)
            end
        end
    end
    logger = ens[1].loggers.trigger
    hlines!(ax, logger.trigger.thresh, color=:goldenrod, label="threshold")
    axislegend(ax, position=:rb)
    return fig, ax
end


# plot snapshots of the trigger history for single-trajectory MD
function plot_trigger_hist(
    logger::TriggerLogger,
    intervals::Vector;
    logscl::Bool = false,
    nsteps::Integer = length(values(logger))-1
)

    # create figure
    figs = Vector{Figure}(undef, length(intervals))
    axs = Vector{Axis}(undef, length(intervals))

    for i = 1:length(intervals)
        figs[i] = Figure(size = (800, 400))
        if logscl == true
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false,
                yscale=log10)
        else
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false)
        end
        int_new = 2:intervals[i] + 1
        lines!(axs[i], 1:nsteps, get_values(logger)[1:nsteps] .+ 1e-16, color=(:black, 0.1))
        lines!(axs[i], int_new, get_values(logger)[int_new] .+ 1e-16, label="trigger function")
        hlines!(axs[i], logger.trigger.thresh, color=:goldenrod, label="threshold")
        axislegend(axs[i], position=:rb)
    end
    return figs
end

# plot snapshots of the trigger history for single-trajectory MD
function plot_error_hist(
    error::Vector,
    intervals::Vector;
    logscl::Bool = false,
    nsteps::Integer = length(values(logger))-1
)

    # create figure
    figs = Vector{Figure}(undef, length(intervals))
    axs = Vector{Axis}(undef, length(intervals))

    for i = 1:length(intervals)
        figs[i] = Figure(size = (800, 400))
        if logscl == true
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false,
                yscale=log10)
        else
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false)
        end
        int_new = 2:intervals[i] + 1
        lines!(axs[i], 1:nsteps, get_values(logger)[1:nsteps] .+ 1e-16, color=(:black, 0.1))
        lines!(axs[i], int_new, get_values(logger)[int_new] .+ 1e-16, label="trigger function")
        hlines!(axs[i], logger.trigger.thresh, color=:goldenrod, label="threshold")
        axislegend(axs[i], position=:rb)
    end
    return figs
end