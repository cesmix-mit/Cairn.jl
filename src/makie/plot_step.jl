export plot_step, plot_step_hist


# plot full history of step component for single-trajectory MD
function plot_step(
    sys::System,
    labels::Vector{String};
    logscl::Bool = false,
    nsteps = length(values(sys.loggers[1]))-1
)
    loggers = sys.loggers
    # check that loggers contains step components
    if !has_step_property(loggers)
        throw(ArgumentError("loggers does not include StepComponentLogger"))
    end
    # create figure
    fig = Figure(resolution = (800, 400))
    # set axes
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
    # plot step components
    lines!(ax, 1:nsteps, norm.(get_values(loggers.knl))[1:nsteps] .+ 1e-16, label="k(x,y) ∇V(x)")
    lines!(ax, 1:nsteps, norm.(get_values(loggers.gradknl))[1:nsteps] .+ 1e-16, label="∇k(x,y)")
    # lines!(ax, 1:nsteps, norm.(get_values(loggers.noise))[1:nsteps] .+ 1e-16, label="noise η")

    # for (logger,label) in zip(loggers,labels)
    #     if has_step_property(logger)
    #         lines!(ax, 1:nsteps, norm.(get_values(logger))[1:nsteps] .+ 1e-16, label=label)
    #     end
    # end

    axislegend(ax, position=:rb)

    return fig
end




# plot full history of step component for ensemble MD 
function plot_step(
    ens::Vector{<:System}, 
    label::String;
    logscl::Bool = false,
    nsteps = length(values(ens[1].loggers[1]))-1
)
    # check that loggers contains step components
    if !has_step_property(ens[1].loggers)
        throw(ArgumentError("loggers does not include StepComponentLogger"))
    end
    # create figure
    fig = Figure(resolution = (800, 400))
    # set axes
    if logscl == true
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            yscale=log10,
            # limits=(0, 10000, 1e-8, 1e-1),
            title=label)
    else
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            title=label)
    end
    # plot step components
    for sys in ens
        lines!(ax, 1:nsteps, norm.(get_values(sys.loggers.ksd))[1:nsteps] .+ 1e-16)

        # for logger in sys.loggers
        #     if has_step_property(logger)
        #         lines!(ax, 1:nsteps, norm.(get_values(logger))[1:nsteps] .+ 1e-16)
        #     end
        # end
    end

    return fig
end


# plot snapshots of the step component history for single-trajectory MD
function plot_step_hist(
    loggers::NamedTuple,
    intervals::Vector, 
    labels::Vector{String};
    logscl::Bool = false,
    nsteps = length(values(loggers[1]))-1
)
    # check that loggers contains step components
    if !has_step_property(loggers)
        throw(ArgumentError("loggers does not include StepComponentLogger"))
    end

    # create figure
    figs = Vector{Figure}(undef, length(intervals))
    axs = Vector{Axis}(undef, length(intervals))

    for i = 1:length(intervals)
        figs[i] = Figure(resolution = (800, 400))
        # set axes
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
        # plot step components
        for (logger,label) in zip(loggers,labels)
            if has_step_property(logger)
                lines!(axs[i], 1:nsteps, norm.(get_values(logger))[1:nsteps] .+ 1e-16, color=(:black, 0.1))
                lines!(axs[i], int_new, norm.(get_values(logger))[int_new] .+ 1e-16, label=label)
            end
        end

        axislegend(axs[i], position=:rb)
    end

    return figs
end


# plot snapshots of the step component history for ensemble MD
function plot_step_hist(
    ens::Vector{<:System}, 
    intervals::Vector, 
    label::String;
    logscl::Bool = false,
    nsteps = length(values(ens[1].loggers[1]))-1
)
    # check that loggers contains step components
    if !has_step_property(ens[1].loggers)
        throw(ArgumentError("loggers does not include StepComponentLogger"))
    end
    # create figure
    figs = Vector{Figure}(undef, length(intervals))
    axs = Vector{Axis}(undef, length(intervals))

    for i = 1:length(intervals)
        figs[i] = Figure(resolution = (800, 400))
        # set axes
        if logscl == true
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false,
                title=label,
                yscale=log10)
        else
            axs[i] = Axis(figs[i][1, 1][1, 1], 
                xlabel="iteration (t)",
                xgridvisible=false,
                ygridvisible=false,
                title=label)
        end
        int_new = 2:intervals[i] + 1
        # plot step components
        for sys in ens
            for logger in sys.loggers
                if has_step_property(logger)
                    lines!(axs[i], 1:nsteps, norm.(get_values(logger))[1:nsteps] .+ 1e-16, color=(:black, 0.1))
                    lines!(axs[i], int_new, norm.(get_values(logger))[int_new] .+ 1e-16)
                end
            end
        end

        axislegend(axs[i], position=:rb)
    end

    return fig
end


function plot_step_mean(
    ens::Vector{<:System}, 
    label::String;
    logscl::Bool = false,
    nsteps = length(values(ens[1].loggers[1]))-1
)
    # check that loggers contains step components
    if !has_step_property(ens[1].loggers)
        throw(ArgumentError("loggers does not include StepComponentLogger"))
    end
    # create figure
    fig = Figure(resolution = (800, 400))
    # set axes
    if logscl == true
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            yscale=log10,
            # limits=(0, 10000, 1e-8, 1e-1),
            title=label)
    else
        ax = Axis(fig[1, 1][1, 1], 
            xlabel="iteration (t)",
            xgridvisible=false,
            ygridvisible=false,
            title=label)
    end
    # plot step components
    N = length(ens)
    ksd_hist = Vector{Float64}(undef, nsteps)
    for sys in ens
        for logger in sys.loggers
            if has_step_property(logger)
                ksd_hist += norm.(get_values(logger))[1:nsteps]
            end
        end
    end
    lines!(ax, 1:nsteps, ksd_hist .+ 1e-16)

    return fig
end