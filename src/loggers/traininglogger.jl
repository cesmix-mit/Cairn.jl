import Molly: GeneralObservableLogger, log_property!

export TrainingLogger

"""
    TrainingLogger(params::T, nsteps::Int, history::Vector{T})

A logger which holds a record of parameters of the ML potential over iterations of training.

# Arguments
- `params::T`   : parameters of type `T`. 
- `n_steps::Int`   : time step interval at which the parameters are recorded. 
- `history::Vector{T}`   : record of values of the parameter.
"""
mutable struct TrainingLogger{T}
    params::T
    history::Vector{T}
end


function TrainingLogger(T::DataType)
    return TrainingLogger{T}(T[], Vector{T}[])
end
TrainingLogger() = TrainingLogger(Vector{Float64})


function log_property!(logger::TrainingLogger, s::System, neighbors=nothing,
    step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
    params = s.general_inters[1].params
    if step_n == 0 || params != logger.history[end]
        push!(logger.history, params)
    end
end

Base.values(logger::TrainingLogger) = logger.history


function Base.show(io::IO, fl::TrainingLogger)
    print(io, "TrainingLogger{", eltype(fl.params), "} with ", length(values(fl)), " frames recorded for ",
            length(values(fl)) > 0 ? length(first(values(fl))) : "?", " parameters")
end
