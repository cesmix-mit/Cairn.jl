import Molly: log_property!

export
    StepComponentLogger,
    has_step_property


"""
    StepComponentLogger(observable::T, nsteps::Int, history::Vector{T})

A logger which holds a record of components of the update step to the coordinate positions. 

# Arguments
- `observable::T`   : observable quantity of type `T`. 
- `n_steps::Int`   : time step interval at which the observable is recorded. 
- `history::Vector{T}`   : record of values of the observable.

"""
mutable struct StepComponentLogger{T}
    observable::T
    n_steps::Int
    history::Vector{T}
end


StepComponentLogger(T::DataType, n_steps::Integer; dims::Integer=3) = StepComponentLogger{T}(T[], n_steps, T[])


function StepComponentLogger(n_steps::Integer; dims::Integer=3)
    T = Vector{Array{SArray{Tuple{dims}, Quantity, 1, dims}, 1}}
    return StepComponentLogger(T, n_steps; dims=dims)
end


Base.values(logger::StepComponentLogger) = logger.history


function log_property!(logger::StepComponentLogger, s::System, neighbors=nothing,
                        step_n::Integer=0; n_threads::Integer=Threads.nthreads(), stepcomp=[], kwargs...)
    logger.observable = stepcomp
    if (step_n % logger.n_steps) == 0
        push!(logger.history, stepcomp)
    end
end


function Base.show(io::IO, fl::StepComponentLogger{T}) where T
    print(io, "StepComponentLogger{", eltype(eltype(values(fl))), "} with n_steps ",
            fl.n_steps, ", ", length(values(fl)), " frames recorded for ",
            length(values(fl)) > 0 ? length(values(fl)) : "?", " atoms")
end


