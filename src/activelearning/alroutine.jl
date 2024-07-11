import PotentialLearning: SubsetSelector, LearningProblem
export
    ActiveLearnRoutine,
    update_sys,
    compute_error_metrics!


"""
    ActiveLearnRoutine

Struct containing all parameters required to trigger retraining during simulation.

# Arguments
- `ref :: Union{GeneralInteraction, PairwiseInteraction}` : reference (ground truth) potential 
- `mlip :: MLInteraction` : machine learning (surrogate) potential)
- `sys_train :: Vector{<:System}` : ensemble of systems representing the training data
- `eval_int` : Integrator specifying mode of integration
- `trigger :: ActiveLearningTrigger` : defines trigger function for active learning
- `error_hist :: Dict` : dictionary of error metrics to record over simulation

# Keyword Arguments
- `update_func :: Function = update_sys` : function for updating the the training and simulation systems
- `train_func :: Function - train_potential_e!` : function defining training objective for `mlip`
- `train_steps :: Vector{<:Integer} = [1]` : vector of simulation steps at which training is trigger_activated
- `param_hist :: Vector{<:Vector} = [mlip.params]` : vector of the parameter history
- `kwargs...` : additional keyword arguments for `update_func` and `train_func`

""" 

mutable struct ActiveLearnRoutine
    ref # :: Union{GeneralInteraction, PairwiseInteraction}
    mlip :: MLInteraction
    trainset :: Vector{<:System} 
    triggers :: Tuple # <: ActiveLearningTrigger
    ss :: SubsetSelector
    lp :: LearningProblem
    data :: Union{Nothing, Dict}
end

function ActiveLearnRoutine(;
    ref,
    mlip::MLInteraction,
    trainset::Vector{<:System},
    triggers::Tuple,
    ss::SubsetSelector,
    lp::LearningProblem,
    kwargs...
)
    # populate sys_train with descriptors
    compute_local_descriptors(trainset, mlip)
    compute_force_descriptors(trainset, mlip)

    # initialize AL data
    aldata = Dict(
        "train_steps" => [1],
        "param_hist" => [mlip.params],
        "trigger_eval" => [],
    )

    return ActiveLearnRoutine(ref, mlip, trainset, triggers, ss, lp, aldata)
end





"""
    update_sys(sim::Simulator)

Performs online active learning by molecular dynamics simulation defined in `sim`, using the retraining criterion defined in `trigger`.

# Arguments
- `sim::Simulator` : simulator of the equations of motion
- `sys::Union{System,Vector{<:System}}` : simulation system
- `sys_train::Vector{<:System}`: Vector of systems in the training data

# Keyword Arguments
- `n_add::Integer=10` : number of samples to add to `sys_train`
- `steps::Integer=1000` : number of steps in the sample path over which to draw new training data
    
"""
function update_sys(sim::OverdampedLangevin,
    sys::System,
    sys_train::Vector{<:System};
    n_add=10,    
    steps=1000,         
)
    coords = sys.loggers.coords.history[end:-1:(end-steps)]
    coords_new = StatsBase.sample(coords, n_add; replace=false)

    sys_new = [System(
        atoms=sys.atoms,
        coords=coords_i,
        boundary=sys.boundary,
        general_inters=sys.general_inters,
    ) for coords_i in coords_new]

    sys_train = reduce(vcat, [sys_train, sys_new])

    return sys_train
end


function update_sys(sim::SteinRepulsiveLangevin,
    sys::System,
    sys_train::Vector{<:System};    
    kwargs...           
)
    sys_new = remove_loggers(sys)   
    # sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new]) # update repulsive set
    sys_train = reduce(vcat, [sys_train, sys_new]) # update training data

    return sys_train
end


function update_sys(sim::StochasticSVGD,
    ens::Vector{<:System}, 
    sys_train::Vector{<:System};    
    kwargs...          
)
    ens_new = [remove_loggers(sys) for sys in ens]
    sys_train = reduce(vcat, [sys_train, ens_new]) # update training data
    sim.sys_fix = ens_new # sys_train
    return sys_train
end




"""
    compute_error_metrics!(al::ActiveLearnRoutine)

Appends error metric calculations (RMSE in energies and forces, Fisher divergence in probability measures) to `al.error_hist`.

"""
function compute_error_metrics!(al::ActiveLearnRoutine)
    fd = compute_fisher_div(al.ref, al.mlip, al.eval_int)
    r_e, r_f = compute_rmse(al.ref, al.mlip, al.eval_int)
    append!(al.error_hist["fd"], fd)
    append!(al.error_hist["rmse_e"], r_e)
    append!(al.error_hist["rmse_f"], r_f)
end

