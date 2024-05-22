export 
    GeneralInteraction,
    MLInteraction,
    Simulator

"""
A general interaction that applies to single atoms (no multi-element interactions).
Custom general interactions should sub-type this abstract type.
"""
abstract type GeneralInteraction end


"""
A machine learning interatomic potential defining interactions between sets of atoms.
Custom machine learning interactions should sub-type this abstract type.
"""
abstract type MLInteraction end


"""
An abstract type for simulators. 
Custom simulators should sub-type this abstract type.
"""
abstract type Simulator end
