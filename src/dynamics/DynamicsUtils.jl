# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - linearize_dynamics(dyn, x, us) - this function linearizes the dynamics given the state and controls.
# - propagate_dynamics(cost, t, x, us) - this function propagates the dynamics to the next timestep.
# Every Dynamics struct must have a sys_info field of type SystemInfo.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# Export the types of dynamics.
export Dynamics, NonlinearDynamics


# Dimensionality helpers.
function num_agents(dyn::Dynamics)
    return num_agents(dyn.sys_info)
end

function xdim(dyn::Dynamics)
    return xdim(dyn.sys_info)
end

function udim(dyn::Dynamics)
    return udim(dyn.sys_info)
end

function udim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx)
end

export num_agents, xdim, udim
