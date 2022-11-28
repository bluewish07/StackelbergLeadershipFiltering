# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - linearize_dynamics(dyn, x, us) - this function linearizes the dynamics given the state and controls.
# - propagate_dynamics(cost, t, x, us) - this function propagates the dynamics to the next timestep.
# Every Dynamics struct must have a sys_info field of type SystemInfo.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
struct LinearDynamics <: Dynamics
    A  # state
    Bs # controls
    sys_info::SystemInfo
end
# Constructor for linear dynamics that auto-generates the system info.
LinearDynamics(A, Bs) = LinearDynamics(A, Bs, SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

function propagate_dynamics(dyn::LinearDynamics, t, x, us)
    N = dyn.sys_info.num_agents
    x_next = dyn.A * x
    for i in 1:N
        ui = reshape(us[i], dyn.sys_info.num_us[i], 1)
        x_next += dyn.Bs[i] * ui
    end
    return x_next
end

function linearize_dynamics(dyn::LinearDynamics, x, us)
    return dyn
end

# Export the types of dynamics.
export Dynamics, NonlinearDynamics, LinearDynamics

# Export the functionality each Dynamics requires.
export propagate_dynamics, linearize_dynamics


# Dimensionality helpers.
function xdim(dyn::Dynamics)
    return dyn.sys_info.num_x
end

function udim(dyn::Dynamics)
    return sum(dyn.sys_info.num_us)
end

function udim(dyn::Dynamics, player_idx)
    return dyn.sys_info.num_us[player_idx]
end

export xdim, udim


# TODO(hamzah) Add better tests for the unroll_feedback, unroll_raw_controls functions.
# TODO(hamzah) Abstract the unroll_feedback, unroll_raw_controls functions to not assume linear feedback P.

# Function to unroll a set of feedback matrices from an initial condition.
# Output is a sequence of states xs[:, time] and controls us[player][:, time].
export unroll_feedback
function unroll_feedback(dyn::Dynamics, Ps, x₁)
    @assert length(x₁) == xdim(dyn)

    N = length(Ps)
    @assert N == length(dyn.Bs)

    horizon = last(size(first(Ps)))

    # Populate state/control trajectory.
    xs = zeros(xdim(dyn), horizon)
    xs[:, 1] = x₁
    us = [zeros(udim(dyn, ii), horizon) for ii in 1:N]
    for tt in 2:horizon
        for ii in 1:N
            us[ii][:, tt - 1] = -Ps[ii][:, :, tt - 1] * xs[:, tt - 1]
        end

        us_prev = [us[i][:, tt-1] for i in 1:N]
        xs[:, tt] = propagate_dynamics(dyn, tt, xs[:, tt-1], us_prev)
    end

    # Controls at final time.
    for ii in 1:N
        us[ii][:, horizon] = -Ps[ii][:, :, horizon] * xs[:, horizon]
    end

    return xs, us
end

# As above, but replacing feedback matrices `P` with raw control inputs `u`.
export unroll_raw_controls
function unroll_raw_controls(dyn::Dynamics, us, x₁)
    @assert length(x₁) == xdim(dyn)

    N = length(us)
    @assert N == length(dyn.Bs)

    horizon = last(size(first(us)))

    # Populate state trajectory.
    xs = zeros(xdim(dyn), horizon)
    xs[:, 1] = x₁
    us = [zeros(udim(dyn, ii), horizon) for ii in 1:N]
    for tt in 2:horizon
        us_prev = [us[i][:, tt-1] for i in 1:N]
        xs[:, tt] = propagate_dynamics(dyn, tt, xs[:, tt-1], us_prev)
    end

    return xs
end
