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

function linearize_dynamics(dyn::LinearDynamics, t, x, us)
    return dyn
end

export LinearDynamics, propagate_dynamics, linearize_dynamics
