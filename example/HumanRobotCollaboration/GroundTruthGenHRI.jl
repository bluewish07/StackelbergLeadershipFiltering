using Symbolics

function craft_control_inputs(X, u1_scaling, tt, dt) #X is the state
    
    @variables x
    # f(x) = sin(x) #Trajectory human wants to follow
    # r(x) = 0.8*(0.8 * sin(x) + 0.1 * sin(0.9 * x) + 0.1 * sin(0.8 * x)) #Trajectory robot wants to follow
    # f(x) = 0.3*x
    # r(x) = 0
    f(x) = 0.1*x + exp(-(x-3)^2)
    r(x) = 0.1*x  
    damping_ratio = 0.0
    K0 = 0.8 #Spring constant
    u1_scaling = 1.0

    x_t = X[1]
    x_dot_t = X[2]
    y_t = X[3]
    y_dot_t = X[4]


    f_prime = Symbolics.derivative(f(x), x)
    f_prime_val = Symbolics.substitute(f_prime, x => x_t)
    f_val = Float64(Symbolics.value(Symbolics.substitute(f(x), x => x_t)))
    
    r_prime = Symbolics.derivative(r(x), x)
    r_prime_val = Symbolics.substitute(r_prime, x => x_t)
    r_val = Float64(Symbolics.value(Symbolics.substitute(r(x), x => x_t)))
    
    x_1_bar = x_t + f_prime_val*(y_t-f_val)/(1+f_prime_val^2)
    y_1_bar = f_val + f_prime_val^2*(y_t-f_val)/(1+f_prime_val^2)

    x_2_bar = x_t + r_prime_val*(y_t-r_val)/(1+r_prime_val^2)
    y_2_bar = r_val + r_prime_val^2*(y_t-r_val)/(1+r_prime_val^2)

    u_1_x_t = exp(-tt)
    u_1_y_t = 0.1*exp(-tt) - 2*exp(-(-4 + exp(-tt) + tt)^2)*(1 - exp(-tt))^2 - 2*exp(-tt - (-4 + exp(-tt) + tt)^2)*(-4 + exp(-tt) + tt) + 4*exp(-(-4 + exp(-tt) + tt)^2)*(1 - exp(-tt))^2*(-4 + exp(-tt) + tt)^2

    u_1_x_n = Float64(Symbolics.value(x_1_bar - x_t))
    u_1_y_n = Float64(Symbolics.value(y_1_bar - y_t))

    
    
    u_1_x = (u_1_x_t + u_1_x_n/2)/u1_scaling 
    u_1_y = (u_1_y_t + u_1_y_n/2)/u1_scaling 


    #u2 is generating force only in the normal direction to r(x)
    
    u_2_x = Float64(Symbolics.value(x_2_bar - x_t))
    u_2_y = Float64(Symbolics.value(y_2_bar - y_t))
    velocity_projection = u_2_x*x_dot_t + u_2_y*y_dot_t
    u_2_x = K0 * (u_2_x - velocity_projection*damping_ratio*x_dot_t)
    u_2_y = K0 * (u_2_y - velocity_projection*damping_ratio*y_dot_t)

    u_2_x = Float64(Symbolics.value(u_2_x))
    u_2_y = Float64(Symbolics.value(u_2_y))
    
    u_1 = vcat(u_1_x, u_1_y)
    u_2 = vcat(u_2_x, u_2_y)
    
    return u_1, u_2
end

function distance_calculator(point::Tuple{Float64, Float64}, fnc)
    #Calculates the distance from a desired point to the closest point on a function called fnc
    @variables x
    x_t, y_t = point
    f_prime = Symbolics.derivative(fnc(x), x)
    f_prime_val = Symbolics(x).substitute(f_prime, x => x_t)
    
    d = sqrt(f_prime_val^2+(2*f_prime_val^2+1)^2)*(y_t - Symbolics.substitute(fnc(x), x => x_t))/(1+f_prime_val^2)
    return d
end

function unroll_raw_controls_4_HRI(dyn::Dynamics, times::AbstractVector{Float64}, x₁)
    @assert length(x₁) == xdim(dyn)

    N = dyn.sys_info.num_agents

    horizon = length(times)
    dt = times[2] - times[1]

    # Populate state trajectory.
    xs = zeros(xdim(dyn), horizon)
    u1s = zeros(udim(dyn, 1), horizon)
    u2s = zeros(udim(dyn, 2), horizon)
    # println(times)
    xs[:, 1] = x₁
    
    u1, u2 = craft_control_inputs(xs[:,1], 1, 0, dt)
    u1s[:, 1] = u1
    u2s[:, 1] = u2
    for tt in 2:horizon
        us_prev = [u1s[:, tt-1], u2s[:, tt-1]]
        time_range = (times[tt-1], times[tt])
        xs[:, tt] = propagate_dynamics(dyn, time_range, xs[:, tt-1], us_prev)
        u1, u2 = tt < horizon/2 ?  craft_control_inputs(xs[:,tt], 1, times[tt], dt) : craft_control_inputs(xs[:,tt], 5, times[tt], dt)
        u1s[:, tt] = u1
        u2s[:, tt] = u2
    end
    return xs, [u1s, u2s]
end

function get_ground_truth_traj(dyn::Dynamics, times::AbstractVector{Float64})
     x₁ = [0;0;0;0]
     xs, us = unroll_raw_controls_4_HRI(dyn, times, x₁)
     return x₁, xs, us
end