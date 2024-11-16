function get_simple_straight_line_2D_traj()
    x1 = [0; 0; 0; 0]
    # human aim to follow y = 0 (x < 80), 1 (x>80) approximated by y = 0.01x
    u1 = vcat(hcat(10*ones(1, 10),
                   zeros(1, 35),
                   -20*ones(1, 5),
                   5*ones(1, 5),
                   zeros(1, 46)
              ),
              hcat(zeros(1, 50),
                   2*ones(1, 5),
                   zeros(1, 42),
                   -2*ones(1, 4)
              ),
        )
     # robot aim to follow y = 0
     u2 = vcat(zeros(1, 101),
               hcat(zeros(1, 50),
                    reshape(-0.001 * [i for i in 1:51], 1, 51)
                    )
               )

    us = [u1, u2]
    return us, x1
end

using Symbolics

function craft_control_inputs(X, u1_scaling) #X is the state
    K0 = 1.0 #Spring constant
    @variables x
    f(x) = sin(x) #Trajectory human wants to follow
    r(x) = 0.8*(0.8 * sin(x) + 0.1 * sin(0.9 * x) + 0.1 * sin(0.8 * x)) #Trajectory robot wants to follow
    # f(x) = 0.3*x
    # r(x) = 0

    x_t = X[1]
    y_t = X[3]
    f_prime = Symbolics.derivative(f(x), x)
    f_prime_val = Symbolics.substitute(f_prime, x => x_t)
    
    r_prime = Symbolics.derivative(r(x), x)
    r_prime_val = Symbolics.substitute(r_prime, x => x_t)
    
    u_1_x_t = 1/sqrt(1+f_prime_val^2)
    u_1_x_n = f_prime_val*(y_t - Symbolics.substitute(f(x), x => x_t)) / (1+f_prime_val^2)
    u_1_x = u_1_x_t + u_1_x_n

    u_1_y_t = f_prime_val/sqrt(1+f_prime_val^2) 
    u_1_y_n = (y_t - Symbolics.substitute(f(x), x => x_t))*(-1) / (1+f_prime_val^2)
    u_1_y = u_1_y_t + u_1_y_n

    u_1_x = Float64(Symbolics.value(u_1_x))/40 / u1_scaling
    u_1_y = Float64(Symbolics.value(u_1_y))/40 * u1_scaling


    #u2 is generating force only in the normal direction to r(x)
    u_2_x = (r_prime_val*(y_t - Symbolics.substitute(r(x), x => x_t)) / (1+r_prime_val^2)) * K0
    u_2_y = ((y_t - Symbolics.substitute(r(x), x => x_t))*(-1) / (1+r_prime_val^2)) * K0

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

    # Populate state trajectory.
    xs = zeros(xdim(dyn), horizon)
    u1s = zeros(udim(dyn, 1), horizon)
    u2s = zeros(udim(dyn, 2), horizon)
    
    xs[:, 1] = x₁
    
    u1, u2 = craft_control_inputs(xs[:,1], 1)
    u1s[:, 1] = u1
    u2s[:, 1] = u2
    for tt in 2:horizon
        us_prev = [u1s[:, tt-1], u2s[:, tt-1]]
        time_range = (times[tt-1], times[tt])
        xs[:, tt] = propagate_dynamics(dyn, time_range, xs[:, tt-1], us_prev)
        u1, u2 = tt < horizon/2 ?  craft_control_inputs(xs[:,tt], 1) : craft_control_inputs(xs[:,tt], 5)
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