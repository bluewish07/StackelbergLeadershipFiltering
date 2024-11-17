function combine_cost_funcs(funcs, weights)
    @assert size(funcs) == size(weights)

    # Create a weighted cost function ready for use with autodiff.
    g(si, x, us, t) = begin
        f_eval = [f(si, x, us, t) for f in funcs]
        return weights' * f_eval
    end

    return g
end

function create_HRC_costs(T, goal_pos_x ) 
    # given human desired traj function
    # y = h1(x), x, y are scalars
    h1(x) = begin
        return 0.1*x + 2*2.7182^(-(x/0.5-4)^2) #sin(x)
        # return 0.5*x
    end
    # closed form derivative of h1
    h1_prime(x) = begin
        return 0.1 - (16*x-32)-2.7182^(-(4-2*x)^2)
        # return 0.5
    end
    traj_deviation_cost_human_fn(si, x, us, t) = begin
        h1_x = h1(x[1])
        h1p_x = h1_prime(x[1]) 
        x_diff = h1p_x * (x[3] - h1_x) / (1 + h1p_x^2)
        y_bar = h1_x + h1p_x^2 * (x[3] - h1_x) / (1 + h1p_x^2)
        y_diff = y_bar - x[2]
        dist_to_goal = (x[1]-goal_pos_x)^2
        return x_diff^2 + y_diff^2 + 0.01 / T * dist_to_goal
    end

    # given robot desired traj function
    # y = h2(x), x, y are scalars
    h2(x) = begin
        return 0.1*x #0.8 * sin(x) + 0.1 * sin(0.9 * x) + 0.1 * sin(0.8 * x)
        # return 0
    end
    # closed form derivative of h2
    h2_prime(x) = begin
        return 0.1 #0.8*cos(x) + 0.1*0.9*cos(0.9*x) + 0.1*0.8*cos(0.8*x)
        # return 0
    end
    traj_deviation_cost_robot_fn(si, x, us, t) = begin
        h2_x = h2(x[1])
        h2p_x = h2_prime(x[1]) 
        x_diff = h2p_x * (x[3] - h2_x) / (1 + h2p_x^2)
        y_bar = h2_x + h2p_x^2 * (x[3] - h2_x) / (1 + h2p_x^2)
        y_diff = y_bar - x[3]
        return x_diff^2 + y_diff^2
    end

    c1_traj = PlayerCost(traj_deviation_cost_human_fn, si)
    c2_traj = PlayerCost(traj_deviation_cost_robot_fn, si)

    ctrl_const = 2
    c1_control = QuadraticCost(zeros(4, 4))
    add_control_cost!(c1_control, 1, ctrl_const * diagm([0.5, 1]))
    add_control_cost!(c1_control, 2, zeros(2, 2)) # human doesn't care about robot's control cost

    c2_control = QuadraticCost(zeros(4, 4))
    add_control_cost!(c2_control, 1, ctrl_const * diagm([0, 1])) # robot should care about not making human pay too much control cost
    add_control_cost!(c2_control, 2, ctrl_const * diagm([0.5, 0.5]))

    costs_p1 = [c1_traj, c1_control]
    weights_p1 = ones(length(costs_p1))
    costs_p2 = [c2_traj, c2_control]
    weights_p2 = ones(length(costs_p2))
    g1 = combine_cost_funcs(get_as_function.(costs_p1), weights_p1)
    sum_cost_p1 = PlayerCost(g1, si)

    g2 = combine_cost_funcs(get_as_function.(costs_p2), weights_p2)
    sum_cost_p2 = PlayerCost(g2, si)

    return [sum_cost_p1, sum_cost_p2]
end