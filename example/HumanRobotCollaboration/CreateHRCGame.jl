function combine_cost_funcs(funcs, weights)
    @assert size(funcs) == size(weights)

    # Create a weighted cost function ready for use with autodiff.
    g(si, x, us, t) = begin
        f_eval = [f(si, x, us, t) for f in funcs]
        return weights' * f_eval
    end

    return g
end

function create_HRC_costs(T, goal_pos_x, h, r, h_prime, r_prime) 
    traj_deviation_cost_human_fn(si, x, us, t) = begin
        h_x = h(x[1])
        hp_x = h_prime(x[1]) 
        x_diff = hp_x * (x[3] - h_x) / (1 + hp_x^2)
        y_bar = h_x + hp_x^2 * (x[3] - h_x) / (1 + hp_x^2)
        y_diff = y_bar - x[2]
        dist_to_goal = (x[1]-goal_pos_x)^2
        return x_diff^2 + y_diff^2 + 0.01 / T * dist_to_goal
    end

    traj_deviation_cost_robot_fn(si, x, us, t) = begin
        r_x = r(x[1])
        rp_x = r_prime(x[1]) 
        x_diff = rp_x * (x[3] - r_x) / (1 + rp_x^2)
        y_bar = r_x + rp_x^2 * (x[3] - r_x) / (1 + rp_x^2)
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