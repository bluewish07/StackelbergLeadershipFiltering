using Symbolics

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
    # given human desired traj function
    # y = h(x), x, y are scalars
    # @variables x_sym
    # h_prime_sym = Symbolics.derivative(h(x_sym), x_sym)
    # h_prime(x_t) = begin
    #     h_prime_val = Symbolics.value(Symbolics.substitute(h_prime_sym, x_sym => x_t))
    #     return h_prime_val
    # end

    traj_deviation_cost_human_fn(si, x, us, t) = begin
        h_x = h(x[1])
        hp_x = h_prime(x[1]) 
        x_diff = hp_x * (x[3] - h_x) / (1 + hp_x^2)
        y_bar = h_x + hp_x^2 * (x[3] - h_x) / (1 + hp_x^2)
        y_diff = y_bar - x[2]
        dist_to_goal = (x[1]-goal_pos_x)^2
        speed_multiplier = norm([[x[2], x[4]]])
        # cost = ((x_diff^2 + y_diff^2)*speed_multiplier + 0.01 / T * dist_to_goal) 
        cost = abs(h_x - x[3])^2*speed_multiplier + 0.005 / T * dist_to_goal
        # println(cost)
        return cost
    end

    # given robot desired traj function
    # y = r(x), x, y are scalars
    # r_prime_sym = Symbolics.derivative(r(x_sym), x_sym)
    # r_prime(x_t) = begin
    #     r_prime_val = Symbolics.value(Symbolics.substitute(r_prime_sym, x_sym => x_t))
    #     return r_prime_val
    # end
    traj_deviation_cost_robot_fn(si, x, us, t) = begin
        r_x = r(x[1])
        rp_x = r_prime(x[1]) 
        x_diff = rp_x * (x[3] - r_x) / (1 + rp_x^2)
        y_bar = r_x + rp_x^2 * (x[3] - r_x) / (1 + rp_x^2)
        y_diff = y_bar - x[3]
        # cost = (x_diff^2 + y_diff^2) 
        cost = abs(r_x - x[3])^2
        # println(cost)
        return cost
    end

    c1_traj = PlayerCost(traj_deviation_cost_human_fn, si)
    c2_traj = PlayerCost(traj_deviation_cost_robot_fn, si)

    ctrl_const = 0.2
    c1_control = QuadraticCost(zeros(4, 4))
    add_control_cost!(c1_control, 1, ctrl_const * diagm([1, 1]))
    add_control_cost!(c1_control, 2, zeros(2, 2)) # human doesn't care about robot's control cost

    c2_control = QuadraticCost(zeros(4, 4))
    add_control_cost!(c2_control, 1, ctrl_const * diagm([2, 2])) # robot should care about not making human pay too much control cost
    add_control_cost!(c2_control, 2, ctrl_const * diagm([1, 1]))

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