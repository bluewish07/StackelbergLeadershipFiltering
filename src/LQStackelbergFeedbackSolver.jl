# A helper function to compute P for all players at time t.
function compute_L_at_t(dyn_at_t::Dynamics, costs_at_t, Zₜ₊₁)

    num_players = size(costs_at_t)[1]
    num_states = xdim(dyn_at_t)
    A = dyn_at_t.A
    lhs_rows = Array{Float64}(undef, 0, num_states ÷ num_players)

    for player_idx in 1:num_players

        # Identify terms.
        B = dyn_at_t.Bs[player_idx]
        Rⁱⁱ = costs_at_t[player_idx].Rs[player_idx]

        # Compute terms for the matrices. First term is (*) in class notes, second is (**).
        first_term = Rⁱⁱ + B' *  Zₜ₊₁[player_idx] * B
        sum_of_other_player_control_matrices = sum(dyn_at_t.Bs) - B
        second_term = B' * Zₜ₊₁[player_idx] * sum_of_other_player_control_matrices

        # Create the LHS rows for the ith player and add to LHS rows.
        lhs_ith_row  = hcat([(i == player_idx) ? first_term : second_term for i in 1:num_players]...)
        lhs_rows = vcat(lhs_rows, lhs_ith_row)
    end

    # Construct the matrices we will use to solve for P.
    lhs_matrix = lhs_rows
    rhs_matrix_terms = [dyn_at_t.Bs[i]' * Zₜ₊₁[i] * A for i in 1:num_players]
    rhs_matrix = vcat(Array{Float64}(undef, 0, num_states), rhs_matrix_terms...)

    # Finally compute P.
    return lhs_matrix \ rhs_matrix
end

# TODO(hamzah) Implement an LQ Stackelberg game.

# Solve a finite horizon, discrete time LQ game to feedback Nash equilibrium.
# Returns feedback matrices P[player][:, :, time]
export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyn::Dynamics, costs::AbstractArray{Cost}, horizon::Int)

    num_players = size(costs)[1]

    # 1. Start at t=T, setting Z^i_t = Q^i_t.
    t = horizon
    Zₜ₊₁ = [costs[i].Q for i in 1:num_players]
    Zₜ = [zeros(size(Zₜ₊₁[i])) for i in 1:num_players]

    num_states = xdim(dyn)
    all_Ps = [zeros(udim(dyn, i), num_states, horizon) for i in 1:num_players]

    while t > 1
        # 2. Decrement t, compute P^{i*}_t and Z^i_t.
        t -= 1

        # Compute Ps for all players at time t and store them.
        Ps = compute_L_at_t(dyn, costs, Zₜ₊₁)
        for i in 1:num_players
            num_inputs = udim(dyn, i)
            index_range = (i-1) * num_inputs + 1 : i * num_inputs

            all_Ps[i][:, :, t] = reshape(Ps[index_range, :], (num_inputs, xdim(dyn), 1))
        end

        for i in 1:num_players
            # Extract other values.
            Qₜ = [costs[i].Q for i in 1:num_players]
            Pⁱₜ = all_Ps[i][:, :, t]

            # Compute Z terms. There are no nonzero off diagonal Rij terms, so we just need to compute the terms with Rii.
            summation_1_terms = [Pⁱₜ' * costs[i].Rs[i][1] * Pⁱₜ]
            summation_1 = sum(summation_1_terms)

            summation_2_terms = [dyn.Bs[j] * all_Ps[j][:, :, t] for j in 1:num_players]
            summation_2 = dyn.A - sum(summation_2_terms)

            Zₜ[i] = Qₜ[i] - summation_1 + summation_2' * Zₜ₊₁[i] * summation_2
        end

        # Update Z_{t+1}
        Zₜ₊₁ = Zₜ

        # 3. Go to (2) until t=1.
    end

    return all_Ps
end
