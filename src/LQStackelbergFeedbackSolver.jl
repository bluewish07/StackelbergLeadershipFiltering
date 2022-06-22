using LinearAlgebra

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

# Solve a finite horizon, discrete time LQ game to feedback Stackelberg equilibrium.
# Returns feedback matrices P[player][:, :, time]
export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyn::Dynamics, costs::AbstractArray{Cost}, horizon::Int, leader_idx::Int)

    # TODO: Add checks for correct input lengths - they should match the horizon.
    num_players = size(costs)[1]
    # horizon = size(dyn)[1]
    # TODO(hamzah) If we ever go beyond a 2-player game, figure out multiple followers.
    follower_idx = (leader_idx == 2) ? 1 : 2
    num_states = xdim(dyn)

    # Define control variables which are the same over all horizon.
    # TODO(hamzah) Alter all of these to be time-indexed and move to loop.
    A = dyn.A
    B_leader = dyn.Bs[leader_idx]
    B_follower = dyn.Bs[follower_idx]
    Q_leader = costs[leader_idx].Q
    Q_follower = costs[follower_idx].Q
    Rs = [costs[leader_idx].Rs[leader_idx], costs[leader_idx].Rs[leader_idx]]
    Rs[leader_idx] = costs[leader_idx].Rs[leader_idx]
    Rs[follower_idx] = costs[follower_idx].Rs[follower_idx]

    # TODO: Change the incentives later, but for now it's zero.
    R₁₂ = zeros(udim(dyn, leader_idx), udim(dyn, follower_idx))
    # This one can be 0.
    R₂₁ = zeros(udim(dyn, follower_idx), udim(dyn, leader_idx))

    # Define recursive variables and initialize variables.
    all_Ss = [zeros(udim(dyn, i), num_states, horizon) for i in 1:num_players]
    all_Ls = [zeros(num_states, num_states, horizon) for i in 1:num_players]
    all_Ls[leader_idx][:, :, horizon] = Q_leader
    all_Ls[follower_idx][:, :, horizon] = Q_follower

    # t will increment from 1 ... K-1. k will decrement from K-1 ... 1.
    for kk = horizon-1:-1:1

        # TODO(hamzah) When making the constants change each time step, put the definitions here.
        # Aₖ = ...
        Lₖ₊₁ = [all_Ls[leader_idx][:, :, kk+1], all_Ls[follower_idx][:, :, kk+1]]

        # 1. Compute Sₖ for each player.
        common_ctrl_cost_term = I + B_follower' * Lₖ₊₁[follower_idx] * B_follower

        # Attempt 1 - Basar
        G₁ = I + Lₖ₊₁[follower_idx] * B_follower * B_follower'
        G₂ = I + B_follower * B_follower' * Lₖ₊₁[follower_idx]
        G₃  = I + B_follower' * Lₖ₊₁[follower_idx] * B_follower
        H  = B_leader' * inv(G₁) * Lₖ₊₁[leader_idx] * inv(G₂) * B_leader
        J  = B_leader' * Lₖ₊₁[follower_idx]' * B_follower * inv(G₃) * R₁₂ * inv(G₃) * B_follower' * Lₖ₊₁[follower_idx] * B_leader
        M  = inv(G₁) * Lₖ₊₁[leader_idx] * G₂
        N  = Lₖ₊₁[follower_idx]' * B_follower * inv(G₃) * R₁₂ * inv(G₃) * B_follower' * Lₖ₊₁[follower_idx] 
        S1ₖ = inv(H * J + I) * B_leader' * (M + N) * A

        # Attempt 2 - self-derived with control costs.
        # G₃  = I + B_follower' * Lₖ₊₁[follower_idx] * B_follower
        # D₂ = inv(G₃) * B_follower' * Lₖ₊₁[follower_idx] * B_leader
        # F₁ = B_leader - B_follower * D₂
        # F₂ = I - B_follower * inv(G₃) * B_follower' * Lₖ₊₁[follower_idx]

        # add1 = D₂ * R₁₂ * inv(G₃) * B_follower' * Lₖ₊₁[follower_idx]
        # add2 = F₁' * Lₖ₊₁[leader_idx] * F₂
        # common_term = add1 + add2
        # S1ₖ = inv(Rs[leader_idx] + common_term * B_leader) * common_term * A

        all_Ss[leader_idx][:, :, kk] = S1ₖ


        # checked
        S2ₖ = (inv(common_ctrl_cost_term)
              * B_follower'
              * Lₖ₊₁[follower_idx]
              * (A - B_leader * S1ₖ))
        all_Ss[follower_idx][:, :, kk] = S2ₖ

        # 2. Compute feedback matrices Lₖ.
        ẋ = A - B_leader * S1ₖ - B_follower * S2ₖ

        all_Ls[leader_idx][:, :, kk] = ẋ' * Lₖ₊₁[leader_idx] * ẋ
                                       + S1ₖ' * S1ₖ
                                       + S2ₖ' * R₁₂ * S2ₖ
                                       + Q_leader

        all_Ls[follower_idx][:, :, kk] = ẋ' * Lₖ₊₁[follower_idx] * ẋ
                                         + S2ₖ' * S2ₖ
                                         + S1ₖ' * R₂₁ * S1ₖ
                                         + Q_follower
        # recurse!
    end

    return all_Ss, all_Ls
end
