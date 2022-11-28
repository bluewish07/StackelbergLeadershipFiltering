using LinearAlgebra

export compute_stackelberg_recursive_step
function compute_stackelberg_recursive_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21)
    """
    A helper which accepts all of the inputs needed for computing the Stackelberg recursion and produces the result of
    one recursive step.

    Returns S1, S2, L1, L2 at the current timestep.
    """
    Dₜ = (R22 + B2' * L2ₜ₊₁ * B2) \ (B2' * L2ₜ₊₁)

    lhs = R11 + B1' * Dₜ' * R12 * Dₜ * B1 + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (B1 - B2 * Dₜ * B1)
    rhs = ((B1' * Dₜ' * R12 * Dₜ) + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (I - B2 * Dₜ)) * A

    S1ₜ = lhs \ rhs
    S2ₜ = Dₜ * (A - B1 * S1ₜ)

    dynamics_tp1 = A - B1 * S1ₜ - B2 * S2ₜ
    L1 = Q1 + S1ₜ' * R11 * S1ₜ + S2ₜ' * R12 * S2ₜ + dynamics_tp1' * L1ₜ₊₁ * dynamics_tp1
    L2 = Q2 + S1ₜ' * R21 * S1ₜ + S2ₜ' * R22 * S2ₜ + dynamics_tp1' * L2ₜ₊₁ * dynamics_tp1
    return [S1ₜ, S2ₜ, L1, L2]
end


# Solve a finite horizon, discrete time LQ game to feedback Stackelberg equilibrium.
# Returns feedback matrices P[player][:, :, time]
export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyn::LinearDynamics, costs::AbstractArray{QuadraticCost}, horizon::Int, leader_idx::Int)

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

    R₁₁ = costs[leader_idx].Rs[leader_idx]
    R₂₂ = costs[follower_idx].Rs[follower_idx]
    R₁₂ = costs[leader_idx].Rs[follower_idx]
    R₂₁ = costs[follower_idx].Rs[leader_idx]

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

        # Run recursive computation for one step.
        outputs = compute_stackelberg_recursive_step(A, B_leader, B_follower, Q_leader, Q_follower, Lₖ₊₁[leader_idx], Lₖ₊₁[follower_idx], R₁₁, R₂₂, R₁₂, R₂₁)

        all_Ss[leader_idx][:, :, kk] = outputs[1]
        all_Ss[follower_idx][:, :, kk] = outputs[2]
        all_Ls[leader_idx][:, :, kk] = outputs[3]
        all_Ls[follower_idx][:, :, kk] = outputs[4]
    end

    return all_Ss, all_Ls
end
