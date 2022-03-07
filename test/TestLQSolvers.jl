# Unit tests for LQ Nash solvers.

using StackelbergControlHypothesesFiltering
using Test: @test, @testset
using Random: seed!

seed!(0)

@testset "TestLQSolvers" begin
    stackelberg_leader_idx = 1

    # Common dynamics, costs, and initial condition.
    A = [1 0.1 0 0;
         0 1   0 0;
         0 0   1 1.1;
         0 0   0 1]
    B₁ = [0 0.1 0 0]'
    B₂ = [0 0   0 0.1]'
    dyn = Dynamics(A, [B₁, B₂])

    Q₁ = [0 0 0  0;
          0 0 0  0;
          0 0 1. 0;
          0 0 0  0]
    c₁ = Cost(Q₁)
    add_control_cost!(c₁, 1, ones(1, 1))

    Q₂ = [1.  0 -1 0;
          0  0 0  0;
          -1 0 1  0;
          0  0 0  0]
    c₂ = Cost(Q₂)
    add_control_cost!(c₂, 2, ones(1, 1))

    costs = [c₁, c₂]

    x₁ = [1., 0, 1, 0]
    horizon = 10


    # Ensure that the feedback solution satisfies Nash conditions of optimality
    # for each player, holding others' strategies fixed.
    @testset "CheckFeedbackSatisfiesNash" begin
        Ps = solve_lq_nash_feedback(dyn, costs, horizon)
        xs, us = unroll_feedback(dyn, Ps, x₁)
        nash_costs = [evaluate(c, xs, us) for c in costs]

        # Perturb each strategy a little bit and confirm that cost only
        # increases for that player.
        ϵ = 1e-1
        for ii in 1:2
            for tt in 1:horizon
                P̃s = deepcopy(Ps)
                P̃s[ii][:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))

                x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)
                new_nash_costs = [evaluate(c, x̃s, ũs) for c in costs]
                @test new_nash_costs[ii] ≥ nash_costs[ii]
            end
        end
    end


    # Ensure that the feedback solution satisfies Stackelberg conditions of optimality
    # for each player, holding others' strategies fixed.
    @testset "CheckFeedbackSatisfiesStackelberg" begin
        Ss, Ls = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, Ss, x₁)
        optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

        # Define some useful constants.
        ϵ = 1e-1
        leader_idx = stackelberg_leader_idx
        follower_idx = 3 - stackelberg_leader_idx
        num_players = follower_idx

        for tt in 1:horizon-1

            # Copy the things we will alter.
            ũs = deepcopy(us)

            # Perturb the leaduer input u1 at the current time.
            ũs[leader_idx][:, tt] += ϵ * randn(udim(dyn, leader_idx))

            # Resolve for the follower input at the current time.
            B₂ = dyn.Bs[follower_idx]
            G = costs[follower_idx].Rs[follower_idx] + B₂' * Ls[follower_idx][:, :, tt+1] * B₂
            ũ1ₜ = ũs[leader_idx][:, tt]
            ũ2ₜ = - inv(G) * B₂' * Ls[follower_idx][:, :, tt+1] * (dyn.A * xs[:,tt] + B₁ * ũ1ₜ)
            ũs[follower_idx][:, tt] = ũ2ₜ

            # The cost computed as x_t^T * L^1_tt x_t for at time tt - not used in the test.
            # TODO(hamzah) - remove after figuring out discrepancy between costs.
            opt_P1_cost_1 = xs[:, tt]' * Ls[leader_idx][:, :, tt] * xs[:, tt]

            # The cost computed manually for optimal inputs using
            # x_t^T Q_t x_t^T + ... + <control costs> + ... + x_{t+1}^T * L^1_{t+1} x_{t+1}.
            # TODO(hamzah) - remove from loop
            u1ₜ = us[leader_idx][:, tt]
            u2ₜ = us[follower_idx][:, tt]
            opt_P1_cost = xs[:, tt]' * costs[leader_idx].Q * xs[:, tt]
            opt_P1_cost += u1ₜ' * costs[leader_idx].Rs[leader_idx] * u1ₜ
            if haskey(costs[leader_idx].Rs, follower_idx)
                opt_P1_cost += u2ₜ' * costs[leader_idx].Rs[follower_idx] * u2ₜ
            end
            xₜ₊₁ = xs[:, tt+1] #dyn.A * xs[:, tt] + dyn.Bs[leader_idx] * u1ₜ + dyn.Bs[follower_idx] * ũ2ₜ
            opt_P1_cost += xₜ₊₁' * Ls[leader_idx][:, :, tt+1] * xₜ₊₁

            # The cost computed manually for perturbed inputs using
            # x_t^T Q_t x_t^T + ... + <control costs> + ... + x_{t+1}^T * L^1_{t+1} x_{t+1}.
            new_P1_cost = xs[:, tt]' * costs[leader_idx].Q * xs[:, tt]
            new_P1_cost += ũ1ₜ' * costs[leader_idx].Rs[leader_idx] * ũ1ₜ
            if haskey(costs[leader_idx].Rs, follower_idx)
                new_P1_cost += ũ2ₜ' * costs[leader_idx].Rs[follower_idx] * ũ2ₜ
            end
            x̃ₜ₊₁ = dyn.A * xs[:, tt] + dyn.Bs[leader_idx] * ũ1ₜ + dyn.Bs[follower_idx] * ũ2ₜ
            new_P1_cost += x̃ₜ₊₁' * Ls[leader_idx][:, :, tt+1] * x̃ₜ₊₁

            println(tt, " - one step + future value - ", new_P1_cost, " ", opt_P1_cost, " ", opt_P1_cost_1)
            @test new_P1_cost ≥ opt_P1_cost

            x̃s = unroll_raw_controls(dyn, ũs, x₁)
            new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
            optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]
            println(tt, " - entire traj evaluated - ", new_stack_costs[leader_idx], " ", optimal_stackelberg_costs[leader_idx])
            @test new_stack_costs[leader_idx] ≥ optimal_stackelberg_costs[leader_idx]
        end

        # Perturb each optimized P2 strategy a little bit and confirm that cost only increases for player 2.
        leader_idx = 2
        follower_idx = 1
        for tt in 1:horizon
            P̃s = deepcopy(Ss)
            P̃s[leader_idx][:, :, tt] += ϵ * randn(udim(dyn, leader_idx), xdim(dyn))

            x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)
            new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
            @test new_stack_costs[leader_idx] ≥ optimal_stackelberg_costs[leader_idx]
        end

    end

end
