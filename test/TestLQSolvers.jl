# Unit tests for LQ Nash solvers.

using StackelbergControlHypothesesFiltering
using Test: @test, @testset
using Random: seed!

seed!(0)

# TODO(hamzah) The Stackelberg test code currently implements the Nash check. Implement the
#              Stackelberg check instead, which requires an optimal control solver.
function solve_optimal_control()
    return
end

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



    # TODO(hamzah) This code currently implements the Nash check. Implement the Stackelberg check instead.
    # TODO(hamzah) Run the implemented stackelberg game.


    # Ensure that the feedback solution satisfies Stackelberg conditions of optimality
    # for each player, holding others' strategies fixed.
    @testset "CheckFeedbackSatisfiesStackelberg" begin
        Ss = solve_lq_stackelberg_feedback(dyn, costs, horizon, stackelberg_leader_idx)
        xs, us = unroll_feedback(dyn, Ss, x₁)
        optimal_stackelberg_costs = [evaluate(c, xs, us) for c in costs]

        # Perturb each P1 strategy a little bit, resolve for P2's input, and ensure the optimal cost for P1 is less than
        # the alterred cost for P1.
        ϵ = 1e-1
        ii = 1
        jj = 2
        num_players = jj
        x̃s = zeros(xdim(dyn), horizon)
        x̃s[:, 1] = x₁
        x̃s[:, 2] = xs[:, 2]
        ũs = [zeros(size(us[jj])) for i in 1:num_players]
        ũs[1][:, 1] = us[1][:, 1]
        ũs[2][:, 1] = us[2][:, 1]

        P̃2s = deepcopy(Ss)[jj]

        # At each horizon, perturb P1 at tt and then resolve for the game from tt until horizon for P2 at tt and
        # x at tt. Then, compute and store the u2 and P2 up to tt.
        for tt in 2:horizon

            # Reset P1s to the optimal P1s and add noise to P1's feedback matrices at tt.
            P̃1s = deepcopy(Ss)[ii]
            P̃1s[:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))

            # Rerun the Stackelberg solver from tt until horizon. Extract the P2 feedback matrices and disregard the
            # those for P1. Note that the first entry in this is for the current time tt.
            P̃2s[:, :, tt:horizon] = solve_lq_stackelberg_feedback(dyn, costs, horizon - tt + 1, stackelberg_leader_idx)[jj]

            # Unroll the states again from the initial state. Note that the states/controls after tt are meaningless
            # since we have not recomputed those yet.
            Ps = [P̃1s, P̃2s]
            partial_xs, partial_us = unroll_feedback(dyn, Ps, x̃s[:, 1])

            # Store the controls and new state at tt.
            x̃s[:, tt] = partial_xs[:, tt]
            ũs[ii][:, tt] = partial_us[ii][:, tt]
            ũs[jj][:, tt] = partial_us[jj][:, tt]

            # Validate the new costs are higher than the optimal ones for the trajectory up to tt.
            ũs_until_tt = [ũs[ii][:, 1:tt], ũs[jj][:, 1:tt]]
            us_until_tt = [us[ii][:, 1:tt], us[jj][:, 1:tt]]
            new_stack_costs_until_tt = [evaluate(c, x̃s[:, 1:tt], us_until_tt) for c in costs]
            optimal_stackelberg_costs_until_tt = [evaluate(c, xs[:, 1:tt], us_until_tt) for c in costs]
            @test new_stack_costs_until_tt[ii] ≥ optimal_stackelberg_costs_until_tt[ii]
        end

        # Perturb each optimized P2 strategy a little bit and confirm that cost only increases for player 2.
        ii = 2
        jj = 1
        for tt in 1:horizon
            P̃s = deepcopy(Ss)
            P̃s[ii][:, :, tt] += ϵ * randn(udim(dyn, ii), xdim(dyn))

            x̃s, ũs = unroll_feedback(dyn, P̃s, x₁)
            new_stack_costs = [evaluate(c, x̃s, ũs) for c in costs]
            @test new_stack_costs[ii] ≥ optimal_stackelberg_costs[ii]
        end

    end

end
