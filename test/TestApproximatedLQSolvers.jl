# Unit tests for LQ Nash solvers.
using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Random: seed!
using Test: @test, @testset
include("TestUtils.jl")

seed!(0)

# These tests run checks to ensure that for the LQ case given appropriate reference trajectories, the approximate
# linearized, quadraticized solutions are identical to those from the LQ solvers.
@testset "TestApproximatedLQSolvers" begin
    stackelberg_leader_idx = 1

    t0 = 0.0
    x₁ = [1., 0, 1, 0]
    horizon = 10

    # Ensure that for an LQ optimal control problem, both the LQ optimal control solution and approximate LQ Nash
    # optimal control with the solution to the LQ optimal control problem as reference, are identical!
    @testset "CheckLQOptCtrlAndApproximatedLQOptCtrlGiveSameOutputForLQGame" begin
        sys_info = SystemInfo(1, 4, [2])
        dyn = generate_random_linear_dynamics(sys_info)
        costs = generate_random_quadratic_costs(sys_info; include_cross_costs=true)

        Ps, Zs = solve_lqr_feedback(dyn, costs[1], horizon)
        xs, us = unroll_feedback(dyn, [Ps], x₁)
        P̃s, Z̃s = solve_approximated_lqr_feedback(dyn, costs[1], horizon, t0, xs, us[1])

        @test Ps == P̃s
        @test Zs == Z̃s
    end
end
