using LinearAlgebra

# Solve a finite horizon, discrete time LQR problem.
# Returns feedback matrices P[:, :, time].

# Shorthand function for LTI dynamics and costs.
export solve_lqr_feedback
function solve_lqr_feedback(
    dyn::Dynamics, costs::Cost, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    costs = [costs for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

# TODO(hamzah): Add interfaces for cases in which one of the arguments is passed in as a list, but the other is not.

export solve_lqr_feedback
function solve_lqr_feedback(dyns::AbstractArray{Dynamics}, costs::AbstractArray{Cost}, horizon::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert(ndims(dyns) == 1 && size(dyns, 1) == horizon)
    @assert(ndims(costs) == 1 && size(costs, 1) == horizon)

    # Note: There should only be one "player" for an LQR problem.
    num_states = xdim(dyns[1])
    num_ctrls = udim(dyns[1], 1)

    Ps = zeros((num_ctrls, num_states, horizon))
    Zₜ₊₁ = costs[horizon].Q

    # base case
    if horizon == 1
        return Ps
    end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:-1:1

        A = dyns[tt].A
        Q = costs[tt].Q
        B = dyns[tt].Bs[1]
        R = costs[tt].Rs[1]

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Zₜ₊₁ * B

        # This is equivalent to inv(r_terms) * B' * Zₜ₊₁ * A
        Ps[:, :, tt] = r_terms \ B' * Zₜ₊₁ * A
        
        # Update Zₜ₊₁ at t+1 to be the one at t as we go to t-1.
        Zₜ₊₁ = Q + A' * Zₜ₊₁ * A - A' * Zₜ₊₁ * B * Ps[:, :, tt]
    end

    return Ps
end