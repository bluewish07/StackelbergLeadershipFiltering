# Utilities for managing quadratic and nonquadratic dynamics.
abstract type Cost end

# Every Cost is assumed to have the following functions defined on it:
# - quadraticize_costs(cost, t, x, us) - this function produces a QuadraticCost at time t given the state and controls
# - evaluate(cost, xs, us) - this function evaluates the cost of a trajectory given the states and controls

# We use this type by making a substruct of it which can then have certain functions defined for it.
abstract type NonQuadraticCost <: Cost end

# Cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct QuadraticCost <: Cost
    Q
    Rs
end
QuadraticCost(Q) = QuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}())

# TODO(hamzah) Add better tests for the QuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::QuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

function quadraticize_costs(cost::QuadraticCost, t, x, us)
    return cost
end

# Evaluate cost on a state/control trajectory.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::QuadraticCost, xs, us)
    horizon = last(size(xs))

    total = 0.0
    for tt in 1:horizon
        total += xs[:, tt]' * c.Q * xs[:, tt]
        total += sum(us[jj][:, tt]' * Rij * us[jj][:, tt] for (jj, Rij) in c.Rs)
    end
    return total
end


# TODO: Make the affine cost structure with homogenenized coordinates.
# struct AffineCost <: Cost end
# function quadraticize_costs(cost::AffineCost, t, x, us)
# end

# Export all the cost types/structs.
export Cost, NonQuadraticCost, QuadraticCost

# Export all the cost types/structs.
export quadraticize_costs, evaluate

