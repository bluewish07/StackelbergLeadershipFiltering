using StackelbergControlHypothesesFiltering

using Random

seed = 1
rng = MersenneTwister(seed)

dt = 0.02
T = 501
horizon = T * dt
times = dt * (cumsum(ones(T)) .- 1)

cont_dyn = ShepherdAndSheepDynamics()
dyn = discretize(cont_dyn, dt)
ss_costs = ShepherdAndSheepCosts(dyn)
num_players = num_agents(dyn)


fs = get_as_function.(ss_costs)
pc_cost_1 = PlayerCost(fs[1], dyn.sys_info)
pc_cost_2 = PlayerCost(fs[2], dyn.sys_info)


leader_idx = 1
# Initial condition chosen randomly. Ensure both have relatively low speed.
x₁ = [2.; 0.; 1.; 0.; -1.; 0; 2; 0]
# x₁ = [1.; 0.; 0.01; 0.; -1.; 0; -0.01; 0]
# x₁ = rand(rng, 8)
x₁[[2, 4, 6, 8]] .= 0

# Initial controls
us_1 = [zeros(udim(dyn, ii), T) for ii in 1:num_agents(dyn)]
# for ii in 1:num_players
#     us_1[ii][1,:] .= -0.1
#     us_1[ii][2,:] .= -0.01
# end
# duration = (T-1) * dt
# us_1[ii][1, :] .= (xf[3] - x0[3]) / duration # omega
# us_1[ii][2, :] .= (xf[4] - x0[4]) / duration # accel

# Test with ideal solution, with noise.
# Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, ss_costs, T, leader_idx)
# _, us_1 = unroll_feedback(dyn, times, Ps_strategies, x₁)
# us_1[1] += 0.1 * rand(rng, udim(dyn, 1), T)
# us_1[2] += 0.1 * rand(rng, udim(dyn, 1), T)
