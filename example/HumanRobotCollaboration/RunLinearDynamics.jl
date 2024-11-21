using StackelbergControlHypothesesFiltering

using Dates
using LinearAlgebra: norm, Diagonal, I, diagm
using Plots
using ProgressBars
using Random: MersenneTwister
using Distributions: Bernoulli, MvNormal
using CSV, Tables

include("GroundTruthGenHRI.jl")
include("CreateHRCGame.jl")

# Two player dynamics
# Dynamics (Euler-discretized double integrator equations with Δt = 0.1s).
# State is shared between two players and is layed out as [x, ẋ, y, ẏ].
# P1 is human, P2 is robot

# Define game and timing related configuration.
num_players = 2

T =421
t0 = 0.0
dt = 0.01
horizon = T * dt
times = dt * cumsum(ones(2*T)) .- dt

# define limits for plots
limits = [-5., 120.]
limits_tuple = tuple(limits...)

# Defined the dynamics of the game.
# let's do 2D for now
A() = [0 1.  0  0;
        0  0  0  0;
        0  0  0 1.;
        0  0  0  0]
B1() = [0   0;
        1   0;
        0   0;
        0   1]
B2() = [0   0;
        1   0;
        0   0;
        0   1]
cont_lin_dyn = ContinuousLinearDynamics(A(), [B1(), B2()])
dyn = discretize(cont_lin_dyn, dt)
si = dyn.sys_info


# Generate a ground truth trajectory on which to run the leadership filter for a merging trajectory.
# u_refs, x1 = get_simple_straight_line_2D_traj()
# x_refs = unroll_raw_controls(dyn, times[1:T], u_refs, x1)

h(x) = 0.1*x + exp(-(x-3)^2)
r(x) = 0.1*x  
h_prime(x) = 0.1 - (2*x-6)-exp(-(x-3)^2)
r_prime(x) = 0.1 
x1, x_refs, u_refs = get_ground_truth_traj(dyn, times[1:T])
h_refs = [h(s) for s in x_refs[1, :]]
r_refs = [r(s) for s in x_refs[1, :]]
traj_plot = plot_trajectory(dyn, times[1:T], x_refs, h, r)
grnd_truth_plt = plot(traj_plot, size=(1000, 500))
display(grnd_truth_plt)

# create some records for csv
state_records = hcat(x_refs[1, :], x_refs[3, :]) 
state_records = hcat(state_records, vec(h_refs), vec(r_refs))
state_records = hcat(state_records, u_refs[1]', u_refs[2]')


# Define simple quadratic costs for the agents.
GetCosts(dyn::Dynamics; ctrl_const=0.1) = begin
    Q1 = zeros(4, 4)
    Q1[1, 1] = 0.0001
    Q1[3, 3] = 1
    Q1[1, 3] = -0.01
    Q1[3, 1] = -0.01
    C1 = QuadraticCost(Q1)
    add_control_cost!(C1, 1, ctrl_const * diagm([1, 1]))
    add_control_cost!(C1, 2, zeros(2, 2))

    Q2 = zeros(4, 4)
    Q2[3, 3] = 1.
    Q2[4, 4] = 0
    C2 = QuadraticCost(Q2)
    add_control_cost!(C2, 2, ctrl_const * diagm([0, 1]))
    add_control_cost!(C2, 1, ctrl_const * diagm([2, 1]))

    return [C1, C2]
end

# costs = GetCosts(dyn)
costs = create_HRC_costs(T, x_refs[1, T], h, r, h_prime, r_prime)


# Run the leadership filter.

# Initial condition chosen randomly. Ensure both have relatively low speed.
pos_unc = 1e-3
vel_unc = 1e-4
P1 = Diagonal([pos_unc, pos_unc, pos_unc, vel_unc])

# Process noise uncertainty
Q = 0.5*1e-2 * Diagonal([1e-2, 1e-3, 1e-2, 1e-3])


# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 1e-3 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 150

p_transition = 0.9
p_init = 0.4

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
s_init_distrib = Bernoulli(p_init)

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


# x_refs = xs_k
# us_refs = us_k

# Augment the remaining states so we have T+Ts-1 of them.
true_xs = hcat(x_refs, zeros(xdim(dyn), Ts-1))
true_us = [hcat(u_refs[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(true_xs[:, tt], R))
end

# l = @layout [a{0.3h}; grid(2, 3)]

# pos_plot, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times[1:T], rotate_state(dyn, x_refs[:, 1:T]), us_refs)

# plot_zs = rotate_state(dyn, zs)
# scatter!(pos_plot, plot_zs[1, :], plot_zs[2, :], color=:turquoise, label="", ms=0.5)
# scatter!(pos_plot, plot_zs[5, :], plot_zs[6, :], color=:orange, label="", ms=0.5)

# plot(pos_plot, p2, p3, p4, p5, p6, p7, layout=l)


threshold = 2*1e-2
max_iters = 100
step_size = 1e-2

x̂s, P̂s, probs, pf, sg_objs, iter_timings = leadership_filter(dyn, costs, t0, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x1,        # initial state at the beginning of simulation
                           P1,        # initial covariance at the beginning of simulation
                           u_refs,   # the control inputs that the actor takes
                           zs,        # the measurements
                           1.1 * R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           # check_valid=check_valid,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=true,
                           ensure_pd=false)

state_records = hcat(state_records, vec(probs[1:T]))
println(size(state_records))

using Dates
gr()

# Create the folder if it doesn't exist
folder_name = "HRC_LQ_$(get_date_str())"
isdir(folder_name) || mkdir(folder_name)

savefig(grnd_truth_plt, joinpath(folder_name, "ground_truth.pdf"))

header = ["x", "y", "y_hdes", "y_rdes", "u1_x", "u1_y", "u2_x", "u2_y", "u1_lead_prob"]
CSV.write(joinpath(folder_name, "viz.csv"), Tables.table(state_records), writeheader=true, header=header)

# Generate the plots for the paper.
snapshot_freq = Int((T - 1)/20)
# Generate a probability plot no timings.
prob_plot = make_probability_plots(times[1:T], probs[1:T])
plot!(prob_plot, title="")
prob_filepath = joinpath(folder_name, "HRC_LQ_scenario_probs.pdf")
savefig(prob_plot, prob_filepath)

p1a = plot_leadership_filter_positions_shared(dyn, true_xs[:, 1:T], x̂s[:, 1:T])
pos_main_filepath = joinpath(folder_name, "LF_merging_scenario_main.pdf")
savefig(p1a, pos_main_filepath)

iter1 = ProgressBar(2:snapshot_freq:T)
global p_i = 1
for t in iter1
    p1b = plot_leadership_filter_measurement_details_shared(num_particles, sg_objs[t], true_xs[:, 1:T], x̂s; include_all_labels=true, t=t)
    pos2_filepath = joinpath(folder_name, "0$(p_i)_LF_merging_scenario_positions_detail.pdf")
    savefig(p1b, pos2_filepath)
    global p_i += 1
end






# make_merging_scenario_pdf_plots(folder_name, snapshot_freq, cfg, limits, sg_objs[1].dyn, T, times, true_xs, true_us, probs, x̂s, zs, num_particles)
