using StackelbergControlHypothesesFiltering

using Plots

include("SimpleKinematic1DDynamics.jl")


# Specify particle filter details.
Ns = 100

# Specify the initial discrete state prior distribution.
p_init = 0.9
s_prior_distrib = Bernoulli(p_init)

# Specify the process noise.
v_bar = zeros(2)
Q = Diagonal([1.0^2, 0.5^2])
process_noise_distribution = MvNormal(v_bar, Q)

# Integrate process noise into the dynamics.
gen_process_noise(rng, num_samples) = rand(rng, process_noise_distribution, num_samples)
f_dynamics_1(t_range, x, u, rng) = f₁(t_range, x, u, gen_process_noise(rng, size(x, 2)))
f_dynamics_2(t_range, x, u, rng) = f₂(t_range, x, u, gen_process_noise(rng, size(x, 2)))
f_dynamics = [f_dynamics_1, f_dynamics_2]

# set controls
u_inputs = zeros(1, num_data) # inputs should always be 1 or 0

x̂✶_PF, P_PF, z̄_PF, P̄_zz_PF, ϵ_bar, ϵ_hat, N̂s_n, s_PF, s_probs, particles = 
    two_state_PF(x̄_prior,P_prior,u_inputs,s_prior_distrib,t,t0,z,R,discrete_state_transition,f_dynamics,[h₁, h₂],
                 ;seed=seed,Ns=Ns)

x̂✶ = permutedims(x̂✶_PF,(2,1))
P = permutedims(P_PF,(3,2,1))
z̄ = permutedims(z̄_PF,(2,1))
P̄_zz = permutedims(P̄_zz_PF,(3,2,1))

# 1. plot velocity signals
s=1
p1 = scatter(t, particles[1, :, :][:, :]', color=:black, markersize=0.25, label="")
p1 = plot!(t, x_true[:, 1], label="true vel.", color=:blue)
p1 = plot!(t, z[:, 1], label="meas.", color=:green)
p1 = plot!(t, x̂✶[:, 1], label="est. vel.", color=:red)

# 2. plot position signals
s=2
p2 = scatter(t, particles[2, :, :][:, :]', color=:black, markersize=0.25, label="")
p2 = plot!(t, x_true[:, 2], label="true pos.", color=:blue)
p2 = plot!(t, z[:, 2], label="meas.", color=:green)
p2 = plot!(t, x̂✶[:, 2], label="est. pos.", color=:red)

# 3. compute expected probabilities and plot discrete state signals
expected_probs = zeros(2, num_data)
expected_probs[:, 1] = [p_init; 1-p_init]
for i in 2:num_data
    expected_probs[:, i] = disc_state_matrix * expected_probs[:, i-1]
end

s=3
p3 = plot(t, expected_probs[1, :], label="expected", title="Probability of being in state 1 (positive acceleration)")
p3 = plot!(t, s_probs', label="actual")

display(plot(p1,p2,p3,layout=(3,1),size=(800,800),
        plot_title="1D kinematics system velocity and position with state"))

# # RMS errors
# errors = x̂✶ - x_true
# # errors = x_true
# position_rms_error = √(sum([errors[k,1]^2 for k in 1:ℓ])/ℓ)
# velocity_rms_error = √(sum([errors[k,2]^2 for k in 1:ℓ])/ℓ)

# println("RMS Errors - Normal Distribution Process Noise Assumption")
# println("Position RMS error (m): ",position_rms_error)
# println("Velocity RMS error (m/s): ",velocity_rms_error)
# println("")

# # error plots
# s=1
# p1 = plot(t, errors[:,s],
#             linecolor=:gray,ylims=ylims_pos,
#             title="Position error (m)",label="Error")
# p1 = plot!(t, 3 .* .√P[:,s,s],label="3σ",linestyle=:dash,linecolor=:green)
# p1 = plot!(t, -3 .* .√P[:,s,s],label="",linestyle=:dash,linecolor=:green)

# s=2
# p2 = plot(t, errors[:,s],
#             linecolor=:gray,ylims=ylims_vel,
#             title="Velocity error (m/s)",label="Error",
#             xlabel="Time (sec)",legend=false)
# p2 = plot!(t, 3 .* .√P[:,s,s],label="3σ",linestyle=:dash,linecolor=:green)
# p2 = plot!(t, -3 .* .√P[:,s,s],label="",linestyle=:dash,linecolor=:green)

# display(plot(p1,p2,layout=(2,1),size=(800,800),
#         plot_title="Errors - Normal Distribution Process Noise Assumption"))
