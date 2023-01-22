######## File: SimpleKinematics1DDynamics.jl ########
using Distributions
using Random

include("ConfigSimpleKinematic1DDynamics.jl")


# random number generator
rng = MersenneTwister(seed)

# Generate data going right.
data = zeros(num_data, num_cols)
data[1, 1] = t0
data[1, 4] = v0
data[1, 5] = x0

a₁ = accel_magnitude
a₂ = -accel_magnitude
is_1 = rand(rng, Bernoulli(true_prob_1), num_data)
accel = zeros(num_data)
accel[is_1] .= a₁
accel[.!is_1] .= a₂

for i in 2:num_data
    time = t0 + (i-1) * dt
    data[i, 1] = time
    data[i, 4] = accel[i] * dt + data[i-1, 4]
    true_pos   = accel[i] * dt^2 + data[i-1, 4] * dt + data[i-1, 5]
    data[i, 5] = true_pos
end

# Measurement distribution
true_R = Diagonal([vel_meas_stdev^2, pos_meas_stdev^2])
eps_distrib = MvNormal(zeros(2), true_R)

# make measurements
data[:, 2:3] = data[:, 4:5] + rand(rng, eps_distrib, num_data)'


t = data[:,1]           # times
z = data[:,2:3]         # measurements
x_true = data[:,4:5]    # true values

ℓ = size(z)[1]          # number of measurements

t0 = t[1]               # time of first measurement

R = true_R              # measurement error covariance

## Define functions
function generate_transition_matrices(a)
    # State transition matrix
    Φ(tᵢ,t₀) = [ exp(tᵢ - t₀)                            0;
                (1/2) * (tᵢ - t₀)^2 * exp(tᵢ - t₀)  exp(tᵢ - t₀)]

    # Control transition matrix
    G(tᵢ,t₀) = a * [ exp(tᵢ - t₀) - (tᵢ - t₀) - 1;
                    (1/2) * (exp(tᵢ - t₀) * (((tᵢ - t₀) - 4) * (tᵢ - t₀) + 8) - (((tᵢ - t₀) + 4) * (tᵢ - t₀)) - 8)][:,:]

    # Process noise transition matrix

    # for only position
    # Γ(tᵢ,t₀) = [ 0; exp(tᵢ - t₀) - 1]

    # For velocity and position
    Γ(tᵢ,t₀) = [ exp(tᵢ - t₀) - 1  0 ; 0  exp(tᵢ - t₀) - 1]

    return Φ, G, Γ
end

function generate_measurement_models()
    # measurement functions
    h(x) = x
    H(x) = [1 0; 0 1]
    return h, H
end

function generate_dynamics(a)
    Φ, G, Γ = generate_transition_matrices(a)
    function f(t_range, x, u, v)
        t0 = t_range[1]
        t = t_range[2]
        @assert t ≥ t0

        return Φ(t, t0) * x + G(t, t0) * u + Γ(t, t0) * v
    end

    A(t) = [1 0; t 1]
    B(t) = a * [t ; 0.5 * t^2][:,:]
    D(t) = [1 0; 0 1]
    function f_dynamics(t_range, x, u, v)
        t0 = t_range[1]
        t = t_range[2]
        @assert t ≥ t0

        dt = t - t0
        dyn = LinearDynamics(A(dt), B(dt), D(dt))

        # return A(dt) * x + B(dt) * u + D(dt) * v[:]
        return propagate_dynamics(dyn, t_range, x, u, v)
    end
    return f
    # return f_dynamics
end

# TODO(hamzah) - vectorize this better
function generate_discrete_state_transition(p₁₁, p₂₂)

    distribs = [Bernoulli(p₁₁), Bernoulli(p₂₂)]

    # state transition matrix of state
    P = [ p₁₁  1-p₂₂;
         1-p₁₁  p₂₂]

    # The discrete state transition stays in state i with probability pᵢ.
    function discrete_state_transition(time_range, s_prev, 𝒳_prev, s_actions, rng)
        @assert length(s_prev) == 1
        s_prev = s_prev[1]
        sample = rand(rng, distribs[s_prev], 1)

        other_state = (s_prev == 1) ? 2 : 1
        s_new = (isone(sample[1])) ? s_prev : other_state

        return [s_new]
    end
    return discrete_state_transition, P
end

# Generate the discrete state transition
discrete_state_transition, disc_state_matrix = generate_discrete_state_transition(p₁₁, p₂₂)

# Generate dynamics and measurement model for state 1 (from homework 5)
# Φ₁, G₁, Γ₁ = generate_transition_matrices(a₁)
# f₁ = generate_dynamics(Φ₁, G₁, Γ₁)
f₁ = generate_dynamics(a₁)

h₁, H₁ = generate_measurement_models()

# Generate dynamics and measurement model for state 2 (from final exam)
# Φ₂, G₂, Γ₂ = generate_transition_matrices(a₂)
# f₂ = generate_dynamics(Φ₂, G₂, Γ₂)
f₂ = generate_dynamics(a₂)
h₂, H₂ = generate_measurement_models()

# saturation causes Nans?
