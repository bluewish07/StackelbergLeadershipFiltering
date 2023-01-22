using LinearAlgebra

# Set the random seed.
seed = 0

# Generative configuration - the data starts at an initial state (t0, v0, x0) and proceeds with 1D kinematic dynamics.
#                            At each time step, the generated data includes the state which dictates the direction of
#                            acceleration.

# Set parameters for data generation.
dt = 0.01
num_data = 501
num_cols = 5

# The true generative probability of being in state 1 - used to generate the truth data.
true_prob_1 = 0.9

# Acceleration magnitude for both state 1 and 2 (which are in opposite directions). Used to generate true data and dynamics.
accel_magnitude = 0.1

# Set the initial state manually.
t0 = 0.0
init_direction = 1.0
v0 = init_direction * 0.0 # true initial velocity point away from acceleration
x0 = 0.0                  # true initial position

# Define the measurement distribution uncertainty - used for
#   1. both generating the measurement data AND
#   2. the measurement covariance.
pos_meas_stdev = 0.2
vel_meas_stdev = 0.1

# Markov chain state transition probabilities - used to define the discrete state transition.
p₁₁ = 0.4
p₂₂ = 1.0

# Define prior state gaussian parameters
x̄_prior = [v0, x0]                  # mean of the prior
P_prior = Diagonal([0.1^2, 0.5^2])  # covariance of the prior
