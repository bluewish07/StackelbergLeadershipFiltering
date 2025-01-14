using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("params_time.jl")

x0 = [0.;0.;0.;0.]# for the double integrator dynamics
xf = [5.; 5.; 0.; 0.]

println("System: 1-player double integrator dynamics with quadratic offset cost")
println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(horizon, sigdigits=4), " seconds.")
println()

#####################################
#        Define the dynamics.       #
#####################################
cont_dyn = ContinuousLinearDynamics([0. 0. 1. 0.;
                                     0. 0. 0. 1.;
                                     0. 0. 0. 0.;
                                     0. 0. 0. 0.],
                                    [vcat(zeros(2,2),
                                     [1. 0; 0 1.])]) # 2d double integrator [x y xdot ydot]
dyn = discretize(cont_dyn, dt)

#####################################
#         Define the costs.         #
#####################################
Q = Matrix(Diagonal(1*[1., 1., 1., 1.]))
R = Matrix(Diagonal(1*[1., 1.]))
quad_cost = QuadraticCost(Q)
add_control_cost!(quad_cost, 1, R)
quad_w_offset_cost = QuadraticCostWithOffset(quad_cost, xf)

println("setting cost to Quadratic Offset Cost")
selected_cost = quad_w_offset_cost
