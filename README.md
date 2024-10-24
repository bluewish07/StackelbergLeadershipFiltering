# Stackelberg Leadership Filtering

This codebase implements the algorithms and examples from [our paper on leadership inference](https://arxiv.org/pdf/2310.18171).

## Structure
- the `src` folder contains source code files that define the core filter
- the `example` folder contains runnable scripts for examples that were implemented
- the `test` folder contains unit tests
- the `derivations` folder contains pdfs describing the math behind the filter, SILQGames, and the solution to an LQ Stackelberg game

## Setup

This repo is structured as a Julia package. To activate this package, type
```console
julia> ]
(@v1.6) pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(StackelbergLeadershipFiltering) pkg>
```
Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using StackelbergLeadershipFiltering
```

## Utilities
Contains useful utilities. The `Cost` struct stores matrices which define a quadratic cost function for each player, and the `Dynamics` struct defines linear game dynamics (both are time-invariant). Functions to evaluate cost functions and unroll trajectories from initial conditions are provided as well.



To run tests locally and avoid polluting your commit history, in the REPL you can type:
```console
julia> ]
(StackelbergLeadershipFiltering) pkg> test
```

Alternatively, you can run:
```console
julia> include("test/runtests.jl")
```
