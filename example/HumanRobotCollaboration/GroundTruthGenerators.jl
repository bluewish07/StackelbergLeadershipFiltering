function get_simple_straight_line_2D_traj()
    x1 = [0; 0; 0; 0]
    # human aim to follow y = 0 (x < 80), 1 (x>80) approximated by y = 0.01x
    u1 = vcat(hcat(10*ones(1, 10),
                   zeros(1, 35),
                   -20*ones(1, 5),
                   5*ones(1, 5),
                   zeros(1, 46)
              ),
              hcat(zeros(1, 50),
                   2*ones(1, 5),
                   zeros(1, 42),
                   -2*ones(1, 4)
              ),
        )
     # robot aim to follow y = 0
     u2 = vcat(zeros(1, 101),
               hcat(zeros(1, 50),
                    reshape(-0.001 * [i for i in 1:51], 1, 51)
                    )
               )

    us = [u1, u2]
    return us, x1
end