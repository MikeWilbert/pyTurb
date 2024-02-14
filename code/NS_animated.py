import pyTurb

N = 128
k_a = 0.5
k_f = 10
c_res = 1.
eps = 1.


pyTurb.init(N, k_a, k_f, c_res, eps)

pyTurb.print_scales()

pyTurb.step()
