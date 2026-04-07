# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import sim_discrete as sd
import time
from tqdm import tqdm


# x1 = 0.2
# x2 = 0.3
delta = 0.6
a_uppercase = 0
n_iter = 100
n_init = 500

n1 = 100
n2 = 100

x1_range = (0.5 + np.arange(n1)) / n1
x2_range = (0.5 + np.arange(n2)) / n2

a_iter_std = np.zeros((n1, n2))

for i1, x1 in enumerate(tqdm(x1_range)):
    for i2, x2 in enumerate(x2_range):

        x0 = np.array((x1, x2))

        # discrete sim with constant density initial condition
        t0 = time.perf_counter()
        x_iter_last, a_iter, x_hist_sd, bin_mids, bins = sd.iterate_densdep_tent_map(
            x0=x0,
            a_uppercase=a_uppercase,
            delta=delta,
            n_iter=n_iter + n_init
        )
        t1 = time.perf_counter()

        # plt.plot(a_iter)
        # plt.show()

        a_iter_std[i1, i2] = a_iter[n_init:].std()

# %%

fname = f"it_takes_two_delta{int(1000*delta):04d}"
np.savez(fname + '.npz', x1_range=x1_range, x2_range=x2_range, \
         delta=delta, a_iter_std=a_iter_std)
plt.figure(figsize=(7, 7))
plt.pcolor(x1_range, x2_range.T, a_iter_std.T, cmap='seismic')
plt.savefig(fname + '.png', dpi=600)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import sim_discrete as sd

a_uppercase = 0.0
delta = 0.6
n_iter = 300

x_hist = np.zeros((n_iter + 1, 2))
a_hist = np.zeros((n_iter + 1,))

x = np.array((0.4, 0.1))
x_hist[0, ...] = x


# perform iterations
for i_iter in range(n_iter):
    a = 1.0 + float(np.mean((x >= a_uppercase) * (x <= a_uppercase + delta)))
    x = sd.tent_map(a, x)
    x_hist[i_iter + 1, ...] = x
    a_hist[i_iter] = a
a_hist[-1] = a = 1.0 + float(np.mean((x >= a_uppercase) * (x <= a_uppercase + delta)))

plt.plot(x_hist)
plt.show()
plt.plot(a_hist)
plt.show()
