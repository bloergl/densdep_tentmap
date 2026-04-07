# %%

# import standard tools
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# import tools for simulation
import sim_discrete as sd
import sim_volume as sv


VERBOSE = False
SAVEIT = False  # save to results directory
METHOD = 'discrete'  # 'discrete' or 'volume'

#
# points of interests, by playing around:
#
#    delta = 0.7628 (period 2)
#    delta = 0.593500 (period 2)
#    delta_start = 0.75077971, delta_stop = 0.75077972
#    delta_start = 0.9150, delta_stop = 09200
#    delta = 0.918
#


# simulation parameters
a_uppercase: float = 0.0
delta_start: float = 0.5  # DEFAULT: 0.5
delta_stop: float =  1.0  # 0.75077972  # DEFAULT: 1.0
delta_n: int = 5  # DEFAULT: 501
n_iter: int = 1000  # DEFAULT: 10000
n_init: int = 200
if METHOD == 'discrete':
    n_discrete = 1000000  # DEFAULT: 100000

# where to store
save_path = 'Results'
os.makedirs(f'.{os.sep}{save_path}', exist_ok=True)

# compute range of deltas to cover
delta = delta_start + (delta_stop - delta_start) * \
    np.arange(delta_n) / (delta_n - 1)

# remember what we were doing
t_history = np.zeros((delta_n,))
a_history = np.zeros((delta_n, n_iter))

# generate time series for all deltas
for delta_i, delta_tmp in enumerate(delta):

    print(f"Performing '{METHOD }' simulation for delta={delta_tmp:.07f}...")

    match METHOD:

        case 'discrete':
            # discrete sim with constant density initial condition
            t0 = time.perf_counter()
            x: np.ndarray = (0.5 + np.arange(n_discrete)) / n_discrete
            x_iter_last, a_iter, x_hist_sd, bin_mids, bins = sd.iterate_densdep_tent_map(
                x0=x,
                a_uppercase=a_uppercase,
                delta=delta_tmp,
                n_iter=n_iter + n_init
            )
            t1 = time.perf_counter()

        case 'volume':
            # box merging sim with constant density initial condition
            t0 = time.perf_counter()
            x_dens: np.ndarray = np.array([1.0,])
            v_dens: np.ndarray = np.array([1.0,])
            a_iter, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
                x=x_dens,
                v=v_dens,
                a_uppercase=a_uppercase,
                delta=delta_tmp,
                n_iter=n_iter + n_init,
                verbose=False,
                warning=False
            )
            t1 = time.perf_counter()
            # print(f"Precision of box merging simulation was {eps}")
            # print("Done!\n")

        case _:
            raise ValueError(f"Unknown METHOD={METHOD} specified!")

    t = t1 - t0
    a = a_iter[n_init:]

    if VERBOSE:
        plt.plot(np.ones_like(a) * delta_tmp, a, 'k.', ms=0.1)

    t_history[delta_i] = t
    a_history[delta_i, :] = a

if VERBOSE:
    plt.show()

    plt.plot(delta, t_history)
    plt.show()

if SAVEIT:
    save_name = f"{save_path}{os.sep}a_iterations_A{int(1000*a_uppercase):04d}_" + \
        f"a{int(1000*delta_start):04d}-{int(1000*delta_stop):04d}-{delta_n}_N{n_iter}-{n_init}_M{METHOD}"
    if METHOD == 'discrete':
        save_name += f"_nd{n_discrete}"
    save_name += '.npz'

    print(save_name)
    np.savez(save_name, t_history=t_history, a_history=a_history, delta=delta)

    # del a_history, t_history, delta
    # res = np.load(save_name)
    # delta = res["delta"]
    # a_history = res["a_history"]
    # t_history = res["t_history"]

plt.plot(delta, a_history, 'k.', ms=0.1)
plt.grid('both')
plt.show()


