# %%
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# import simulation tools
import sim_volume as sv
import sim_discrete as sd


# range of delta values to span...
delta_range = (0.5, 0.9)
n_delta_bins = 10  # 800

# simulate discrete and volume...
a_uppercase: float = 0.0
n_iter: int = 10000
n_iter_transient: int = 100
n: int = 10000  # only for discrete sim; size of ensemble

# ============================================================
deltas = delta_range[0] + (delta_range[1] - delta_range[0]) * np.arange(n_delta_bins + 1) / n_delta_bins
os.makedirs('./ReturnMap/SimDiscrete', exist_ok=True)
os.makedirs('./ReturnMap/SimVolume', exist_ok=True)

# initial conditions discrete sim
x0: np.ndarray = (0.5 + np.arange(n)) / n

# initial conditions volume sim
x_dens: np.ndarray = np.array([1.0,])
v_dens: np.ndarray = np.array([1.0,])

t1 = time.perf_counter()
for i, delta in enumerate(deltas):

    print(f"Processing delta={delta}")

    for method in ('SimDiscrete', 'SimVolume'):
        if method == 'SimDiscrete':
            # perform simulation with ensemble of particles
            x_iter_last, a_iter, x_hist, bin_mids, bins = sd.iterate_densdep_tent_map(
                x0=x0,
                a_uppercase=a_uppercase,
                delta=delta,
                n_iter=n_iter + n_iter_transient
            )
        else:
            # perform simulation with box merging algorithm
            a_iter, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
                x=x_dens,
                v=v_dens,
                a_uppercase=a_uppercase,
                delta=delta,
                n_iter=n_iter + n_iter_transient,
                verbose=False,
                warning=False,
            )
            print(f"Precision of volume simulation was {eps}")

        # extract return map
        r_prev = a_iter[n_iter_transient:-1]
        r_next = a_iter[n_iter_transient + 1:]

        for zoom in ('none', 'tight'):

            # plot it...
            title = f"delta={delta}, method={method}"
            plt.figure(figsize=(7, 7))
            plt.plot([1, 2], [1, 2], color=[0.5, 0.5, 0.5])
            plt.plot(r_prev, r_next, 'b.', markersize=0.5)
            plt.xlabel('a[n]')
            plt.ylabel('a[n+1]')
            plt.title(title)

            if zoom == 'none':
                plt.xlim([1, 2])
                plt.ylim([1, 2])
            else:
                mi = r_prev.min()
                ma = r_prev.max()
                rg = [mi - (ma - mi) / 10, ma + (ma - mi) / 10]
                plt.xlim(rg)
                plt.ylim(rg)

            pname = f"ReturnMap/{method}/returnmap_delta-{delta:03f}_zoom-{zoom}.png"
            plt.savefig(pname, dpi=300)

            plt.show()

t2 = time.perf_counter()
print(f"Time elapsed: {t2-t1} secs.")
