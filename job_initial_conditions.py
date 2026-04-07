# %%

# import standard tools
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats

# import tools for simulation
import sim_discrete as sd
import sim_volume as sv
import sim_tools as st


VERBOSE = False

# simulation parameters
a_uppercase: float = 0.0
delta_start: float = 0.50
delta_stop: float = 0.999
delta_n: int = 1000
# delta: float = 0.63
# n_init: int = 200

n_init: int = 200
n_iter: int = 300
n_comp: int = 10
n_vols_min: int = 2
n_vols_max: int = 8
x_start: float = 0.0
x_end: float = 1.0

# uniform initial density
x0_uni = np.array((1,))
v0_uni = np.array((1,))

# compute range of deltas to cover
if delta_n == 1:
    delta = np.array((delta_start,))
else:
    delta = delta_start + (delta_stop - delta_start) * np.arange(delta_n) / (
        delta_n - 1
    )

# start with none, or well-defined seed
seed = 43
rng = np.random.default_rng(seed)

# remember results
res_kstest = np.zeros((delta_n, n_comp))
res_a_uni = np.zeros((delta_n, n_init + n_iter))
res_a_inh = np.zeros((delta_n, n_comp, n_init + n_iter))

for i_delta in range(delta_n):

    # get current value of delta
    temp_delta = delta[i_delta]

    # get dynamics evolving from uniform density
    a_uni, x_uni, v_uni, eps = sv.iterate_densdep_tent_map_vols(
        x=x0_uni,
        v=v0_uni,
        a_uppercase=a_uppercase,
        delta=temp_delta,
        n_iter=n_init + n_iter,
        verbose=False,
        warning=False,
    )
    res_a_uni[i_delta, :] = a_uni

    for i_comp in range(n_comp):

        # how many volumes?
        n_vols = rng.integers(n_vols_min, n_vols_max + 1)
        flag_compact = bool(rng.integers(2))
        sv_seed = rng.integers(100000)

        # non-uniform initial density
        x0_inh, v0_inh = sv.construct_random_density(
            n_vols=n_vols,
            x_start=x_start,
            x_end=x_end,
            compact=flag_compact,
            rng_seed=sv_seed,
        )

        # verbose...
        if VERBOSE:
            sv.plot_density(x=x0_uni, v=v0_uni)
            sv.plot_density(x=x0_inh, v=v0_inh)
            plt.title("Initial condition")
            plt.show()

        a_inh, x_inh, v_inh, eps = sv.iterate_densdep_tent_map_vols(
            x=x0_inh,
            v=v0_inh,
            a_uppercase=a_uppercase,
            delta=temp_delta,
            n_iter=n_init + n_iter,
            verbose=False,
            warning=False,
        )
        h_uni = np.histogram(a_uni, 50, [1.0, 2.0])
        h_inh = np.histogram(a_inh, 50, [1.0, 2.0])
        kt = stats.kstest(a_uni[n_init:], a_inh[n_init:])

        if VERBOSE:
            plt.plot(a_uni[n_init:])
            plt.plot(a_inh[n_init:])
            plt.title("time series of a")
            plt.show()

            plt.plot(h_uni[1][:-1], h_uni[0])
            plt.plot(h_inh[1][:-1], h_inh[0])
            plt.title(f"histograms, KS={kt.pvalue:.03f}")
            plt.show()

        # store results
        res_kstest[i_delta, i_comp] = kt.pvalue
        res_a_inh[i_delta, i_comp, :] = a_inh

print("Finished!")
plt.plot(res_kstest.T)
plt.show()

# %%

plt_path = "Plots"

os.makedirs(f".{os.sep}{plt_path}", exist_ok=True)
os.makedirs(f".{os.sep}{plt_path}{os.sep}InitialCond", exist_ok=True)

for i_comp in range(n_comp):
    fname = f"init_nonuni_rep{i_comp}"
    st.plot_bifurcations(x=delta, a_xr=res_a_inh[:, i_comp, n_init:])
    plt.title(f"Non-uniform init, rep={i_comp}")
    plt.xlabel("d")
    plt.ylabel("a")
    plt.savefig(
        f"{plt_path}{os.sep}InitialCond{os.sep}{fname}.png",
        dpi=1200,
        orientation="landscape",
    )
    plt.show()

st.plot_bifurcations(x=delta, a_xr=res_a_uni[:, n_init:])
plt.title(f"Uniform initialization")
plt.xlabel("d")
plt.ylabel("a")
plt.savefig(
    f"{plt_path}{os.sep}InitialCond{os.sep}init_uniform.png",
    dpi=1200,
    orientation="landscape",
)
plt.show()
