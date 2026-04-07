# %%
import matplotlib.pyplot as plt
import numpy as np
import corr

# import tools for simulation
import sim_volume as sv
import sim_discrete as sd

# common parameters
a_uppercase: float = 0.0
delta: float = 0.75
n_iter: int = 1000

# discrete sim with constant density initial condition
print("Performing discrete simulation...")
n: int = 1000000
x: np.ndarray = (0.5 + np.arange(n)) / n
x_iter_last, a_iter_sd, x_hist_sd, bin_mids, bins = sd.iterate_densdep_tent_map(
    x0=x,
    a_uppercase=a_uppercase,
    delta=delta,
    n_iter=n_iter
)
print("Done!\n")


# box merging sim with constant density initial condition
print("Performing volume simulation...")
x_dens: np.ndarray = np.array([1.0,])
v_dens: np.ndarray = np.array([1.0,])
a_iter_sv, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
    x=x_dens,
    v=v_dens,
    a_uppercase=a_uppercase,
    delta=delta,
    n_iter=n_iter,
    verbose=False,
    warning=False
)
print(f"Precision of box merging simulation was {eps}")
print("Done!\n")


plt.plot(a_iter_sd)
plt.xlabel('iterations n, discrete')
plt.ylabel('a(n)')
plt.title(f"delta={delta:.04f}")
plt.show()

plt.plot(a_iter_sv)
plt.xlabel('iterations n, volume')
plt.ylabel('a(n)')
plt.title(f"delta={delta:.04f}")
plt.show()


cd = corr.cross(a_iter_sd, n_limit=100)
plt.plot(cd[1], cd[0])
plt.xlabel('delay')
plt.ylabel('autocorrelation, discrete')
plt.title(f"delta={delta:.04f}")
plt.show()

cv = corr.cross(a_iter_sv, n_limit=100)
plt.plot(cv[1], cv[0])
plt.xlabel('delay')
plt.ylabel('autocorrelation, volume')
plt.title(f"delta={delta:.04f}")
plt.show()
