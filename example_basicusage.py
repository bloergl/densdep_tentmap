# %%
import matplotlib.pyplot as plt
import numpy as np

# import tools for simulation
import sim_volume as sv
import sim_discrete as sd

# common parameters
a_uppercase: float = 0.0
delta: float = 0.6
n_iter: int = 200

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

# make histograms from volume sim
n_bins = bin_mids.size
rho_x_sv = np.zeros((n_iter + 1, n_bins))
for i in range(n_iter + 1):
    x_bin, v_bin = sv.bin_volumes(x_iter[i], v_iter[i], n_bins=n_bins)
    rho_x_sv[i, :] = v_bin

# normalize discrete sim
rho_x_sd = x_hist_sd / n * n_bins

# prepare results for display...
n_iter = a_iter_sd.size
n_iter_display = min(500, n_iter)
t = np.arange(n_iter + 1)  # iteration axis
a_hist_sd = np.histogram(a_iter_sd, bins=1 + bins)[0]
a_hist_sv = np.histogram(a_iter_sv, bins=1 + bins)[0]

# cf Figure 4.9 (for n_iter = 500, A = 0, delta = 0.5)
plt.figure(figsize=(7.5, 11.5))

plt.subplot(5, 1, 1)
plt.plot(t[1:n_iter_display + 1], a_iter_sd[:n_iter_display], label='discrete')
plt.plot(t[1:n_iter_display + 1], a_iter_sv[:n_iter_display], label='volume')
plt.xlim([1, n_iter_display])
plt.legend()
plt.xlabel('iteration n')
plt.ylabel('a')

# cf Figure 4.10 (for n_iter = 10000, A = 0, delta = 0.5)
plt.subplot(5, 1, 2)
plt.plot(bin_mids + 1, a_hist_sd / n_iter, label='discrete')
plt.plot(bin_mids + 1, a_hist_sv / n_iter, label='volume')
plt.xlim([1, 2])
plt.legend()
plt.xlabel('a')
plt.ylabel('density of a')

plt.subplot(5, 1, 3)
plt.pcolor(t[:n_iter_display + 1], bin_mids,
           rho_x_sd[:n_iter_display + 1, :].T, vmin=0, vmax=rho_x_sv.max())
plt.xlabel("iteration n")
plt.ylabel("x")
plt.title("f(x) - discrete")

plt.subplot(5, 1, 4)
plt.pcolor(t[:n_iter_display + 1], bin_mids,
           rho_x_sv[:n_iter_display + 1, :].T, vmin=0, vmax=rho_x_sv.max())
plt.xlabel("iteration n")
plt.ylabel("x")
plt.title("f(x) - volume")

plt.subplot(5, 1, 5)
plt.pcolor(t[:n_iter_display + 1], bin_mids,
           np.abs((rho_x_sv - rho_x_sd)[:n_iter_display + 1, :].T), vmin=0, vmax=np.abs(rho_x_sd - rho_x_sv).max())
plt.xlabel("iteration n")
plt.ylabel("x")
plt.title("f(x) - difference")

# p_name = f"comp_densdep_tent_map_A{a_uppercase:.3f}_delta{delta:.3f}.png"
# plt.savefig(p_name, dpi=300)
plt.show()
