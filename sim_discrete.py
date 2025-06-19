# %%
import numpy as np
import matplotlib.pyplot as plt


def tent_map(a: float, x: np.ndarray) -> np.ndarray:
    """ Computes the tent map with parameter 'a' on a vector 'x'

    """
    return a * np.min(np.concatenate((x[np.newaxis, :], 1 - x[np.newaxis, :])), axis=0)


def iterate_densdep_tent_map(
        x0: np.ndarray,
        delta: float = 0.5,
        a_uppercase: float = 0.0,
        n_iter: int = 500,
        n_bins: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate density-dependent tent map dynamics

    Parameters from Eq. (4.28) in preprint:
    ---------------------------------------
    delta: float = 0.5  # integration range
    a_uppercase: float = 0.0  # lower integration boundary

    Simulation parameters:
    ----------------------
    x0: np.ndarray  # vector with discrete values x representing continous density f(x)
    seed: int = 42  # rng seed
    n_bins: int = 250  # binning of f(x) for display/histograms
    n_iter: int = 500  # map iterations to compute

    Output values:
    --------------
    x: np.ndarray  # (n) discrete values x after last iteration
    a_iter: np.ndarray  # (n_iter) values of a for each iteration
    x_hist: np.ndarray  # (n_iter+1) x (n_bins) histograms for f(x)
    bin_mids: np.ndarray  # (n_bins) midpoints of histogram bins
    bins: np.ndarray  # (n_bins+1) boundaries of histogram bins
    """

    # initialization of simulation
    x = x0.copy()  # make copy to avoid Python views!
    bins = np.arange(n_bins + 1) / n_bins  # bin boundaries for histogram
    bin_mids = 0.5 * (bins[1:] + bins[:-1])  # bin mids for displaying histogram

    # store a[f] in every iteration
    a_iter = np.zeros((n_iter,))

    # store f(x) as historgram in every iteration
    x_hist = np.zeros((n_iter + 1, n_bins))
    x_hist[0, ...] = np.histogram(x, bins=bins)[0]

    # perform iterations
    for i_iter in range(n_iter):
        a = 1.0 + float(np.mean((x >= a_uppercase) * (x <= a_uppercase + delta)))
        x = tent_map(a, x)
        a_iter[i_iter] = a
        x_hist[i_iter + 1] = np.histogram(x, bins=bins)[0]

    return x, a_iter, x_hist, bin_mids, bins


if __name__ == "__main__":

    # test simulation
    a_uppercase: float = 0.0
    delta: float = 0.6
    n_iter: int = 1000
    n: int = 10000
    x0 = (np.arange(n) + 0.5) / n  # approx. equidistribution in [0, 1]

    # simulate in two chunks with n_iter iterations each...
    print("Simulation in two chunks")
    x, a_iter, x_hist, bin_mids, bins = iterate_densdep_tent_map(x0, a_uppercase=a_uppercase, delta=delta, n_iter=n_iter)
    y, a_iter, y_hist, bin_mids, bins = iterate_densdep_tent_map(x, a_uppercase=a_uppercase, delta=delta, n_iter=n_iter)

    # simulate in one chunk with 2*n_iter iterations...
    print("Simulation in one chunk")
    z, a_iter, z_hist, bin_mids, bins = iterate_densdep_tent_map(x0, a_uppercase=a_uppercase, delta=delta, n_iter=2 * n_iter)

    # check consistency of the two sims...
    assert np.sum(np.abs(z - y)) == 0, "There is an inconsistency between the two sims, stopping!"
    print("Simulations consistent!")

    # prepare results of last sim for display...
    n_iter_display = min(500, a_iter.size)
    t = np.arange(n_iter_display + 1)  # iteration axis
    a_hist = np.histogram(a_iter, bins=1 + bins)[0]
    # p_name = f"densdep_tent_map_A{a_uppercase:.3f}_delta{delta:.3f}.png"

    plt.figure(figsize=(11, 7))

    # cf Figure 4.10 (for n_iter = 10000, A = 0, delta = 0.5)
    plt.subplot(3, 1, 1)
    plt.plot(bin_mids + 1, a_hist / n_iter)
    plt.xlim([1, 2])
    plt.xlabel('a')
    plt.ylabel('density of a')

    # cf Figure 4.9 (for n_iter = 500, A = 0, delta = 0.5)
    plt.subplot(3, 1, 2)
    plt.plot(t[1:n_iter_display + 1], a_iter[:n_iter_display])
    plt.xlim([1, n_iter_display])
    plt.xlabel('iteration n')
    plt.ylabel('a')

    plt.subplot(3, 1, 3)
    plt.pcolor(t[:n_iter_display], bin_mids, z_hist[:n_iter_display, :].T)
    plt.xlabel("iteration n")
    plt.ylabel("x")
    plt.title("f(x)")

    # plt.savefig(p_name, dpi=300)
    plt.show()
