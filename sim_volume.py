# %%
import numpy as np
import matplotlib.pyplot as plt

"""
The collection of functions in this module work with a representation of
a density as a set of boxes which span the range x in [0, 1]. Each box k
is located between x[k-1] and x[k] and has a volume of v[k]. x[0] is not
explicitly represented since it is assumed to be zero. For a valid density,
the volumes of all boxes have to sum to 1.
"""


def partial_integral(x: np.ndarray, v: np.ndarray, delta: float, a_uppercase: float = 0.0) -> float:
    """
    Computes the integral a over the interval [a_uppercase, a_uppercase+delta]
    over a set of boxes with volumes v with upper limits in x

    Returns: a
    """

    # check if integration parameters are well defined...
    assert (a_uppercase >= 0.0) and (a_uppercase <= 1.0), "a_uppercase must be in [0, 1]!"
    assert (delta > 0.0) and (a_uppercase + delta <= 1.0), "delta must be in [0, 1-a_uppercase]!"

    # we do not do any check on the consistency of x and v, assuming it is done before. 
    a = 1.0
    x_pre = 0.0
    flag_above_a_uppercase = x_pre >= a_uppercase
    for i_cur, x_cur in enumerate(x):

        if not flag_above_a_uppercase:

            # count (partial) volume if x_cur above lower boundary
            if x_cur >= a_uppercase:
                if x_cur < a_uppercase + delta:
                    # take volume up to x_cur...
                    a += v[i_cur] * (x_cur - a_uppercase) / (x_cur - x_pre)
                    flag_above_a_uppercase = True
                else:
                    # take volume ONLY up to a_uppercase + delta, and exit!
                    a += v[i_cur] * delta / (x_cur - x_pre)
                    break

        else:

            if x_cur < a_uppercase + delta:
                # take full volume from x_prev to x_cur...
                a += v[i_cur]
            else:
                # take volume ONLY up to a_uppercase + delta, and exit!
                a += v[i_cur] * (a_uppercase + delta - x_pre) / (x_cur - x_pre)
                break

        x_pre = x_cur

    return a


def bin_volumes(x: np.ndarray, v: np.ndarray, n_bins: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the density (x, v) with non-equidistance x
    with n_bin equidistant bins. Use only for display purpose, the binning
    introduces some smoothin/binning artefacts.

    Normalization is performed such that: np.sum(v_bin) / n_bins == 1 

    Returns: x_mid, v_bin
    """

    dx_bin = 1.0 / n_bins
    x_bin = dx_bin * np.arange(1, n_bins + 1)
    # print(x_bin[-1])
    x_bin[-1] = 1.0
    assert x[-1] == 1.0, f"Last entry of x must be 1.0, but is {x[-1]}!"
    # mid_bins = 0.5 * (bins[1:] + bins[:-1])
    v_bin = np.zeros((n_bins,))

    # define pointer to bins and x-boundaries
    idx_bin: int = 0
    idx: int = 0

    # get current and previous positions
    x_bin_cur = x_bin[idx_bin]
    x_cur = x[idx]
    x_pre: float = 0.0

    # get current volume to distribute over bins
    v_cur = v[idx]

    while True:

        if x_bin_cur > x_cur:
            # distribute remaining volume to current bin
            # dx = x_cur - x_pre
            dv = v_cur
            v_bin[idx_bin] += dv

            # advance to next entry in x
            idx += 1
            x_pre = x_cur
            x_cur = x[idx]
            v_cur = v[idx]

        else:
            # distribute partial remaining volume to current bin
            dx_bin = x_bin_cur - x_pre
            dx = x_cur - x_pre
            dv = v_cur * (dx_bin / dx)
            v_bin[idx_bin] += dv
            v_cur -= dv

            # finished?
            if x_bin_cur == 1:
                break

            # advance to next entry in x_bin
            idx_bin += 1
            x_pre = x_bin_cur
            x_bin_cur = x_bin[idx_bin]

    v_bin *= n_bins

    return x_bin - 0.5 / n_bins, v_bin


def plot_density(x: np.ndarray, v: np.ndarray) -> None:
    """
    Displays density as a filled polygon.
    """

    # duplicate boundaries of volumes
    x_extd = np.concatenate(((0.0,), x))
    x_extd_dup = np.tile(x_extd[:, np.newaxis], (1, 2)).flatten()

    # compute heights of volumes, duplicate
    dx = np.diff(x_extd)
    y = v / dx
    y_dup = np.tile(y[:, np.newaxis], (1, 2)).flatten()
    y_extd_dup = np.concatenate(((0.0,), y_dup, (0.0,)))

    # plot distribution
    plt.fill(x_extd_dup, y_extd_dup)

    return


def check_consistency(x: np.ndarray, v: np.ndarray, precision: float, default_type: type = np.float64):
    """
    Checks consistency of a density (x, v), i.e.:
    normalization, range, positivity, ...

    Breaks with error if some inconsistency is detected...
    """

    assert x.dtype == default_type, f"type of x must be {default_type}"
    assert v.dtype == default_type, f"type of v must be {default_type}"
    n = x.size
    assert n == v.size, "x and v must have same size!"
    assert np.sum(x >= 0.0) == n, "x entries must be > 0!"
    for i in range(1, n):
        assert x[i] >= x[i - 1], "x must be ascending!"
    assert x[-1] == 1, "Last element of x must be 1!"
    for i in range(n):
        assert v[i] >= 0.0, "Elements of v must be >= 0!"
    v_sum = v.sum()
    if v_sum != 1.0:
        assert np.abs(v_sum - 1) < precision, "sum(v) must be 1 within precision lims!"
        print(f"WARNING: sum(v) error is {v_sum - 1}, but within req precision")

    return


def iterate_densdep_tent_map_vols(
        delta: float = 0.5,
        a_uppercase: float = 0.0,
        x: np.ndarray = np.array([1.0,]),
        v: np.ndarray = np.array([1.0,]),
        n_iter: int = 500,
        verbose: bool = False,
        debug: bool = False,
        warning: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    """
    Simulate density-dependent tent map dynamics for n_iter iterations
    from initial conditions (x, v).

    Parameters in Eq. (4.28):
    delta: float = 0.5  # integration range
    a_uppercase: float = 0.0  # lower integration boundary

    Simulation parameters:
    x: np.ndarray = np.array([1.0,])
        # vector with right boundaries of volumes
    v: np.ndarray = np.array([1.0,])
        # vector with volumes (sums to 1)
    n_iter: int = 500  # map iterations to compute
    verbose: bool = False  # give a few messages what is done
    debug: bool = False  # give more debugging info
    warning: bool = True  # issue numerical warnings

    Returns:
        a_iter: (partial) integral over density forneach iteration
        x_iter: list of volume boundaries for each iteration
        v_iter: list of volumes for each iteration
        eps: numerical precision used to check for putative inconsistencies
    """

    # get numerical precision
    default_type = np.float64
    eps: float = float(np.finfo(default_type).eps)
    if verbose:
        print(f"SDTMV: Numerical precision is {eps}")

    # convert inputs to required data type
    if verbose:
        print("SDTMV: Startup, checking parameter consistency")
    x_new = x.astype(default_type)
    v_new = v.astype(default_type)
    check_consistency(x=x_new, v=v_new, precision=eps * 10, default_type=default_type)

    # store integrals over density
    a_iter = np.zeros((n_iter,))
    x_iter = [x_new.copy(),]
    v_iter = [v_new.copy(),]

    for i in range(n_iter):

        if verbose:
            print(f"SDTMV: Iteration {i + 1} of {n_iter}...")

        # free *_new to accumulate new results
        x = x_new.copy()
        v = v_new.copy()

        # compute integral for determining tent map param a
        a = partial_integral(x=x, v=v, delta=delta, a_uppercase=a_uppercase)
        if a < 1:
            if np.abs(a - 1.0) > eps * 10:
                raise ValueError(f"a < 1, a={a}")
            else:
                if warning:
                    print("SDTMV: WARNING - a is less than 10*eps smaller than 1, clamping...")
                a = 1.0
        if a > 2:
            if np.abs(a - 2.0) > eps * 100:
                raise ValueError(f"a > 2, a={a}")
            else:
                if warning:
                    print("SDTMV: WARNING - a is less than 10*eps larger than 2, clamping...")
                a = 2.0

        a_iter[i] = a
        if debug:
            print(f"DEBUG: x = {x}")
            print(f"DEBUG: v = {v}")
            print(f"DEBUG: A = {a_uppercase}, delta = {delta}")
            print(f"DEBUG: a = {a} ----------< INTEGRAL")

        dx = np.diff(np.concatenate(((0,), x)))
        # TODO check that x[0] does not become < 0

        # select volumes fully on LHS
        flag_left = x <= 0.5
        x_left = x[flag_left]
        v_left = v[flag_left]

        # handle volume stretching across border
        if x_left.size == 0:
            x_left_last = 0.0
        else:
            x_left_last = x_left[-1]

        if x_left_last < 0.5:
            idx_middle = x_left.size
            v_middle = v[idx_middle]
            v_left_middle = v_middle * (0.5 - x_left_last) / dx[idx_middle]
            x_left = np.concatenate((x_left, (0.5,)))
            v_left = np.concatenate((v_left, (v_left_middle,)))
            v[idx_middle] -= v_left_middle

        # select volumes fully/partially on RHS
        flag_right = x > 0.5
        x_right = (1 - x[flag_right])[::-1]
        v_right = (v[flag_right])[::-1]
        if np.abs(x_right[0]) > 10 * eps:
            raise ValueError("x_right[0] must be close to zero!")
        else:
            if x_right[0] != 0.0:
                if warning:
                    print("SDTMV: WARNING - x_right[0] less than 10*eps deviating from 0, clamping...")
                x_right[0] = 0.0
        assert x_right[0] == 0, "x_right[0] is still not zero, something is fishy!"
        x_right = np.concatenate((x_right[1:], (0.5,)))
        assert x_right[-1] == 0.5, "x_right[-1] not properly set to 0.5!"
        assert x_left[-1] == 0.5, "x_left[-1] not properly set to 0.5"

        if debug:
            print(f"DEBUG: x={x}, v={v}: -->")
            print(f"DEBUG: x_left={x_left}, v_left={v_left}")
            print(f"DEBUG: x_right={x_right}, v_right={v_right} --------< up/down tent splitting")

        # assert v_left.sum() + v_right.sum() == 1
        idx_left = 0
        idx_right = 0
        v_left_temp = v_left[idx_left]
        v_right_temp = v_right[idx_right]
        x_temp = 0.0
        v_new_list: list = []
        x_new_list: list = []
        while True:
            # print(f"idx_left={idx_left}, idx_right={idx_right}")
            if (idx_left == v_left.size - 1) and (idx_right == v_right.size - 1):
                # print("Handle last!")
                v_new_list.append(v_left_temp + v_right_temp)
                x_new_list.append(0.5)
                break
            if x_left[idx_left] < x_right[idx_right]:
                f_partial = (x_left[idx_left] - x_temp) / (x_right[idx_right] - x_temp)
                # print("A", v_left_temp, f_partial, v_right_temp)
                v_combined = v_left_temp + f_partial * v_right_temp
                v_right_temp = (1 - f_partial) * v_right_temp
                x_temp = x_left[idx_left]
                idx_left += 1
                v_left_temp = v_left[idx_left]
            else:
                f_partial = (x_right[idx_right] - x_temp) / (x_left[idx_left] - x_temp)
                # print("B", v_right_temp, f_partial, v_left_temp)
                v_combined = v_right_temp + f_partial * v_left_temp
                v_left_temp = (1 - f_partial) * v_left_temp
                x_temp = x_right[idx_right]
                idx_right += 1
                v_right_temp = v_right[idx_right]
            v_new_list.append(v_combined)
            x_new_list.append(x_temp)
            # print(f"x_temp={x_temp}, v_combined={v_combined}")

        if debug:
            print(f"DEBUG: x_new={x_new_list}, v_new={v_new_list} ---------< FINAL MERGE STATE")

        x_new = a * np.array(x_new_list)
        v_new = np.array(v_new_list)
        if a < 2.0:
            x_new = np.concatenate((x_new, (1.0,)))
            v_new = np.concatenate((v_new, (0.0,)))
        v_new_sum = v_new.sum()

        if debug:
            print(f"DEBUG: x={x_new}")
            print(f"DEBUG: v={v_new}")
            print(f"DEBUG: sum(v)={v_new_sum}")

        if np.abs(v_new_sum - 1.0) > 10 * eps:
            raise ValueError("sum(v) is not 1!")
        else:
            # v_new /= v_new_sum
            if warning:
                print(f"SDTMV: WARNING - sum(v) is less than 10*eps from 1 at {v_new_sum}!")
            # print(f"SDTMV:           After clamping, diff is now {v_new.sum() - 1.0}")

        if verbose:
            if i > n_iter - 10:
                plot_density(x_new, v_new)
                plt.show()

        x_iter.append(x_new.copy())
        v_iter.append(v_new.copy())

    if verbose:
        plt.plot(a_iter)
        plt.show()

    return a_iter, x_iter, v_iter, eps


if __name__ == "__main__":

    default_type = np.float64
    eps = float(np.finfo(default_type).eps)

    # test_mode = 1:
    #   define density, check consistency, show density
    # test_mode = 2:
    #   test computation of (partial) integral
    # test_mode = 3:
    #   test binning of densities
    # test_mode = 4:
    #   test density-dependent tent map iteration
    test_mode = 4
    if test_mode == 1:

        # define density f from three volumes
        v = np.array((0.1, 0.2, 0.7 - eps), dtype=default_type)
        x = np.array((0.4, 0.8, 1.0), dtype=default_type)

        # check if definition of density f is consistent
        print("Checking consistency:")
        check_consistency(x=x, v=v, precision=10 * eps, default_type=default_type)
        print("...okay!")

        # plot density f
        plot_density(x=x, v=v)
        plt.show()

    if test_mode == 2:

        v = np.array((0.1, 0.7, 0.2,))
        x = np.array((0.1, 0.2, 1.0,))
        check_consistency(x=x, v=v, precision=10 * eps, default_type=default_type)

        print("Checking computation of partial integral over density")
        print(f"x={x}, v={v}...")
        print("")
        print("Results:")
        print(f"A=0, delta=0.3: {partial_integral(x, v, 0.3)}")
        print(f"A=0.11, delta=0.08: {partial_integral(x, v, 0.08, a_uppercase=0.11)}")
        print(f"A=0.05, delta=0.2: {partial_integral(x, v, 0.2, a_uppercase=0.05)}")

    if test_mode == 3:

        n_bins = 25

        v = np.array((1.0,))
        x = np.array((1.0,))
        x_mid, v_bin = bin_volumes(x=x, v=v, n_bins=n_bins)
        print(f"x={x}, v={v}")
        print(f"x_bin={x_mid}, v_bin={v_bin}")
        print()

        plot_density(x=x, v=v)
        plt.plot(x_mid, v_bin, 'ko-')
        plt.show()

        v = np.array((0.1, 0.7, 0.2,))
        x = np.array((0.1, 0.2, 1.0,))
        x_mid, v_bin = bin_volumes(x=x, v=v, n_bins=n_bins)
        print(f"x={x}, v={v}")
        print(f"x_bin={x_mid}, v_bin={v_bin}")
        print()

        plot_density(x=x, v=v)
        plt.plot(x_mid, v_bin, 'ko')
        plt.show()

    if test_mode == 4:

        n_iter = 1000

        # delta: float = 0.412607
        delta: float = 0.65
        verbose: bool = False
        warning: bool = False
        a_iter, x_iter, v_iter, eps = iterate_densdep_tent_map_vols(
            delta=delta, verbose=verbose, n_iter=n_iter, warning=warning)

        iter_show = -1
        x_bin, v_bin = bin_volumes(x=x_iter[iter_show], v=v_iter[iter_show], n_bins=100)
        plot_density(x=x_iter[iter_show], v=v_iter[iter_show])
        plt.plot(x_bin, v_bin, 'ko-')
        plt.show()
