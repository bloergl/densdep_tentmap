# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def tentmap(x: float, a:float) -> float:
    """
    computes the normal tent map
    """

    if x > 0.5:
        x_new = a * (1 - x)
    else:
        x_new = a * x

    return x_new


@njit
def dd_tentmap_singledelta(x: float, delta: float, n_iter: int=1) -> list[np.ndarray, np.ndarray]:
    """
    computes the density-dependent tent map
    on a density composed of a single delta distribution,
    and for n_iter iterations
    """

    assert n_iter > 0, "n_iter must be larger than one!"

    x_iter = np.zeros((n_iter,))
    a_iter = np.zeros((n_iter,))
    for i_iter in range(n_iter):
        a = 1.0 + (x < delta)
        x = tentmap(x, a)
        a_iter[i_iter] = a
        x_iter[i_iter] = x

    return x_iter, a_iter


@njit
def dd_tentmap_doubledelta(x1: float, x2: float, delta: float, n_iter: int=1) -> list[np.ndarray, np.ndarray]:
    """
    computes the density-dependent tent map
    on a density composed of two delta distributions,
    and for n_iter iterations
    """

    assert n_iter > 0, "n_iter must be larger than one!"

    x_iter = np.zeros((2, n_iter))
    a_iter = np.zeros((n_iter,))
    for i_iter in range(n_iter):
        a = 1.0 + 0.5 * ((x1 < delta) + (x2 < delta))
        # if (a == 2) and (x1 == 0):
        #     x1 = 1e-14 * np.random.random()
        # if (a == 2) and (x2 == 0):
        #     x2 = 1e-14 * np.random.random()
        # if (a == 2) and (x1 == 1):
        #     x1 = 1 - 1e-14 * np.random.random()
        # if (a == 2) and (x2 == 1):
        #     x2 = 1 - 1e-14 * np.random.random()
        x1 = tentmap(x1, a)
        x2 = tentmap(x2, a)
        a_iter[i_iter] = a
        x_iter[0, i_iter] = x1
        x_iter[1, i_iter] = x2

    return x_iter, a_iter


# CODE NOT USED!
#
# ...idea was to achieve a better numerical computation by
# representing state x in a nominator/denominator representation
#  
# ...this is not really helpful, since we still can only represent
# rational numbers
#

# def dd_tentmap_singledelta_nomdenom(nom: int, denom: int, delta: float) -> int:
#
#     if nom / denom <= delta:  # for a=2
#         if 2 * nom > denom:
#             nom_new = 2 * (denom - nom)
#         else:
#             nom_new = 2 * nom
#     else:  # for a=1
#         if 2 * nom > denom:
#             nom_new = denom - nom
#         else:
#             nom_new = nom
#     return nom_new


def plot_dd_tentmap_singledelta(delta: float):
    """
    plots the density-dependent tent map acting on a single delta distrib
    """

    x_low = min(delta, 0.5)
    x_high = max(delta, 0.5)

    # lower branch
    plt.plot([0, x_low], [0, 2 * x_low], 'k-', linewidth=4)

    # middle branch
    if delta < 0.5:
        plt.plot([x_low, 0.5], [x_low, 0.5], 'k-', linewidth=4)
    else:
        plt.plot([0.5, x_high], [1, 2 * (1 - x_high)], 'k-', linewidth=4)

    # upper branch
    plt.plot([x_high, 1], [1 - x_high, 0], 'k-', linewidth=4)

    # diagonal
    plt.plot([0, 1], [0, 1], 'k--')

    return


def dd_tentmap_singledelta_bitch(
    x, 
    delta, 
    n_iter: int=1,
    randomize: bool=False
) -> list[np.ndarray, np.ndarray]:
    """
    computes the density-dependent tent map
    on a density composed of a single delta distribution,
    and for n_iter iterations
    --- ON BINARY REPRESENTATION! ---
    """

    assert n_iter > 0, "n_iter must be larger than one!"

    dtype = x.dtype
    assert delta.dtype == dtype, "x and delta must have same UINT data type!"

    assert np.isdtype(dtype, 'unsigned integer') == True, "x and delta must be unsigned integers!"

    x_max = np.iinfo(dtype).max
    one = np.array(x_max // 2).astype(dtype) + 1
    half = np.array(one // 2).astype(dtype)
    x_iter = np.zeros((n_iter,), dtype=dtype)
    a_iter = np.zeros((n_iter,), dtype=dtype)

    for i_iter in range(n_iter):
        x_plus = False
        if x < delta:
            # tent map with a=2
            a_iter[i_iter] = one
            if randomize:
                x_plus = (np.random.random() >= 0.5)
            x = np.left_shift(x, 1) + x_plus
            if x > one:
                x = one - (x - one)
        else:
            # tent map with a=1
            if x > half:
                x = half - (x - half)

        # store it!
        x_iter[i_iter] = x


    return x_iter, a_iter


# defines some bit-shift...
def bit_ch(x, n_iter=1, randomize=False):

    dtype = x.dtype
    # min_value = np.iinfo(dtype).min
    x_max = np.iinfo(dtype).max
    one = np.array(x_max // 2).astype(dtype) + 1
    x_iter = np.zeros((n_iter + 1,), dtype=dtype)

    x_iter[0] = x
    x_plus = False
    for i_iter in range(n_iter):
        if randomize:
            x_plus = (np.random.random() >= 0.5)
        x = np.left_shift(x, 1) + x_plus
        if x > one:
            x = one - (x - one)
        x_iter[i_iter + 1] = x

    return x_iter


# %%



if __name__ == "__main__":

    import os
    import time
    from tqdm import tqdm

    plt_path = 'Plots'
    os.makedirs(f'.{os.sep}{plt_path}', exist_ok=True)

    # -----------------------------------------------------
    # compare arbitrary with float precision for single delta...
    #  
    dtype = np.uint64
    half = dtype(np.iinfo(dtype).max // 2 + 1)

    n_iter = 500
    x0 = np.random.random()
    x0_int = dtype(half * x0)
    delta = 0.5001
    delta_int = dtype(half * delta)

    plot_dd_tentmap_singledelta(delta)
    plt.show()

    x_iter_int, a_iter_int = \
        dd_tentmap_singledelta_bitch( \
        x0_int, delta_int, n_iter=n_iter, randomize=True)
    x_iter_int = (x_iter_int).astype(float) / half

    x_iter, a_iter = dd_tentmap_singledelta(x0, delta, n_iter=n_iter)

    plt.plot(x_iter_int, 'r')
    plt.plot(x_iter, 'b')
    plt.legend(('"arb" prec', 'float prec'))
    plt.xlabel('#(iterations)')
    plt.ylabel('x')
    plt.title(f"Example dynamics, density with one delta-distrib, delta={delta:.4f}")
    plt.savefig(f"{plt_path}{os.sep}one_delta_dynamics.png", dpi=300)
    plt.show()


    # -----------------------------------------------------
    # 'bifurcation' diagram...
    #  
    dtype = np.uint64
    half = dtype(np.iinfo(dtype).max // 2 + 1)

    n_iter = 1000
    n_init = 1000
    n_delta = 250

    deltas = (np.arange(n_delta) + 0.5) / n_delta
    x0 = 0.765

    std = np.zeros((2, n_delta))
    sum = np.zeros((2, n_delta))

    for i_delta, delta in enumerate(deltas): 

        x0_int = dtype(half * x0)
        delta_int = dtype(half * delta)

        x_iter_int, a_iter_int = dd_tentmap_singledelta_bitch(x0_int, delta_int, n_iter=n_iter + n_init, randomize=True)
        x_iter_int = (x_iter_int).astype(float) / half

        x_iter, a_iter = dd_tentmap_singledelta(x0, delta, n_iter=n_iter + n_init)

        # if n_delta < 10:
        #     plt.plot(x_iter_int)
        #     plt.plot(x_iter)
        #     plt.legend(('"arb" prec', 'floating'))
        #     plt.show()

        std[1, i_delta] = x_iter[n_init:].std()
        std[0, i_delta] = x_iter_int[n_init:].std()
        sum[1, i_delta] = np.sum(np.abs(x_iter - x_iter[-1]))
        sum[0, i_delta] = np.sum(np.abs(x_iter_int - x_iter_int[-1]))

        plt.plot(delta * np.ones((n_iter,)), x_iter_int[n_init:], 'r.', markersize=0.2)
        plt.plot(delta * np.ones((n_iter,)), x_iter[n_init:], 'b.', markersize=0.2)

    plt.xlabel('delta')
    plt.ylabel('x[n]')
    plt.title('Bifurcation diagram for density with one delta-distrib')
    plt.legend(('"arb" prec', 'floating'))
    plt.savefig(f"{plt_path}{os.sep}one_delta_bifurcation.png", dpi=300)
    plt.show()

    plt.plot(deltas, std.T)
    plt.legend(('"arb" prec', 'floating'))
    plt.show()

    # ...assess convergence
    # plt.plot(deltas, sum[1, ...])
    # plt.show()


    # -----------------------------------------------------
    # show some iterations on two example maps
    #  
    pname = f".{os.sep}{plt_path}"
    x0 = 0.399
    n_iter = 25
    deltas = (0.42, 0.72)

    for delta in deltas:

        # for storing the trajectory
        x_iter, a_iter = dd_tentmap_singledelta(x=x0, delta=delta, n_iter=n_iter)
        x_00 = np.array([x0, x0])
        xy = np.concatenate((x_00, np.stack((x_iter, x_iter)).T.flatten()))

        fname = f"{pname}{os.sep}iter_atentmap_x0-{int(1000 * x0):04d}_d-{int(1000 * delta):04d}"
        
        plt.figure(figsize=(4, 4))
        plot_dd_tentmap_singledelta(delta)
        plt.plot(xy[:-2], xy[1:-1], 'bo-')
        plt.xlabel('x_n')
        plt.ylabel('x_(n+1)')
        plt.savefig(fname + '.png', dpi=300)
        plt.show()


    # -----------------------------------------------------
    # show some 2D-trajectories...
    #
    # ...results are not really insightful. Commented out!
    #  
    
    # delta = 0.6
    # n_iter = 1000
    # x1 = 0.83244565
    # x2 = 0.64553654
    # x_iter, a_iter = dd_tentmap_doubledelta(
    #     x1=x1, x2=x2, delta=delta, n_iter=n_iter)
    
    # plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k-')
    # plt.plot(x_iter[0, ...], x_iter[1, ...], 'ko')
    # plt.show()


    # -----------------------------------------------------
    # show some 2D-trajectories...
    #
    # ...results are not really insightful. Commented out!
    # There is also the problem with limited precision/rational
    # numbers, which should be solved for a=2!
    #  
    
    # delta = 0.47
    # n_iter = 100
    # n_init = 500

    # n1 = 999
    # n2 = 999

    # x1_range = (0.5 + np.arange(n1)) / n1
    # x2_range = (0.5 + np.arange(n2)) / n2

    # a_iter_std = np.zeros((n1, n2))

    # for i1, x1 in enumerate(tqdm(x1_range)):
    #     for i2, x2 in enumerate(x2_range):

    #         x0 = np.array((x1, x2))

    #         # discrete sim with constant density initial condition
    #         t0 = time.perf_counter()
    #         x_iter, a_iter = dd_tentmap_doubledelta(x1=x1, x2=x2, delta=delta, n_iter=n_init + n_iter)
    #         t1 = time.perf_counter()

    #         a_iter_std[i1, i2] = a_iter[n_init:].std()

    # plt.pcolor(a_iter_std)
    # plt.colorbar()
    # plt.savefig('test.png', dpi=1200)
    # plt.show()



# %%




