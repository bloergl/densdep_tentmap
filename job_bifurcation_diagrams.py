# %%
#
# simulate some chaotic systems
#
import sim_tools as st
import numpy as np
import matplotlib.pyplot as plt
from types import FunctionType
from numba import njit
import time
import os


@njit
def radic_map(state, params):

    return (params * state) % 1


@njit
def tent_map(state, params):

    if state > 0.5:
        return params * (1 - state)
    else:
        return params * state


@njit
def logistic_map(state, params):

    return params * state * (1 - state)


@njit
def rand_map(state: float, params: float):

    return params * state * float(np.random.random()) / params / state


def iterate(system: FunctionType, x0: float, a: float, n: int):

    x = np.zeros((n + 1,))
    x[0] = x0

    for i in range(n):
        x[i + 1] = system(x[i], a)

    return x[1:]


path_res = 'Results'
path_plt = 'Plots'
os.makedirs(f'.{os.sep}{path_plt}', exist_ok=True)

n_iter = 100
n_init = 500

map = 'logistic'
# map = 'tent'

# map = f"{path_res}{os.sep}a_iterations_A0000_a0500-1000-501_N{1000}-{100}_Mvolume"

map_name = map
match map:

    case 'tent':
        tent_n = 200
        tent_start = 1.0
        tent_stop = 2.0  # this point is EXCLUDED!
        tent_range = tent_start + (tent_stop - tent_start) * np.arange(tent_n) / tent_n
        tent_x = np.zeros((tent_n, n_iter))
        x0 = 0.5
        for i in range(tent_n):
            tent_x[i, :] = iterate(tent_map, x0, tent_range[i], n_iter + n_init)[n_init:]
        a_range = tent_range
        x_n = tent_x

    case 'logistic':
        logi_n = 200
        logi_start = 2.9
        logi_stop = 4.0  # this point is EXCLUDED!
        logi_range = logi_start + (logi_stop - logi_start) * np.arange(logi_n) / logi_n
        logi_x = np.zeros((logi_n, n_iter))
        x0 = 0.5
        for i in range(logi_n):
            logi_x[i, :] = iterate(logistic_map, x0, logi_range[i], n_iter + n_init)[n_init:]
        a_range = logi_range
        x_n = logi_x

    case _:
        print(f"Interpreting map named '{map}' as filename!")
        res = np.load(map + ".npz")
        x_n = res["a_history"]
        a_range = res["delta"]
        # t_history = res["t_history"]
        map_name = 'densdep-hat'


"""
# try to get fractal dimension by computing box count
n = a_range.size
n_boco_iter = 100
boco = np.zeros((n, n_boco_iter))
for i in range(n):
    if np.unique(x_n[i, :]).size == 1:
        boco[i, :] = 0
    else:
        tmp = st.boxcount(x_n[i, :], np.arange(1, n_boco_iter + 1))
        boco[i, :] = tmp[0, :] / tmp[1, :] 
"""
        
st.plot_bifurcations(a_range, x_n)
plt.xlabel('control parameter a')
plt.ylabel('iterations x_n')
plt.title(f"{map_name} map")
plt.savefig(f'{path_plt}{os.sep}{map_name}_map.png', dpi=600)
plt.show()



# %%
QUICK_PLOT = False

path_res = 'Results'
path_plt = 'Plots'

# load file and extract results
# name = "Results/a_iterations_A0000_a0500-1000-501_N1000-100_Mdiscrete_nd100000"
n_iter = 1000
n_init = 100
name = f"a_iterations_A0000_a0500-1000-501_N{n_iter}-{n_init}_Mvolume"
res = np.load(f"{path_res}{os.sep}{name}.npz")
a_history = res["a_history"]
t_history = res["t_history"]
delta = res["delta"]

# try to get fractal dimension by computing box count
n = delta.size
n_boco_iter = 100
boco = np.zeros((n, n_boco_iter))
for i in range(n):
    if delta[i] == 1:
        boco[i, :] = 0
    else:
        tmp = st.boxcount(a_history[i, :], np.arange(1, n_boco_iter + 1))
        boco[i, :] = tmp[0, :] / tmp[1, :] 


if QUICK_PLOT:
    
    plt.figure(figsize=(10, 7))

    plt.subplot(4, 1, (1, 3))
    plt.plot(delta, a_history, 'k.', ms=0.05)
    # plt.plot(delta, 2 * np.sqrt(delta), 'b-')
    plt.xlabel('delta')
    plt.ylabel('a_n')

    plt.subplot(4, 1, 4)
    plt.plot(delta, boco[:, -1])
    plt.xlabel('delta')
    plt.ylabel('boxcount dim')

    plt.savefig(f"{path_plt}{os.sep}{name}_QUICK.svg", orientation='landscape')
    plt.savefig(f"{path_plt}{os.sep}{name}_QUICK.png", dpi=1200, orientation='landscape')
    plt.show()

else:

    s = np.arange(n, dtype=np.int64)

    plt.figure(figsize=(10, 7))

    plt.subplot(4, 1, (1, 3))
    st.plot_bifurcations(delta[s], a_history[s, :])
    plt.xlim((delta[s[0]], delta[s[-1]]))
    plt.xlabel('delta')
    plt.ylabel('a_n')
    plt.title(f"n_iter={n_iter}, n_init={n_init}")

    plt.subplot(4, 1, 4)
    plt.plot(delta[s], boco[s, -1])
    plt.xlim((delta[s[0]], delta[s[-1]]))
    plt.xlabel('delta')
    plt.ylabel('boxcount dim')

    print("Saving figure as png...")
    t0 = time.perf_counter()
    plt.savefig(f"{path_plt}{os.sep}{name}.png", dpi=1200, orientation='landscape')
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"...done in {dt:.03f} secs!")

    print("Saving figure as svg...")
    t0 = time.perf_counter()
    plt.savefig(f"{path_plt}{os.sep}{name}.svg", orientation='landscape')
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"...done in {dt:.03f} secs!")

    print("Showing figure...")
    t0 = time.perf_counter()
    plt.show()
    t1 = time.perf_counter()
    dt = t1 - t0   
    print(f"...done in {dt:.03f} secs!")


"""
if __name__ == "__main__":
    import pynamicalsys
    from tqdm import tqdm

    tent_n = 200
    tent_range = 0 + 2 * np.arange(tent_n) / (tent_n - 1)
    x0 = 0.5  # 126436586861264623974
    n = 1000
    lamb_range = np.zeros((tent_n,))

    ds = pynamicalsys.DiscreteDynamicalSystem(mapping=tent_map, system_dimension=1, number_of_parameters=1)
    for i, a in enumerate(tent_range):

        # ds.se((a,))
        x = ds.trajectory(x0, n, parameters=float(a))
        # x2 = iterate(tent_map, x0, a, n)
        
        plt.plot(a * np.ones((n,)), x, 'k.', ms=0.1)
        # lamb_range[i] = nolds.lyap_e(x)[0]
        
        output = ds.lyapunov(x0, n, parameters=a, transient_time=100, return_history=True)
        lamb_range[i] = output[-1]

        # print(x1 - x2)
    plt.show()
    
    plt.plot(tent_range, lamb_range)
    plt.xlabel('a')
    plt.ylabel('Lyapunov exponent')
    plt.show()
"""
