# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import sim_volume as sv
import os


def get_steady_state():

    # outputs
    sy.symbols('a d v x0 x1 x2 x3 v1 v2 v3')

    # find a
    sy.symbols('ax')
    sols = sy.solve('ax * ax * ax - 2 * ax - 2')
    n_sols = len(sols)
    n_real = 0
    for i_sol in range(n_sols):
        sol = sols[i_sol]
        if sol.is_real:
            print(f"Found real solution: a={sol}")
            a = sol
            n_real += 1
    assert n_real == 1, "We are looking for ONE real solution only!"
    a = a.simplify()

    # compute d
    d = (3 * a ** 3 - 2 * a ** 2 - 5 * a + 5) / (a ** 2 * (1 + a))
    d = d.simplify()
    print(f"Corresponding solution: d={d}") 

    # compute all others
    v = ((a + 1) / (2 * a + 3)).simplify()
    v1 = (1 - 2 * v).simplify()
    v2 = v
    v3 = v

    x0 = (1 / (a * (1 + a))).simplify()
    x1 = (1 / (1 + a)).simplify()
    x2 = (a / (1 + a)).simplify()
    x3 = (1 - 1 / (a ** 2 * (1 + a))).simplify()

    return d, a, x0, x1, x2, x3, v1, v2, v3


d, a, x0, x1, x2, x3, v1, v2, v3 = get_steady_state()

x0f = x0.evalf()
x1f = x1.evalf()
x2f = x2.evalf()
x3f = x3.evalf()
v1f = v1.evalf()
v2f = v2.evalf()
v3f = v3.evalf()
df = d.evalf()
af = a.evalf()

x_dens: np.ndarray = np.array([x0f, x1f, x2f, x3f, 1.0])
v_dens: np.ndarray = np.array([0.0, v1f, v2f, v3f, 0.0])
a_iter, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
    x=x_dens,
    v=v_dens,
    a_uppercase=0.0,
    delta=d,
    n_iter=2,
    verbose=False,
    warning=False
)

path_plt = 'Plots'
os.makedirs(f'.{os.sep}{path_plt}', exist_ok=True)

plt.grid('both')
sv.plot_density(x_dens, v_dens)
plt.xlabel('x')
plt.ylabel('f_n(x)')
plt.savefig(f'{path_plt}{os.sep}invariant_density.png')
plt.show()

print(x_iter[0])
print(v_iter[0])
print(af, df)

sv.plot_density(x_iter[1], v_iter[1])
print()
print(x_iter[1])
print(v_iter[1])





"""


def get_fake_steady_state(a: float):

    d = (3 * a ** 3 - 2 * a ** 2 - 5 * a + 5) / (a ** 2 * (1 + a))
    v = (a + 1) / (2 * a + 3)

    x1 = 1 / (a * (1 + a))
    x2 = 1 / (1 + a)
    x3 = a / (1 + a)
    x4 = 1 - 1 / (a ** 2 * (1 + a))

    v1 = 1 - 2 * v
    v2 = v
    v3 = v

    assert 0 < d, "0 < delta not satisfied!"
    assert d < 1, "delta < 1 not satisfied!"
    assert 0 < x1, "0 < x1 not satisfied!"
    assert x1 < x2, "x1 < x2 not satisfied!"
    assert x2 < x3, "x2 < x3 not satisfied!"
    assert x3 < x4, "x3 < x4 not satisfied!"
    assert x4 < 1, "x4 < 1 not satisfied!"
    assert 0.5 < x3, "0.5 < x3 not satisfied!"
    assert x2 < 0.5, "x2 < 0.5 not satisfied!"
    assert x3 < d, "x3 < d not satisfied!"
    assert d < x4, "d < x4 not satisfied!"
    assert 0 < v, "0 < v not satisfied!"
    assert v < 1, "v < 1 not satisfied!"
    assert 0 < v1, "0 < v1 not satisfied!"

    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "v1": v1, "v2": v2, "v3": v3, "d": d, "a": a}

a = 1.769 # DEF 1.769

ssa = get_steady_state(a)

print(f"Steady state for a={a} would be:")
print(f"x1 = {ssa['x1']}")
print(f"x2 = {ssa['x2']}")
print(f"x3 = {ssa['x3']}")
print(f"x4 = {ssa['x4']}")
print(f"v1 = {ssa['v1']}")
print(f"v2 = {ssa['v2']}")
print(f"v3 = {ssa['v3']}")
print(f"d = {ssa['d']}")
print(f"a = {ssa['a']}")


# %%

x1 = 0.204
x2 = 0.361
x3 = 0.639
x4 = 0.884

v1 = 0.155
v2 = 0.425
v3 = 0.420

d  = 0.751

print(f"x1*a=x2: {x1 * a}, {x2}")
print(f"x2*a=x3: {x2 * a}, {x3}")
print(f"a*(1-x3)=x3: {a * (1 - x3)}, {x3}")
print(f"a*(1-x4)=x1: {a * (1 - x4)}, {x1}")

print(f"x1=1/a/(1+a): {x1}, {1 / (a * (1 + a))}")
print(f"x2=1/(1+a): {x2}, {1 / (1 + a)}")
print(f"x3=a/(1+a): {x3}, {a / (1 + a)}")
print(f"x4=1-1/a/a/(1+a): {x4}, {1 - 1 / (a * a * (1 + a))}")

print(f"v1=v3*(x2-x1)/(x3-x1): {v1}, {v3 * (x2 - x1) / (x3 - x1)}")
print(f"v2=v1+v3*(x3-x2)/(x3-x1): {v2}, {v1 + v3 * (x3 - x2) / (x3 - x1)}")
print(f"v3=v2: {v3}, {v2}")

print(f"a=v1+v2+v3*(d-x3)/(x4-x3)+1: {a}, {v1 + v2 + v3 * (d - x3) / (x4 - x3) + 1}")

print(f"v1=(x2-x1)/(2*x3+x2-3*x1): {v1}, {(x2 - x1) / (2 * x3 + x2 - 3 * x1)}")
print(f"v2=(x3-x1)/(2*x3+x2-3*x1): {v2}, {(x3 - x1) / (2 * x3 + x2 - 3 * x1)}")

print(f"v2=(a+1)/(2*a+3): {v2}, {(a + 1) / (2 * a + 3)}")
tmp = (a + 1) / (2 * a + 3)
print(f"a=2-v+v*(a*a*(a+1)*d-a*a*a)/(a*a-1): {a}, {2 - v2 + v2 * (a * a * (a + 1) * d - a * a * a)/(a * a - 1)}")
print(f"a=2-tmp+tmp*(a*a*(a+1)*d-a*a*a)/(a*a-1): {a}, {2 - tmp + tmp * (a * a * (a + 1) * d - a * a * a)/(a * a - 1)}")

print(f"a=2+1/(2*a+3)*(a*a*(1+a)*d-a*a*a-(a+1)*(a-1))/(a-1): {a}, {2+1/(2*a+3)*(a*a*(1+a)*d-a*a*a-(a+1)*(a-1))/(a-1)}")

print(f"(a-2)*(2*a+3)*(a-1)=a*a*(1+a)*d-a*a*a-(a+1)*(a-1): {(a-2)*(2*a+3)*(a-1)}, {a*a*(1+a)*d-a*a*a-(a+1)*(a-1)}")
print(f"d=(3*a*a*a-2*a*a-5*a-5)/(a*a*(1+a)): {d}, {(3 * a * a * a - 2 * a * a - 5 * a + 5) / (a * a * (1 + a))}")

print(f"0=a ** 4 - a ** 3 - 2 * a ** 2 + 2: {0}, {a ** 4 - a ** 3 - 2 * a ** 2 + 2}")


# %%
print(f"d=2-7/a/a+a/(a+1): {d}, {2 - 7 / a / a + a / (a + 1)}")


# %%
import sympy as sy

sy.symbols('a')
# sy.factor('3 * a * a * a + 2 * a * a - 5 * a + 5')
# sy.factor('a * a * a * a - a * a * a - 2 * a * a + 2')
# sy.factor('a * a * a - 2 * a * a - 2')
sols = sy.solve('a * a * a - 2 * a - 2')

# %%
import sim_volume as sv

a = 1.6

ssa = get_steady_state(a)

x_dens: np.ndarray = np.array([ssa["x1"], ssa["x2"], ssa["x3"], ssa["x4"], 1.0])
v_dens: np.ndarray = np.array([0.0, ssa["v1"], ssa["v2"], ssa["v3"], 0.0])
a_iter, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
    x=x_dens,
    v=v_dens,
    a_uppercase=0.0,
    delta=ssa["d"],
    n_iter=100,
    verbose=False,
    warning=False
)
print(x_iter[0])
print(v_iter[0])
print(ssa["a"], ssa["d"])

print()
print(x_iter[1])
print(v_iter[1])


# %%
import matplotlib.pyplot as plt
import numpy as np
import sim_volume as sv

delta = 0.750780527485225
a = 1.76929235423863

for n_bins in np.arange(2, 80):
    for n_reps in np.arange(10):

        # n_bins = 8
        v_dens = np.random.random((n_bins,))
        v_dens /= v_dens.sum()
        x_dens = np.random.random((n_bins,))
        x_dens.sort()
        x_dens /= x_dens[-1]

        a_iter, x_iter, v_iter, eps = sv.iterate_densdep_tent_map_vols(
            x=x_dens,
            v=v_dens,
            a_uppercase=0.0,
            delta=delta,
            n_iter=100,
            verbose=False,
            warning=False
        )

        iters = np.arange(a_iter.size)
        # plt.plot(iters, a_iter)

        sv.plot_density(x_iter[-1], v_iter[-1])
plt.show()


"""