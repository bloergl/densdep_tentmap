'''
corr.py

Provides code for computing cross/auto-correlation

Version 1.0
12.08.2024
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm


def periodic_axis(n):

    x = np.arange(n)
    y = np.array([xi if xi <= n / 2 else -(n - xi) for xi in x])

    return y


def cross_periodic(a, b=None, method="fft", n_dims=1, n_limit=None):

    # auto- or crosscorrelation?
    if b is None:
        auto = True
        b = a
    else:
        auto = False

    # check if method and boundary condition arguments are correct
    assert (method == "sum") or (method == "fft"), "Valid methods: sum, fft!"
    assert (n_dims == 1) or (n_dims == 2), "Only 1D or 2D correlations possible!"

    # check shape consistency
    ash = list(a.shape)
    bsh = list(b.shape)
    assert (len(ash) >= n_dims) and (len(ash) <= n_dims + 1), \
        f"'a' must have {n_dims} or {n_dims + 1} dimensions!"
    assert (len(bsh) >= n_dims) and (len(bsh) <= n_dims + 1), \
        f"'b' must have {n_dims} or {n_dims + 1} dimensions!"
    assert any(np.array(ash[:n_dims]) == np.array(bsh[:n_dims])), \
        "Both signals must have same size!"

    # bring both arrays to a consistent 2D/3D-form
    if len(ash) > n_dims:
        na = ash[-1]
    else:
        na = 1
        ash.append(na)
    if len(bsh) > n_dims:
        nb = bsh[-1]
    else:
        nb = 1
        bsh.append(nb)

    a = a.reshape(ash)
    b = b.reshape(bsh)

    # get number of elements in correlation dimensions
    # limit computation to +/-n_limit bins around 0
    if n_dims == 1:
        n = ash[0]
        if n_limit is None:
            n_limit = (n - 1) // 2
        corr = np.zeros((n_limit * 2 + 1, na, nb))
        corr_axis = periodic_axis(n_limit * 2 + 1)

    elif n_dims == 2:
        nx = ash[0]
        ny = ash[1]
        if n_limit is None:
            nx_limit = (nx - 1) // 2
            ny_limit = (ny - 1) // 2
        else:
            nx_limit = n_limit[0]
            ny_limit = n_limit[1]

        corr = np.zeros((nx_limit * 2 + 1, ny_limit * 2 + 1, na, nb))
        corr_axisx = periodic_axis(nx_limit * 2 + 1)[:, np.newaxis]
        corr_axisy = periodic_axis(ny_limit * 2 + 1)[np.newaxis, :]

    if method == "sum":

        # loop it!
        for ia in range(na):
            for ib in range(nb):

                if n_dims == 1:
                    for i in range(-nx_limit, nx_limit + 1):
                        corr[i, ia, ib] = (np.roll(a[:, ia], -i) * b[:, ib]).mean(axis=0)

                elif n_dims == 2:
                    for ix in range(-nx_limit, nx_limit + 1):
                        for iy in range(-ny_limit, ny_limit + 1):
                            corr[ix, iy, ia, ib] = (np.roll(np.roll(a[:, :, ia], -ix, axis=0), -iy, axis=1) * b[:, :, ib]).mean(axis=(0, 1))

        if n_dims == 1:
            corr = corr[corr_axis, :, :]
            return np.squeeze(corr), corr_axis
        elif n_dims == 2:
            corr = corr[corr_axisx, corr_axisy, :, :]
            return np.squeeze(corr), corr_axisx, corr_axisy

    if method == "fft":

        if n_dims == 1:

            b_flip = np.zeros((n, nb))
            b_flip[0, :] = b[0, :]
            b_flip[-1:-n:-1, :] = b[1:, :]

            # pre-compute fft
            a_fft = np.fft.fft(a, axis=0)
            b_flip_fft = np.fft.fft(b_flip, axis=0)

            # loop it!
            c_norm = n
            for ia in range(na):
                # print(f"{ia+1} of {na}: ", end="")
                # for ib in tqdm(range(nb)):
                for ib in range(nb):
                    if auto and (ib < ia):
                        corr[:, ia, ib] = np.flip(corr[:, ib, ia])
                    else:
                        tmp = np.real(np.fft.ifft(a_fft[:, ia] * b_flip_fft[:, ib]))
                        corr[:, ia, ib] = (tmp / c_norm)[corr_axis]

            return np.squeeze(corr), corr_axis

        elif n_dims == 2:

            b_flip = np.zeros((nx, ny, nb))
            b_tmp = np.zeros((nx, ny, nb))
            b_tmp[0, :, :] = b[0, :, :]  # copy X=0
            b_tmp[-1:-nx:-1, :, :] = b[1:, :, :]  # copy and flip X
            b_flip[:, 0, :] = b_tmp[:, 0, :]  # copy Y=0
            b_flip[:, -1:-ny:-1, :] = b_tmp[:, 1:, :]  # copy and flip Y

            # pre-compute fft
            a_fft = np.fft.fft2(a, axes=(0, 1))
            b_flip_fft = np.fft.fft2(b_flip, axes=(0, 1))

            # loop it!
            c_norm = nx * ny
            for ia in range(na):
                # print(f"{ia+1} of {na}: ", end="")
                # for ib in tqdm(range(nb)):
                for ib in range(nb):
                    if auto and (ib < ia):
                        corr[:, :, ia, ib] = np.flip(corr[:, :, ib, ia], axis=(0, 1))
                    else:
                        tmp = np.real(np.fft.ifft2(a_fft[:, :, ia] * b_flip_fft[:, :, ib]))
                        corr[:, :, ia, ib] = (tmp / c_norm)[corr_axisx, corr_axisy]

            return np.squeeze(corr), corr_axisx, corr_axisy


def cross_open(a, b=None, method="fft", n_dims=1, n_limit=None):

    # auto- or crosscorrelation?
    if b is None:
        auto = True
        b = a
    else:
        auto = False

    # check if method arguments are correct
    assert (method == "sum") or (method == "fft"), "Valid methods: sum, fft!"
    assert (n_dims == 1) or (n_dims == 2), "Only 1D or 2D correlations possible!"

    # check shape consistency
    ash = list(a.shape)
    bsh = list(b.shape)
    assert (len(ash) >= n_dims) and (len(ash) <= n_dims + 1), \
        f"'a' must have {n_dims} or {n_dims + 1} dimensions!"
    assert (len(bsh) >= n_dims) and (len(bsh) <= n_dims + 1), \
        f"'b' must have {n_dims} or {n_dims + 1} dimensions!"
    assert any(np.array(ash[:n_dims]) == np.array(bsh[:n_dims])), \
        "Both signals must have same size!"

    # bring both arrays to a consistent 2D/3D-form
    if len(ash) > n_dims:
        na = ash[-1]
    else:
        na = 1
        ash.append(na)
    if len(bsh) > n_dims:
        nb = bsh[-1]
    else:
        nb = 1
        bsh.append(nb)

    a = a.reshape(ash)
    b = b.reshape(bsh)

    # get number of elements in correlation dimensions
    # limit computation to +/-n_limit bins around 0
    if n_dims == 1:
        n = ash[0]
        n_extd = n + (n - 1)
        if n_limit is None:
            n_limit = n - 1
        corr = np.zeros((n_limit * 2 + 1, na, nb))
        corr_axis = np.arange(n_limit * 2 + 1) - n_limit

    elif n_dims == 2:
        nx = ash[0]
        ny = ash[1]
        nx_extd = nx + (nx - 1)
        ny_extd = ny + (ny - 1)
        if n_limit is None:
            nx_limit = nx - 1
            ny_limit = ny - 1
        else:
            nx_limit = n_limit[0]
            ny_limit = n_limit[1]
        corr = np.zeros((nx_limit * 2 + 1, ny_limit * 2 + 1, na, nb))
        corr_axisx = np.arange(nx_limit * 2 + 1)[:, np.newaxis] - nx_limit
        corr_axisy = np.arange(ny_limit * 2 + 1)[np.newaxis, :] - ny_limit

    if method == "sum":

        # loop it!
        for ia in range(na):
            for ib in range(nb):

                if n_dims == 1:
                    for i in range(n_limit + 1):
                        corr[i, ia, ib] = (a[i:, ia] * b[:n - i, ib]).sum() / (n - i)
                    for i in range(1, n_limit + 1):
                        corr[-i, ia, ib] = (b[i:, ib] * a[:n - i, ia]).sum() / (n - i)

                elif n_dims == 2:
                    for ix in range(nx_limit + 1):
                        for iy in range(ny_limit + 1):
                            corr[+ix, +iy, ia, ib] = (a[ix:, iy:, ia] * b[:nx - ix, :ny - iy, ib]).sum() / (nx - ix) / (ny - iy)
                        for iy in range(1, ny_limit + 1):
                            corr[+ix, -iy, ia, ib] = (a[ix:, :ny - iy, ia] * b[:nx - ix, iy:, ib]).sum() / (nx - ix) / (ny - iy)
                    for ix in range(1, nx_limit + 1):
                        for iy in range(ny_limit + 1):
                            corr[-ix, +iy, ia, ib] = (a[:nx - ix, iy:, ia] * b[ix:, :ny - iy, ib]).sum() / (nx - ix) / (ny - iy)
                        for iy in range(1, ny_limit + 1):
                            corr[-ix, -iy, ia, ib] = (a[:nx - ix, :ny - iy, ia] * b[ix:, iy:, ib]).sum() / (nx - ix) / (ny - iy)

        if n_dims == 1:
            corr = corr[corr_axis, :, :]
            return np.squeeze(corr), corr_axis
        elif n_dims == 2:
            corr = corr[corr_axisx, corr_axisy, :, :]
            return np.squeeze(corr), corr_axisx, corr_axisy

    if method == "fft":

        if n_dims == 1:
            a_extd = np.zeros((n_extd, na))
            a_extd[:n, :] = a

            b_extd = np.zeros((n_extd, nb))
            b_extd[0, :] = b[0, :]
            b_extd[-1:-n:-1, :] = b[1:, :]

            # pre-compute fft
            a_ext_fft = np.fft.fft(a_extd, axis=0)
            b_ext_fft = np.fft.fft(b_extd, axis=0)

            # loop it!
            c_norm = np.maximum(n - np.arange(n_extd), np.arange(n_extd) - (n - 1))
            for ia in range(na):
                # print(f"{ia+1} of {na}: ", end="")
                # for ib in tqdm(range(nb)):
                for ib in range(nb):
                    if auto and (ib < ia):
                        corr[:, ia, ib] = np.flip(corr[:, ib, ia])
                    else:
                        tmp = np.real(np.fft.ifft(a_ext_fft[:, ia] * b_ext_fft[:, ib]))
                        corr[:, ia, ib] = (tmp / c_norm)[corr_axis]

            return np.squeeze(corr), corr_axis

        elif n_dims == 2:
            a_extd = np.zeros((nx_extd, ny_extd, na))
            a_extd[:nx, :ny, :] = a

            b_extd = np.zeros((nx_extd, ny_extd, nb))
            b_tmp = np.zeros((nx_extd, ny, nb))
            b_tmp[0, :, :] = b[0, :, :]  # copy X=0
            b_tmp[-1:-nx:-1, :, :] = b[1:, :, :]  # copy and flip X
            b_extd[:, 0, :] = b_tmp[:, 0, :]  # copy Y=0
            b_extd[:, -1:-ny:-1, :] = b_tmp[:, 1:, :]  # copy and flip Y

            # pre-compute fft
            a_ext_fft = np.fft.fft2(a_extd, axes=(0, 1))
            b_ext_fft = np.fft.fft2(b_extd, axes=(0, 1))

            # loop it!
            c_normx = (np.maximum(nx - np.arange(nx_extd), np.arange(nx_extd) - (nx - 1)))[:, np.newaxis]
            c_normy = (np.maximum(nx - np.arange(nx_extd), np.arange(nx_extd) - (nx - 1)))[np.newaxis, :]
            c_norm = c_normx * c_normy
            for ia in range(na):
                # print(f"{ia+1} of {na}: ", end="")
                # for ib in tqdm(range(nb)):
                for ib in range(nb):
                    if auto and (ib < ia):
                        corr[:, :, ia, ib] = np.flip(corr[:, :, ib, ia], axis=(0, 1))
                    else:
                        tmp = np.real(np.fft.ifft2(a_ext_fft[:, :, ia] * b_ext_fft[:, :, ib]))
                        corr[:, :, ia, ib] = (tmp / c_norm)[corr_axisx, corr_axisy]

            return np.squeeze(corr), corr_axisx, corr_axisy


def cross(a, b=None, method="fft", boundary='open', n_dims=1, n_limit=None):

    assert (boundary == "open") or (boundary == "periodic"), "Valid boundary: open, periodic!"
    if boundary == "open":
        res = cross_open(a, b=b, method=method, n_dims=n_dims, n_limit=n_limit)
    elif boundary == "periodic":
        res = cross_periodic(a, b=b, method=method, n_dims=n_dims, n_limit=n_limit)

    return res


def cross_multi(a, b=None, method="fft", boundary="open", n_dims=1, n_limit=None):

    n_avg = a.shape[-1]
    if b is None:
        pass
    else:
        assert n_avg == b.shape[-1], "Last dimensions of a, b must be same!"

    for i_avg in range(n_avg):
        if b is None:
            b_i_avg = None
        else:
            b_i_avg = b[..., i_avg]
        res = cross(a[..., i_avg], b=b_i_avg, method=method, boundary=boundary, n_dims=n_dims, n_limit=n_limit)
        if i_avg == 0:
            cor = np.zeros(res[0].shape + (n_avg,))
        cor[..., i_avg] = res[0]

    if len(res) > 2:
        return cor, res[1], res[2]
    else:
        return cor, res[1]


if __name__ == "__main__":

    print("Testing cross-correlation in two dimensions...")

    a = np.zeros((10, 10))
    b = np.zeros((10, 10))

    a[2, 2] = 1
    b[5, 6] = 3

    cof, xof, yof = cross(a, b, n_dims=2, method='fft', boundary='open')
    cos, xos, yos = cross(a, b, n_dims=2, method='sum', boundary='open')
    cpf, xpf, ypf = cross(a, b, n_dims=2, method='fft', boundary='periodic')
    cps, xps, yps = cross(a, b, n_dims=2, method='sum', boundary='periodic')

    plt.imshow(cof)
    plt.xticks(np.arange(xof.size), labels=xof.flatten())
    plt.yticks(np.arange(yof.size), labels=yof.flatten())
    plt.colorbar()
    plt.show()

    plt.imshow(cpf)
    plt.xticks(np.arange(xpf.size), labels=xpf.flatten())
    plt.yticks(np.arange(ypf.size), labels=ypf.flatten())
    plt.colorbar()
    plt.show()

    print("Testing one-dimensional correlations...")

    n = 4
    rng = np.random.default_rng(42)

    # check 1D
    # a = rng.random((n,))
    # b = rng.random((n,))
    # a = np.array((1, 2, 0, 0))
    # b = np.array((0, 0, 1, 0))
    a = np.array((0.0, 0.0, 2.0, 0, 0, 0))
    b = np.array((2.0, 1.0, 0.5, 0, 0, 0))
    # a = np.array((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    # b = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    c_fft, c_axis = cross(a, b, method="fft")
    c_sum, c_axis = cross(a, b, method="sum")

    z_norm = np.correlate(np.ones_like(a), np.ones_like(b), 'full')
    plt.plot(c_axis, c_fft, 'ro')
    plt.plot(c_axis, c_sum, 'bx')
    plt.plot(c_axis, np.correlate(a, b, 'full') / z_norm, '-')
    plt.plot(c_axis, c_fft - c_sum)
    plt.show()

    # check 2D, cross
    a = rng.random((n, 2))
    b = rng.random((n, 3))

    c_fft, c_axis = cross(a, b, method="fft")
    c_sum, c_axis = cross(a, b, method="sum")

    plt.plot(c_fft.flatten(), 'ro')
    plt.plot(c_sum.flatten(), 'bx')
    plt.plot((c_fft - c_sum).flatten())
    plt.show()

    # check 2D, auto
    a = rng.random((n, 2))

    c_fft, c_axis = cross(a, method="fft")
    c_sum, c_axis = cross(a, method="sum")

    plt.plot(c_fft.flatten(), 'ro')
    plt.plot(c_sum.flatten(), 'bx')
    plt.plot((c_fft - c_sum).flatten())
    plt.show()

    # check 2D, auto, limit
    a = rng.random((n, 2))

    c_fft, c_axis = cross(a, method="fft", n_limit=2)
    c_sum, c_axis = cross(a, method="sum", n_limit=2)

    plt.plot(c_fft.flatten(), 'ro')
    plt.plot(c_sum.flatten(), 'bx')
    plt.plot((c_fft - c_sum).flatten())
    plt.show()

    # get advantage over np.correlate
    import time

    nns = (10 ** np.array((2, 3, 4, 5))).astype(np.int64)
    ns = nns.size
    x = np.random.random(nns.max())
    y = np.random.random(nns.max())

    for i, n in enumerate(nns):

        print(f"Vector lenth: {n}")

        t1 = time.perf_counter()
        _ = np.correlate(x[:n], y[:n], 'full')
        t2 = time.perf_counter()
        dt1 = t2 - t1

        t1 = time.perf_counter()
        _ = cross(x[:n], y[:n])
        t2 = time.perf_counter()
        dt2 = t2 - t1
        print(f"np.correlate vs. cross: {dt1:.4f} -- {dt2:.4f}: factor {dt1/dt2:.2f} x faster\n\n")
