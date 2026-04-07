# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_bifurcations(x: np.ndarray, a_xr: np.ndarray, lw=None, dx=None):

    # get and check input dimensions
    nx = x.size
    nr = a_xr.shape[-1]
    assert (x.ndim == 1) or (x.ndim == 0), "'x' must be 1- or 2-dimensional!"
    assert (a_xr.ndim == 1) or (a_xr.ndim == 2), "'a_xr' must have 1 or 2 dims!"
    if a_xr.ndim == 1:
        assert nx == 1, "'x' must match to 'a_xr'"
        a_xr = np.reshape(a_xr, (1, nr))
        x = np.array((x,))
    else:
        assert nx == a_xr.shape[0], "'x' must match to 'a_xr'"

    if lw is None:
        lw = np.sqrt(1.0 / nr)

    if dx is None:
        if nx == 1:
            dx = 1
        else:
            dx = np.diff(x).mean()
    x_bar = np.array((-dx / 2, dx / 2))

    a_rep = np.reshape(np.array((1.0, 1.0)), [2, 1])
    for ix in tqdm(range(nx)):
        y_vals = a_xr[ix, :] * a_rep
        x_vals = x[ix] + x_bar
        plt.plot(x_vals, y_vals, 'k-', linewidth=lw)

    return


def boxcount(states: np.ndarray, ns: np.ndarray) -> np.ndarray:

    if (states.ndim < 1) or (states.ndim > 2):
        raise ValueError('"states" has to be a one- or two-dimensional array!')
    if states.ndim == 1:
        states = np.reshape(states, (1, states.size))

    n_dim = states.shape[0]
    n_pts = states.shape[1]

    if (n_dim < 1) or (n_dim > 3):
        raise ValueError('coordinates in "states" must be 1, 2, or 3-dim!')

    states_min = np.min(states, axis=-1, keepdims=True)
    states_max = np.max(states, axis=-1, keepdims=True)
    states_d = 2.0 * (states_max - states_min)
    states_norm = (states.copy() - states_min) / states_d + 0.005

    n_depth = ns.size
    nom_denom = np.zeros((2, n_depth))
    for i in range(n_depth):

        n = ns[i]
        idcs = np.floor(n * states_norm).astype(np.int64)
        match n_dim:
            case 1:
                vol = np.zeros((n,))
                vol[idcs[0]] = 1
            case 2:
                vol = np.zeros((n, n))
                vol[idcs[0], idcs[1]] = 1
            case 3:
                vol = np.zeros((n, n, n))
                vol[idcs[0], idcs[1], idcs[2]] = 1
            case _:
                raise ValueError("There's something fishy with 'states'-dimensionality...")

        nom_denom[1, i] = np.log10(n)
        nom_denom[0, i] = np.log10(np.sum(vol))

    return nom_denom


def files2video(files, fps=25, filename=None):

    for i_file, file in enumerate(files):

        image = cv2.imread(file)

        if i_file == 0:  # first file, open video file (and display window)
            s = image.shape
            if filename is None:
                win_name = "Preview"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, s[0], s[1])
            else:
                if len(s) < 3:
                    color = False
                else:
                    if s[2] == 1:
                        color = False
                    else:
                        color = True
                print(f"Processing file {file}...") 
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v
                video = cv2.VideoWriter(filename, fourcc, fps, (s[0], s[1]), color)

        if filename is None:
            cv2.imshow(win_name, image)
            q = cv2.waitKey(max(1, int(1000 / fps)))
            if q == 113:
                break
        else:
            video.write(image)

    if filename is None:
        cv2.destroyWindow(win_name)
    else:
        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    import glob
    import os

    plt_path = 'Plots'
    mov_path = 'ReturnMap'

    # file_pats = (
    #     'box_merging/*none*png',
    #     'box_merging/*tight*png',
    #     'ensemble/*none*png',
    #     'ensemble/*tight*png'
    # )
    # file_nams = ('box_merging/returnmap_video_none.mp4')

    # get all files
    files_box_none = glob.glob(f'{plt_path}{os.sep}{mov_path}{os.sep}SimVolume{os.sep}*none*png')
    files_box_tight = glob.glob(f'{plt_path}{os.sep}{mov_path}{os.sep}SimVolume{os.sep}*tight*png')
    files_ens_none = glob.glob(f'{plt_path}{os.sep}{mov_path}{os.sep}SimDiscrete{os.sep}*none*png')
    files_ens_tight = glob.glob(f'{plt_path}{os.sep}{mov_path}{os.sep}SimDiscrete{os.sep}*tight*png')

    # sort file names
    files_box_none.sort()
    files_box_tight.sort()
    files_ens_none.sort()
    files_ens_tight.sort()

    files2video(files_box_none, filename=f'{plt_path}{os.sep}{mov_path}{os.sep}SimVolume{os.sep}movie_volume_none.mp4')
    files2video(files_box_tight, filename=f'{plt_path}{os.sep}{mov_path}{os.sep}SimVolume{os.sep}movie_volume_tight.mp4')
    files2video(files_ens_none, filename=f'{plt_path}{os.sep}{mov_path}{os.sep}SimDiscrete{os.sep}movie_discrete_none.mp4')
    files2video(files_ens_tight, filename=f'{plt_path}{os.sep}{mov_path}{os.sep}SimDiscrete{os.sep}movie_discrete_tight.mp4')
