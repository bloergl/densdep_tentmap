# %%
import cv2
# import numpy as np
# import matplotlib.pyplot as plt


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

    # file_pats = (
    #     'box_merging/*none*png',
    #     'box_merging/*tight*png',
    #     'ensemble/*none*png',
    #     'ensemble/*tight*png'
    # )
    # file_nams = ('box_merging/returnmap_video_none.mp4')

    # get all files
    files_box_none = glob.glob('ReturnMap/SimVolume/*none*png')
    files_box_tight = glob.glob('ReturnMap/SimVolume/*tight*png')
    files_ens_none = glob.glob('ReturnMap/SimDiscrete/*none*png')
    files_ens_tight = glob.glob('ReturnMap/SimDiscrete/*tight*png')

    # sort file names
    files_box_none.sort()
    files_box_tight.sort()
    files_ens_none.sort()
    files_ens_tight.sort()

    files2video(files_box_none, filename='ReturnMap/SimVolume/movie_volume_none.mp4')
    # files2video(files_box_tight, filename='ReturnMap/SimVolume/movie_volume_tight.mp4')
    # files2video(files_ens_none, filename='ReturnMap/SimDiscrete/movie_discrete_none.mp4')
    # files2video(files_ens_tight, filename='ReturnMap/SimDiscrete/movie_discrete_tight.mp4')
