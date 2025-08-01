
from __future__ import division
import os
import glob
import numpy
from matplotlib import pyplot
import pycwt as wavelet
from pycwt.helpers import find
import pandas as pd
import numpy as np
import cv2
import scipy.io

from tqdm import tqdm
from io import BytesIO


def cwt_function(dat):
    t0 = 0
    dt = 0.1
    N = dat.size
    t = np.arange(0, N) * dt + t0

    # Quadratic detrending
    p = np.polyfit(t - t0, dat, 2)
    dat_notrend = dat - np.polyval(p, t - t0)

    # Check trend magnitude after detrending
    trend_magnitude = np.abs(np.polyfit(t - t0, dat_notrend, 1)[0])
    if trend_magnitude > 0.3:
        print(f"Skipping segment due to high post-detrend trend: {trend_magnitude}")
        return None

    std = dat_notrend.std()
    if std == 0 or np.isnan(std):
        print("Standard deviation is zero or NaN. Skipping normalization.")
        return None

    dat_norm = dat_notrend / std
    mother = wavelet.Morlet(6)
    s0 = 2 * dt
    dj = 1 / 12
    J = 7 / dj

    # Robust AR(1) estimation
    try:
        alpha, _, _ = wavelet.ar1(dat_notrend)
    except Warning as w:
        print(f"Skipping segment due to AR(1) warning: {w}")
        return None

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    power /= scales[:, None]
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    pyplot.close('all')
    pyplot.ioff()
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    pyplot.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap='hsv')
    extent = [t.min(), t.max(), 0, max(period)]
    pyplot.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=0, extent=extent)

    pyplot.gca().set_axis_off()
    pyplot.margins(0,0)
    pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
    pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())

    buf = BytesIO()
    # pyplot.savefig("filename.jpg", bbox_inches = 'tight', pad_inches = 0)
    pyplot.savefig(buf, bbox_inches = 'tight', pad_inches = 0)
    pyplot.close()
    buf.seek(0)

    # image = cv2.imread("filename.jpg")
    image_array = np.frombuffer(buf.read(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image




def cwt_make(data, label):
    images = []

    excluded_indexes = [1, 4, 6, 7, 8, 12, 13, 17, 18, 23, 26, 32, 36, 37, 38, 39]
    
    for i, idx in enumerate(range(40)):
        if i not in excluded_indexes:
            img = cwt_function(data[i])
            if img is not None:
                images.append(img)
            else:
                print(f"Segment {i} skipped (image is None)")

    # Check if you have enough images to form the grid
    if len(images) < 24:
        print(f"Not enough images to form a 4x6 grid (have {len(images)}). Skipping this block.")
        return None, None

    # Only use the first 24 images to form a 4x6 grid
    images = images[:24]
    im_list_2d = [images[i*6:(i+1)*6] for i in range(4)]
    im_h = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    re_size = 224
    final_image = cv2.resize(im_h, (re_size, re_size), interpolation=cv2.INTER_CUBIC)

    return [np.array(final_image), np.array(label)]





if __name__ == "__main__":

    SAMPLING_FREQUENCY = 500
    ROOT_DIR = "D:/My EEG Work/EEG M.Sc Ongoing Work/EEG_Nahid Vai/MAT Fatigue Dataset"

    for filename in sorted(os.listdir(ROOT_DIR)):
        index = filename.split("_")[2].split(".")[0]
        label_name = filename.split("_")[0].lower()

        mat = scipy.io.loadmat(os.path.join(ROOT_DIR,filename))

        filedata = mat[list(mat.keys())[3]]

        print(f"File No: {index} - Label: {label_name}")
        
        for i in tqdm(range(filedata.shape[1] // SAMPLING_FREQUENCY)):
        
            j = i * SAMPLING_FREQUENCY
            k = (i+1) * SAMPLING_FREQUENCY

            data = filedata[:,j:k]
            image_final, label = cwt_make(data, label_name)
            if image_final is None:
                print("Skipping saving due to insufficient images.")
                continue

            directory = f"Images/{index}/{label_name}/{i}.png"
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            cv2.imwrite(directory, image_final)


