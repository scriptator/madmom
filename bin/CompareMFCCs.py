#!/usr/bin/env python
# encoding: utf-8
"""
Evaluation script for the MFCC functionality

"""
import numpy as np
from madmom.audio import ShortTimeFourierTransform
from numpy import genfromtxt

from madmom.audio.cepstrogram import MFCC
from madmom.audio.filters import Filterbank
from madmom.audio.signal import Signal, FramedSignal

TESTFILE = './tuba_ff.wav'


def assert_df_equals_csv(df, csv_path, dtype=np.float32):
    loaded_df = genfromtxt(csv_path, dtype=dtype, delimiter=',')

    if df.shape == loaded_df.shape[::-1]:
        loaded_df = loaded_df.T

    difference = abs(df - loaded_df)
    if (difference > 0.01).any():
        raise AssertionError("{} != {}".format(df, loaded_df))


def convolve(data, filter, slice, mode="same"):
    return np.convolve(data, filter, mode=mode)[slice]


def main():
    # common parameters
    fs = 44100
    frame_len = 100  # [ms]
    hopsize = 20  # [ms]
    no_frame_samples = int(fs * frame_len / 1000)
    no_hop_samples = int(fs * hopsize / 1000)

    no_filters = 30

    pywav = Signal(TESTFILE, dtype=np.float32)
    assert_df_equals_csv(pywav, "./wav.csv")

    # Frame signal
    framed_pywav = FramedSignal(pywav, frame_size=no_frame_samples,
                                hop_size=no_hop_samples, origin="right")
    pyfft = ShortTimeFourierTransform(framed_pywav, window=np.hamming)
    # matlab enframe function:
    # By default, the number of frames will be rounded down to the nearest
    # integer and the last few samples of x() will be ignored unless its length
    # is lw more than a multiple of inc.
    pyfft = pyfft[:-5, :]
    assert_df_equals_csv(pyfft, "./fft.csv",
                         dtype=np.complex64)

    # create FFTs
    filterdata = genfromtxt("./melbank-44100.csv",
                            dtype=np.float32, delimiter=',').T
    filterbank = Filterbank(filterdata, pyfft.bin_frequencies)

    mfcc = MFCC(pyfft, filterbank=filterbank, num_bands=no_filters, fmin=0,
                fmax=fs / 2)
    assert_df_equals_csv(mfcc, "./dct.csv")

    deltastack = mfcc.calc_voicebox_deltas()
    assert_df_equals_csv(deltastack, "./mfcc.csv")

    assert (np.allclose(mfcc.deltas, deltastack[:, no_filters:2 * no_filters]))
    assert (np.allclose(mfcc.deltadeltas[1:-1],
                        deltastack[1:-1, no_filters * 2:]))
    # the first and last frame are different, because of the padding

    print("Congrats, the results are equal!")


if __name__ == '__main__':
    main()
