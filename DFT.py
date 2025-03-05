#!/usr/bin/python
from math import *
import sys
import matplotlib.pyplot as plt

from cpe367_wav import cpe367_wav


############################################
############################################
# define routine for implementing a digital filter
def read_signal(fpath_wav_in: str, N: int) -> list[int] | bool:


    # construct objects for reading/writing WAV files
    #  assign each object a name, to facilitate status and error reporting
    wav_in = cpe367_wav('wav_in', fpath_wav_in)

    # open wave input file
    ostat = wav_in.open_wav_in()
    if not ostat:
        print('Cant open wav file for reading')
        return False


    xin = 0
    wave = []
    cnt = 0
    while xin is not None and cnt < N:
        # read next sample (assumes mono WAV file)
        #  returns None when file is exhausted
        xin = wav_in.read_wav()
        wave.append(xin)

        cnt += 1
    # close input and output files
    #  important to close output file - header is updated (with proper file size)
    wav_in.close_wav()
    return wave

def DFT(x: list[int], N: int) -> list[dict[str, float]]:
    X = [{"real" : 0, "imag" : 0, "magnitude" : 0} for _ in range(N)]
    for k in range(N):
        for n in range(N):
            X[k]["real"] += 1/N * x[n] * cos((2 * pi * k * n)/N)
            X[k]["imag"] += 1/N * x[n] * sin((2 * pi * k * n)/N)

    return X


def Magnitude(X: list[dict[str, float]]) -> list[float]:
    mags = []
    for k in X:
        k["magnitude"] = sqrt(k["real"]**2 + k["imag"]**2)
        mags.append(k["magnitude"])
    return mags

############################################
############################################
# define main program
def main():
    # check python version!
    major_version = int(sys.version[0])
    if major_version < 3:
        print('Sorry! must be run using python3.')
        print('Current version: ')
        print(sys.version)
        return False

    # grab file names
    fpath_wav_in = 'impulse.wav'
    N= 4000
    xn = read_signal(fpath_wav_in, N) #get x[n]
    X = DFT(xn, N) #take DFT
    Magnitude(X) #compute magnitude for each X[k]

    fs = 8000 #sampling rate
    mag_values = [X[k]["magnitude"] for k in range(1, 2000)] #list of magnitudes
    freq_axis = [k * fs//N for k in range(1,2000)]

    plt.plot(xn)
    plt.show()
    plt.plot(freq_axis, mag_values)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Magnitude Spectrum")
    plt.xlim([0, 2000])
    plt.show()

    return


############################################
############################################
# call main function
if __name__ == '__main__':
    main()
    quit()