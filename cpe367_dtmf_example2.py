#!/usr/bin/python

import sys
from curses.ascii import isdigit

from DFT import DFT, Magnitude
from cpe367_sig_analyzer import cpe367_sig_analyzer
from my_fifo import my_fifo as fifo

from collections import Counter


############################################
############################################
# define routine for detecting DTMF tones
def process_wav(fpath_sig_in):
    ###############################
    # define list of signals to be displayed by and analyzer
    #  note that the signal analyzer already includes: 'symbol_val','symbol_det','error'
    more_sig_list = ['sig_1', 'sig_2']

    # sample rate is 4kHz
    fs = 4000

    # instantiate signal analyzer and load data
    s2 = cpe367_sig_analyzer(more_sig_list, fs)
    s2.load(fpath_sig_in)
    s2.print_desc()

    ########################
    # students: setup filters
    # filter freq response coefficients
    h = {
        # ROWS
        697: (1, -0.885, 0.930, 0.034),
        770: (1, -0.686, 0.94, 0.030),
        852: (1, -0.442, 0.925, 0.037),
        941: (1, -0.18, 0.910, 0.045),
        # COLS
        1209: (1, 0.625, 0.938, 0.089),
        1336: (1, 0.976, 0.939, 0.087),
        1477: (1, 1.252, 0.939, 0.115),
        1633: (1, 1.626, 0.940, 0.085)
    }

    N = 6
    C = 128

    # multiplies the coefficients by the constant to use integer multiplication for faster computation
    h = {freq: tuple(int(round(i * C)) for i in coeffs) for freq, coeffs in h.items()}  # scale to large int

    # fifos for filter outputs
    y = {key: fifo(N * 1) for key in h.keys()}

    symbols = {
        (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
        (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
        (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
        (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D'
    }

    valid_low_freqs = list(h.keys())[:4]
    valid_high_freqs = list(h.keys())[4:]

    # process input
    low_freq = 0
    high_freq = 0

    detections = []

    for n_curr in range(s2.get_len()):

        # read next input sample from the signal analyzer
        xin = s2.get('xin', n_curr)

        ########################
        # students: evaluate each filter and implement other processing blocks
        # pass through all filters
        for freq in y.keys():
            if True:
                update_val = (h[freq][0] * xin
                              - (h[freq][1]) * y[freq].get(0)
                              - (h[freq][2]) * y[freq].get(1)
                              )
                update_val /= C  # unscale
                y[freq].update(update_val)

        ########################
        # students: combine results from filtering stages
        #  and find (best guess of) symbol that is present at this sample time
        if n_curr % 2 == 0 and n_curr != 0:
            Y_DFT = []

            for freq in y.keys():
                magnitudes = Magnitude(DFT(y[freq].buff, N))  # Y(F) magnitudes
                Y_DFT.append((freq, max(magnitudes)))  # append freq and corresponding max magnitude

            # seperate highs and lows
            low_candidates = [(freq, mag) for freq, mag in Y_DFT if valid_low_freqs[-1] >= freq >= valid_low_freqs[0]]
            high_candidates = [(freq, mag) for freq, mag in Y_DFT if
                               valid_high_freqs[-1] >= freq >= valid_high_freqs[0]]

            # peaks are the 2 highest magnitudes
            low_freq = max(low_candidates, key=lambda i: i[1])[0]  # low freq with the highest magnitude
            high_freq = max(high_candidates, key=lambda i: i[1])[0]  # high freq w the highest magnitude

        # closest matching freqs
        found_low = find_closest_freq(low_freq, valid_low_freqs)
        found_high = find_closest_freq(high_freq, valid_high_freqs)

        # freq found
        symbol_val_det = symbols.get((found_low, found_high)) if (low_freq != 0 and high_freq != 0) else 0
        symbol_val_det = int(symbol_val_det) if isdigit(symbol_val_det) else 0
        detections.append(symbol_val_det)

        # save intermediate signals as needed, for plotting
        #  add signals, as desired!
        s2.set('sig_1', n_curr, xin)
        s2.set('sig_2', n_curr, 2 * xin)

    # smoothing function call
    smoothed_detections = majority_polling(detections, window_size=100)

    for n_curr, detection in enumerate(smoothed_detections):

        # save detected symbol
        s2.set('symbol_det', n_curr, detection)

        # get correct symbol (provided within the signal analyzer)
        symbol_val = s2.get('symbol_val', n_curr)

        # compare detected signal to correct signal
        symbol_val_err = 0
        if symbol_val != detection: symbol_val_err = 1

        # save error signal
        s2.set('error', n_curr, symbol_val_err)

    # display mean of error signal
    err_mean = s2.get_mean('error')
    print('mean error = ' + str(round(100 * err_mean, 1)) + '%')

    # define which signals should be plotted
    plot_sig_list = ['sig_1', 'sig_2', 'symbol_val', 'symbol_det', 'error']

    # plot results
    s2.plot(plot_sig_list)

    return True


# finds the closest frequency of the valid frequencies
def find_closest_freq(detected_freq: int, valid_freqs: list[int], ):
    closest_freq = min(valid_freqs, key=lambda f: abs(f - detected_freq))
    return closest_freq


# uses majority polling to smooth out the detected symbols and get rid of any outliers
def majority_polling(data, window_size):
    smoothed = data[:]  # copies data list
    half_window = window_size // 2

    # uses the most common symbols within a window to adjust outlying values causing error
    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window: i + half_window + 1]
        most_common = Counter(window).most_common(1)[0][0]  # Get most frequent value
        smoothed[i] = most_common
    smoothed = smoothed[10:] + smoothed[:10]

    return smoothed


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

    # assign file name
    # fpath_sig_in = 'dtmf_signals_slow.txt'
    fpath_sig_in = 'dtmf_signals_fast.txt'

    # let's do it!
    return process_wav(fpath_sig_in)


############################################
############################################
# call main function
if __name__ == '__main__':
    main()
    quit()
