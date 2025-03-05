#!/usr/bin/python

import sys
import time

import base64
import random as random

import datetime
import time
import math

import matplotlib.pyplot as plt
import numpy as np

from DFT import DFT, Magnitude
from cpe367_wav import cpe367_wav
from cpe367_sig_analyzer import cpe367_sig_analyzer
from my_fifo import my_fifo as fifo


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
	#H[F] = (a0,b1,b2)
	h = {
		#COLS
		1209: (1, 0.591, 0.822),
		1336: (1, 0.906, 0.827),
		1447: (1, 1.214, 0.77),
		1633: (1, 1.532, 0.83),
		#ROWS
		697: (1, -0.877, 0.932),
		770: (1, -0.686, 0.94),
		852: (1, -0.442, 0.925),
		941: (1, -0.18, 0.910)
	}

	cnt = 0
	N = 10
	y = {
		1209 : fifo(N),
		1336 : fifo(N),
		1447 : fifo(N),
		1633 : fifo(N),
		697 : fifo(N),
		770 : fifo(N),
		852 : fifo(N),
		941 : fifo(N),
	}

	# process input
	xin = 0
	for n_curr in range(s2.get_len()):

		# read next input sample from the signal analyzer
		xin = s2.get('xin', n_curr)

		########################
		# students: evaluate each filter and implement other processing blocks
		for freq in y.keys():
			y[freq].update(h[freq][0] * xin + h[freq][1] * y[freq].get(1) + h[freq][2] * y[freq].get(2))
		if cnt % N == 0 and cnt != 0:
			Y_DFT = []

			for freq in y.keys():
				y_samples = {freq: y[freq].buff} #y[n]
				magnitudes = Magnitude(DFT(y_samples[freq], N)) #Y(F) magnitudes
				Y_DFT.append((freq, max(magnitudes))) #append freq and corresponding max magnitude
			peaks = [0, 0]
			Y_DFT.sort(key=lambda x: x[1], reverse=True) #sort by magnitudes
			peaks[0], peaks[1] = Y_DFT[0][0], Y_DFT[1][0] #peaks are the 2 highest magnitudes








		########################
		# students: combine results from filtering stages
		#  and find (best guess of) symbol that is present at this sample time


		# pass through all filters

		symbol_val_det = 0

		# save intermediate signals as needed, for plotting
		#  add signals, as desired!
		s2.set('sig_1', n_curr, xin)
		s2.set('sig_2', n_curr, 2 * xin)

		# save detected symbol
		s2.set('symbol_det', n_curr, symbol_val_det)

		# get correct symbol (provided within the signal analyzer)
		symbol_val = s2.get('symbol_val', n_curr)

		# compare detected signal to correct signal
		symbol_val_err = 0
		if symbol_val != symbol_val_det: symbol_val_err = 1

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
	fpath_sig_in = 'dtmf_signals_slow.txt'
	# fpath_sig_in = 'dtmf_signals_fast.txt'

	# let's do it!
	return process_wav(fpath_sig_in)


############################################
############################################
# call main function
if __name__ == '__main__':
	main()
	quit()
