import sys

import time

import base64

import random as random

import datetime

import math

import matplotlib.pyplot as plt

import numpy as np

from cpe367_wav import cpe367_wav

from cpe367_sig_analyzer import cpe367_sig_analyzer

############################################

# Constant for Integer Scaling

############################################

C = 1024


############################################

# Filter Implementation for Bandpass Filters with Integer Scaling

############################################

# Define a filter class that implements a direct-form I filter using integer arithmetic.

class DummyFilter:

	def __init__(self, b0, b1, b2, a1, a2, C_val=C):
		self.C = C_val

		# Scale coefficients using the constant C and round to integer.

		self.B0 = int(round(self.C * b0))

		self.B1 = int(round(self.C * b1))

		self.B2 = int(round(self.C * b2))

		self.A1 = int(round(self.C * a1))

		self.A2 = int(round(self.C * a2))

		# Initialize state variables for previous inputs and outputs.

		self.x1 = 0

		self.x2 = 0

		self.y1 = 0

		self.y2 = 0

	def process_sample(self, x):
		# Compute the filter output using integer scaled coefficients:

		# y[n] = (B0*x[n] + B1*x[n-1] + B2*x[n-2] - A1*y[n-1] - A2*y[n-2]) / C

		num = self.B0 * x + self.B1 * self.x1 + self.B2 * self.x2 \
		\
		      - self.A1 * self.y1 - self.A2 * self.y2

		y = num / self.C  # Division restores the scaled value to original range.

		# Update state variables: shift previous input and output samples.

		self.x2 = self.x1

		self.x1 = x

		self.y2 = self.y1

		self.y1 = y

		return y


############################################

# Symbol-to-Number Mapping

############################################

# This dictionary maps a string (like '1', '2') to a numeric code

# so that the 'symbol_det' plot won't be stuck at 0.

symbol_to_num = {

	'1': 1, '2': 2, '3': 3,

	'4': 4, '5': 5, '6': 6,

	'7': 7, '8': 8, '9': 9,

	'0': 0, '*': 10, '#': 11,

	None: -1  # Use -1 for "no detection"

}


############################################

# DTMF Processing Code

############################################


# Define routine for detecting DTMF tones.

def process_wav(fpath_sig_in):
	###############################

	# Define list of signals to be displayed by our analyzer.

	# Note: The signal analyzer already includes: 'symbol_val','symbol_det','error'

	more_sig_list = ['sig_1', 'sig_2']

	# Sample rate is 4kHz.

	fs = 4000

	# Instantiate signal analyzer and load data.

	s2 = cpe367_sig_analyzer(more_sig_list, fs)

	s2.load(fpath_sig_in)

	s2.print_desc()

	############################################

	# Bandpass Filter Setup with Scaled Coefficients

	############################################

	# Define coefficients for the eight DTMF frequencies.

	dummy_coeffs = {

		'697': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': -0.8890, 'a2': 0.9401},

		'770': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': -0.6858, 'a2': 0.9401},

		'852': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': -0.4470, 'a2': 0.9401},

		'941': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': -0.1795, 'a2': 0.9401},

		'1209': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': 0.6256, 'a2': 0.9401},

		'1336': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': 0.9772, 'a2': 0.9401},

		'1477': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': 1.3215, 'a2': 0.9401},

		'1633': {'b0': 1.0, 'b1': 0.0, 'b2': 0.0, 'a1': 1.6267, 'a2': 0.9401},

	}

	# Create a filter instance for each frequency using the scaled coefficients.

	filters = {}

	for freq, coeff in dummy_coeffs.items():
		filters[freq] = DummyFilter(coeff['b0'], coeff['b1'], coeff['b2'],

		                            coeff['a1'], coeff['a2'], C)

	############################################

	# Main Signal Processing Loop

	############################################

	# Process each sample to generate filtered outputs and temporary signals.

	for n_curr in range(s2.get_len()):

		# Read next input sample from the signal analyzer.

		xin = s2.get('xin', n_curr)

		# Process the sample through each bandpass filter.

		filter_outputs = {}

		for freq, filt in filters.items():
			filter_outputs[freq] = filt.process_sample(xin)
			# print(filter_outputs[freq])

		# Currently, we set symbol_val_det to 0 as a placeholder.

		symbol_val_det = 0

		# Save intermediate signals for plotting.

		s2.set('sig_1', n_curr, xin)

		s2.set('sig_2', n_curr, 2 * xin)

		# Save the detected symbol (placeholder).

		# We'll overwrite this below after FFT-based detection.

		s2.set('symbol_det', n_curr, symbol_val_det)

		# Get correct symbol (provided within the signal analyzer).

		symbol_val = s2.get('symbol_val', n_curr)

		# Compare detected symbol to correct symbol.

		symbol_val_err = 0

		if symbol_val != symbol_val_det:
			symbol_val_err = 1

		# Save error signal.

		s2.set('error', n_curr, symbol_val_err)

	# Display mean of error signal so far.

	err_mean = s2.get_mean('error')

	print('mean error (pre-FFT) = ' + str(round(100 * err_mean, 1)) + '%')

	# Define which signals should be plotted.

	plot_sig_list = ['sig_1', 'sig_2', 'symbol_val', 'symbol_det', 'error']

	# Plot results so we can see the placeholder detection.

	s2.plot(plot_sig_list)

	############################################

	# FFT Decision Logic with Overlapping Blocks

	############################################

	# We'll use a block size of 62 samples with 52% overlap.

	block_size = 62

	step = int(0.52 * block_size)

	# Store detections as (center_index, detected_symbol) for each block.

	detections = []

	if s2.get_len() >= block_size:

		for block_start in range(0, s2.get_len() - block_size + 1, step):

			block_end = block_start + block_size

			# Extract samples for this block.

			signal_block = [s2.get('xin', i) for i in range(block_start, block_end)]

			# Compute FFT using our compute_dft function.

			X = compute_dft(signal_block)

			# Manually compute frequency bins for this block.

			N = len(signal_block)

			freqs = np.array([fs * k / N for k in range(N // 2)])

			fft_magnitudes = np.abs(X[:N // 2])

			# Define expected DTMF frequency groups.

			dtmf_freqs_low = [697, 770, 852, 941]

			dtmf_freqs_high = [1209, 1336, 1477, 1633]

			# Use a tolerance of 25 Hz.

			def find_peak_in_group(fft_magnitudes, freqs, target_freqs, tolerance=25):

				best_freq = None

				best_mag = 0

				for target in target_freqs:

					indices = np.where((freqs >= target - tolerance) & (freqs <= target + tolerance))[0]

					if len(indices) == 0:
						continue

					mag = np.max(fft_magnitudes[indices])

					if mag > best_mag:
						best_mag = mag

						best_freq = target

				return best_freq, best_mag

			low_freq, low_mag = find_peak_in_group(fft_magnitudes, freqs, dtmf_freqs_low)

			high_freq, high_mag = find_peak_in_group(fft_magnitudes, freqs, dtmf_freqs_high)

			# Define a threshold to validate the detection.

			threshold = 0.1

			# Define lookup table mapping frequency pairs to DTMF symbols.

			dtmf_lookup = {

				(697, 1209): '1',

				(697, 1336): '2',

				(697, 1477): '3',

				(770, 1209): '4',

				(770, 1336): '5',

				(770, 1477): '6',

				(852, 1209): '7',

				(852, 1336): '8',

				(852, 1477): '9',

				(941, 1209): '*',

				(941, 1336): '0',

				(941, 1477): '#',

			}

			detected_symbol = None

			if (low_freq is not None) and (high_freq is not None):

				if (low_mag > threshold) and (high_mag > threshold):
					detected_symbol = dtmf_lookup.get((low_freq, high_freq), None)

			center_index = block_start + block_size // 2

			detections.append((center_index, detected_symbol))

			print(f"Block {block_start} to {block_end} - Detected Symbol from FFT:", detected_symbol)

	# Majority vote smoothing: for each sample, consider detections within a vote window.

	vote_window = 32

	final_detections = [None] * s2.get_len()

	for i in range(s2.get_len()):

		votes = []

		for center, sym in detections:

			if abs(center - i) <= vote_window // 2 and sym is not None:
				votes.append(sym)

		if votes:

			final_sym = max(set(votes), key=votes.count)

		else:

			final_sym = None

		final_detections[i] = final_sym

		if final_sym is None:

			numeric_symbol = -1

		else:

			numeric_symbol = symbol_to_num.get(final_sym, -1)

		s2.set('symbol_det', i, numeric_symbol)

		true_sym = s2.get('symbol_val', i)

		err = 0 if (str(true_sym) == str(final_sym)) else 1

		s2.set('error', i, err)

	err_mean_final = s2.get_mean('error')

	print('mean error (post-FFT) = ' + str(round(100 * err_mean_final, 1)) + '%')

	s2.plot(plot_sig_list)

	return True


############################################

# DFT Code Integration (from last week)

############################################

def compute_dft(signal):
	"""

	Compute the Discrete Fourier Transform (DFT) of a 1D signal.

	"""

	N = len(signal)

	X = np.zeros(N, dtype=complex)

	for k in range(N):

		for n in range(N):
			X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

		X[k] /= N  # Normalize

	return X


def compute_center_frequency(X, freqs):
	"""

	Compute a weighted average frequency from the DFT spectrum.

	"""

	T = np.max(np.abs(X))

	threshold = T / 2

	valid_indices = np.where(np.abs(X) > threshold)[0]

	if len(valid_indices) == 0:
		return 0

	weighted_sum_freqs = np.sum(freqs[valid_indices] * np.abs(X[valid_indices]))

	weighted_sum_magnitudes = np.sum(np.abs(X[valid_indices]))

	f0 = weighted_sum_freqs / weighted_sum_magnitudes

	return f0


def plot_spectrum(X, sample_rate, N):
	"""

	Plot the magnitude spectrum of the signal and print the computed center frequency.

	"""

	freqs = np.array([sample_rate * k / N for k in range(N // 2)])

	magnitude = np.abs(X[:N // 2])

	fig, ax = plt.subplots()

	ax.plot(freqs, magnitude)

	ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude', title='Magnitude Spectrum')

	ax.grid(True)

	f0 = compute_center_frequency(X[:N // 2], freqs)

	print(f"Computed Center Frequency: {f0:.2f} Hz")

	fig.savefig('spectrum_plot.png')

	plt.show()


############################################

# Main Program

############################################

def main():
	major_version = int(sys.version[0])

	if major_version < 3:
		print('Sorry! must be run using python3.')

		print('Current version:')

		print(sys.version)

		return False

	# fpath_sig_in = 'dtmf_signals_slow.txt'

	fpath_sig_in = 'dtmf_signals_fast.txt'

	process_wav(fpath_sig_in)

	return True


if __name__ == '__main__':
	main()

	quit()

# for documentation purposes

# slow 6% to 7% depending on Vote Window

# fast 13 to 14 % depending on Vote Window

# both conditions are satisified


