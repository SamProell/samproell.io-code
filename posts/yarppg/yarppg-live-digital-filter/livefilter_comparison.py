import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

from digitalfilter import LiveLFilter, LiveSosFilter

np.random.seed(42)  # for reproducibility
# create time steps and corresponding sine wave with Gaussian noise
fs = 30  # sampling rate, Hz
ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds

ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
yraw = ys + yerr

# define lowpass filter with 2.5 Hz cutoff frequency
b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
y_scipy_lfilter = scipy.signal.lfilter(b, a, yraw)

live_lfilter = LiveLFilter(b, a)
# simulate live filter - passing values one by one
y_live_lfilter = [live_lfilter(y) for y in yraw]

print(f"lfilter error: {mae(y_scipy_lfilter, y_live_lfilter):.5g}")

plt.figure(figsize=[6.4, 2.4])
plt.plot(ts, yraw, label="Noisy signal")
plt.plot(ts, y_scipy_lfilter, alpha=1, lw=2, label="SciPy lfilter")
plt.plot(ts, y_live_lfilter, lw=4, alpha=1, ls="dashed", label="LiveLFilter")

plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=3,
           fontsize="smaller")
plt.xlabel("Time / s")
plt.ylabel("Amplitude")
plt.tight_layout()

# define lowpass filter with 2.5 Hz cutoff frequency
sos = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low",
                             ftype="butter", output="sos")
y_scipy_sosfilt = scipy.signal.sosfilt(sos, yraw)

live_sosfilter = LiveSosFilter(sos)
# simulate live filter - passing values one by one
y_live_sosfilt = [live_sosfilter(y) for y in yraw]

print(f"sosfilter error: {mae(y_scipy_sosfilt, y_live_sosfilt):.5g}")

plt.figure(figsize=[6.4, 2.4])
plt.plot(ts, yraw, label="Noisy signal")
plt.plot(ts, y_scipy_sosfilt, alpha=1, lw=2, label="SciPy sosfilt")
plt.plot(ts, y_live_sosfilt, lw=4, alpha=1, ls="dashed",
         label="LiveSosFilter")

plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=3,
           fontsize="smaller")
plt.xlabel("Time / s")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
