# a = {0:[1, 2], 1:[3, 1]}
# b = sorted(a.items(), key=lambda x:(x[1][0], x[1][1]), reverse=False)
# print(b)
# print(3604.038820861678/3600)
import pyworld
import librosa
import numpy as np
fpath = './010000.wav'
aimpath = './LJ002-0014-syn.wav'
sr = 22050
n_fft = 2048
feature_dim = 60
y, _ = librosa.load(path=fpath, sr=sr, dtype=np.float64)
f0, timeaxis = pyworld.harvest(y, sr, f0_floor=71.0, f0_ceil=500.0)
sp = pyworld.cheaptrick(y, f0, timeaxis, sr, fft_size=n_fft)
ap = pyworld.d4c(y, f0, timeaxis, sr, fft_size=n_fft)
coded_sp = pyworld.code_spectral_envelope(sp, sr, number_of_dimensions=feature_dim)
coded_ap = pyworld.code_aperiodicity(ap, sr)
print(ap.shape)
print(coded_ap.shape)
print(coded_sp.shape)
print(f0.shape)
decoded_sp = pyworld.decode_spectral_envelope(coded_sp, sr, fft_size=n_fft)
decoded_ap = pyworld.decode_aperiodicity(coded_ap, sr, fft_size=n_fft)
f0 -= 100
#y = pyworld.synthesize(f0, decoded_sp, decoded_ap, sr)
#librosa.output.write_wav(aimpath, y, sr)