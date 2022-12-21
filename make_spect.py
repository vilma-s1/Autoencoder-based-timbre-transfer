import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
#mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
mel_basis = mel(sr = 22050, n_fft= 1024, fmin=30, fmax=13000, n_mels=80).T   # MODIFIED BY ME
print(mel_basis.shape)
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
#rootDir = '/content/drive/MyDrive/timbre_transfer/urmp_train_data'  # training data
rootDir = '/content/drive/MyDrive/timbre_transfer/urmp_test_data'


# spectrogram directory
#targetDir = '/content/drive/MyDrive/timbre_transfer/spmel_train'  # training spectrogram directory
targetDir = '/content/drive/MyDrive/timbre_transfer/spmel_test'


dirName, subdirList, _ = next(os.walk(rootDir))
print(subdirList)
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):

    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    #prng = RandomState(int(subdir[1:]))
    prng = RandomState(int(subdir[4:]))  # MODIFIED BY ME
    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Add a little random noise for model robustness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        print(D.shape)
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    
        # save spect    
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        
