import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def extract_features(file_path):
    """
    Extracts acoustic features from an audio file using Librosa.
    Returns a numpy array of features.
    """
    try:
        # Load audio (mono, 22050 Hz)
        y, sr = librosa.load(file_path, sr=22050)
        return extract_features_from_array(y, sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features_from_array(y, sr):
    """
    Extracts acoustic features from a loaded librosa audio array.
    """
    try:
        
        # Pad signal if it's too short for standard n_fft=2048 windows
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')
            
        # 1. MFCCs (Increase to 40 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # 2. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 3. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(spectral_rolloff)
        
        # 5. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # 7. Chroma STFT (12 bins)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 8. Mel Spectrogram (128 bins)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        
        # 9. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(spectral_bandwidth)
        
        # 10. Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(spectral_flatness)
        
        # 11. Tonnetz (6 tonal centroid features)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = np.mean(tonnetz.T, axis=0)
        except:
            tonnetz_mean = np.zeros(6)
            
        # Combine all features into a massive high-dimensional vector
        features = np.hstack([
            mfccs_mean,
            zcr_mean,
            centroid_mean,
            rolloff_mean,
            contrast_mean,
            rms_mean,
            chroma_mean,
            mel_mean,
            bandwidth_mean,
            flatness_mean,
            tonnetz_mean
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features from array: {e}")
        return None
