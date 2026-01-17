import numpy as np
import torch
from scipy import signal
import pywt

class CSIProcessor:
    def __init__(self, target_length=500, n_subcarriers=30):
        self.target_length = target_length
        self.n_subcarriers = n_subcarriers
        
    def process(self, raw_csi):
        '''
        Process raw CSI data into model-ready format
        
        Args:
            raw_csi: numpy array of shape (time_steps, n_subcarriers, 2) 
                    where last dimension is [real, imaginary]
                    
        Returns:
            processed_csi: numpy array of shape (time_steps, n_subcarriers * 2)
        '''
        # Extract amplitude and phase
        amplitude = np.abs(raw_csi[:,:,0] + 1j * raw_csi[:,:,1])
        phase = np.angle(raw_csi[:,:,0] + 1j * raw_csi[:,:,1])
        
        # Stack features
        features = np.concatenate([amplitude, phase], axis=1)
        
        # Resize to target length
        if len(features) != self.target_length:
            features = self._resize_time_series(features)
        
        # Normalize
        features = self._normalize(features)
        
        return features
    
    def _resize_time_series(self, data):
        '''Resize time series to target length using interpolation'''
        from scipy.interpolate import interp1d
        
        old_indices = np.linspace(0, 1, len(data))
        new_indices = np.linspace(0, 1, self.target_length)
        
        resized = np.zeros((self.target_length, data.shape[1]))
        for i in range(data.shape[1]):
            f = interp1d(old_indices, data[:, i], kind='linear')
            resized[:, i] = f(new_indices)
        
        return resized
    
    def _normalize(self, data):
        '''Normalize data to zero mean, unit variance'''
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (data - mean) / std
    
    def denoise_wavelet(self, csi_data, wavelet='db4', level=3):
        '''Apply wavelet denoising to CSI data'''
        denoised = np.zeros_like(csi_data)
        
        for i in range(csi_data.shape[1]):  # For each subcarrier
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(csi_data[:, i], wavelet, level=level)
            
            # Threshold coefficients (universal threshold)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(csi_data[:, i])))
            
            # Apply soft thresholding
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            
            # Reconstruct signal
            denoised[:, i] = pywt.waverec(coeffs_thresh, wavelet)[:len(csi_data[:, i])]
        
        return denoised

# Example usage
if __name__ == '__main__':
    # Generate synthetic CSI data for testing
    time_steps = 1000
    subcarriers = 30
    
    # Complex CSI data (real + imaginary parts)
    synthetic_csi = np.random.randn(time_steps, subcarriers, 2)
    
    # Process the data
    processor = CSIProcessor()
    processed = processor.process(synthetic_csi)
    
    print(f'Input shape: {synthetic_csi.shape}')
    print(f'Output shape: {processed.shape}')
    print('Processing complete!')

