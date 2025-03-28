import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# Create Sound folder if it doesn't exist
os.makedirs('Sound', exist_ok=True)

class Filters:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sample_rate = None
        self.audio_data = None

    def load_audio(self):   
        try: 
            self.sample_rate, self.audio_data = wavfile.read(self.file_path)
            # Convert stereo to mono if necessary
            if len(self.audio_data.shape )> 1:
                self.audio_data = self.audio_data[:, 0]

            if self.audio_data.dtype == np.int16:
                self.audio_data = self.audio_data.astype(np.float32) / 32767.0
            elif self.audio_data.dtype == np.int32:
                self.audio_data = self.audio_data.astype(np.float32) / 2147483647.0

            return self.sample_rate, self.audio_data
        except FileNotFoundError:
            print(f"Error: Could not find '{self.file_path}' in the Sound folder")
            exit(1)

    def create_lowpass_fir(self, order, cutoff):
        nyquist = self.sample_rate / 2
        cutoff = cutoff / nyquist
        
        # Ensure order is odd for symmetric filter
        if order % 2 == 0:
            order += 1
        
        # Create time indices centered at zero
        n = np.arange(-(order-1)//2, (order-1)//2 + 1)
        
        # Create ideal lowpass filter (sinc function)
        h_ideal = 2 * cutoff * np.sinc(2 * cutoff * n)
        
        # Apply Hamming window for better frequency response
        window = np.hamming(order)
        h = h_ideal * window
        
        # Normalize to ensure unity gain at DC
        h = h / np.sum(h)
        
        return signal.lfilter(h, 1.0, self.audio_data)
        

    # Define Kalman filter function
    def apply_kalman_filter(self,process_variance=1e-3, measurement_variance=1e-2):
        """
        Apply a simple Kalman filter to the signal.
        
        Parameters:
        - signal_data: Input audio signal
        - process_variance: Process noise variance (Q)
        - measurement_variance: Measurement noise variance (R)
        
        Returns:
        - Filtered signal
        """
        # Initialize Kalman filter state
        x_est = self.audio_data[0]  # Initial state estimate
        p_est = 1.0             # Initial error estimate
        
        # Output array
        filtered_signal = np.zeros_like(self.audio_data)
        filtered_signal[0] = x_est
        
        # Apply Kalman filter
        for i in range(1, len(self.audio_data)):
            # Prediction step
            x_pred = x_est
            p_pred = p_est + process_variance
            
            # Update step (correction)
            k_gain = p_pred / (p_pred + measurement_variance)  # Kalman gain
            x_est = x_pred + k_gain * (self.audio_data[i] - x_pred)
            p_est = (1 - k_gain) * p_pred
            
            # Store the filtered value
            filtered_signal[i] = x_est
        
        return filtered_signal

    # Save each filtered audio file
    def save_audio(self, filename):
        audio_int = np.clip(self.audio_data * 32767, -32767, 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int)
        print(f"Saved audio to {filename}")

    # Function to compute and plot frequency response
    def plot_frequency_response(self, signals, titles, colors, filename, cutoff=None):
        plt.figure(figsize=(12, 10))
        n = len(self.audio_data)
        window = np.hanning(n)
        freq = np.fft.rfftfreq(n, d=1/self.sample_rate)
        nyquist = self.sample_rate / 2
        
        for i, (signal_data, title, color) in enumerate(zip(signals, titles, colors)):
            # Compute FFT
            fft_data = np.abs(np.fft.rfft(signal_data * window))
            # Normalize
            fft_data = fft_data / np.max(fft_data)
            # Convert to dB
            fft_db = 20 * np.log10(fft_data + 1e-10)
            
            plt.subplot(len(signals), 1, i+1)
            plt.fill_between(freq, fft_db, -100, alpha=0.7, color=color)
            plt.title(title)
            plt.ylabel('Amplitude (dB)')
            if i == len(signals) - 1:
                plt.xlabel('Frequency (Hz)')
            plt.grid(True, alpha=0.3)
            if cutoff and 'Low-Pass' in title:
                plt.axvline(cutoff, color='r', linestyle='--', alpha=0.7, 
                           label=f'Cutoff: {cutoff:.1f} Hz')
                plt.legend()
            plt.xlim(0, nyquist)
            plt.ylim(-60, 5)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Plot the theoretical FIR filter response
    def plot_filter_response(self, filter_order, cutoff, filename):
        plt.figure(figsize=(10, 6))
        nyquist = self.sample_rate / 2
        cutoff_norm = cutoff / nyquist
        
        # Create filter coefficients for visualization
        if filter_order % 2 == 0:
            filter_order += 1
        n = np.arange(-(filter_order-1)//2, (filter_order-1)//2 + 1)
        h_ideal = 2 * cutoff_norm * np.sinc(2 * cutoff_norm * n)
        window = np.hamming(filter_order)
        h = h_ideal * window
        h = h / np.sum(h)
        
        # Get frequency response
        w, h_freq = signal.freqz(h)
        freq = (w / np.pi) * nyquist
        magnitude = 20 * np.log10(abs(h_freq))  # Convert to dB

        # Use fill_between for the filter response
        plt.fill_between(freq, magnitude, -100, alpha=0.7, color='green')
        plt.title('Low-Pass FIR Filter Theoretical Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True, alpha=0.3)
        plt.axvline(cutoff, color='r', linestyle='--', alpha=0.7, label=f'Cutoff: {cutoff:.1f} Hz')
        plt.legend()
        plt.ylim(-80, 5)  # Limit y-axis for better visualization
        plt.xlim(0, nyquist)  # Limit x-axis to Nyquist frequency
        plt.savefig(filename)
        plt.close()

    # Generate spectrograms for each signal
    def plot_spectrograms(self, signals, titles, filename):
        plt.figure(figsize=(12, 12))
        
        for i, (signal_data, title) in enumerate(zip(signals, titles)):
            plt.subplot(len(signals), 1, i+1)
            f, t, Sxx = signal.spectrogram(signal_data, self.sample_rate)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            plt.title(f'{title} Spectrogram')
            plt.ylabel('Frequency (Hz)')
            if i == len(signals) - 1:
                plt.xlabel('Time (s)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    # Apply sequential filtering (one filter after another)
    def apply_sequential_filtering(self, filter_order=101, cutoff_freq=1000):
        """
        Apply FIR filter followed by Kalman filter in sequence.
        
        Parameters:
        - filter_order: Order of the FIR filter
        - cutoff_freq: Cutoff frequency for the FIR filter in Hz
        
        Returns:
        - Sequentially filtered signal
        """
        # First apply the FIR low-pass filter
        fir_filtered = self.create_lowpass_fir(filter_order, cutoff_freq)
        
        # Store the FIR filtered data temporarily
        original_data = self.audio_data
        self.audio_data = fir_filtered
        
        # Then apply the Kalman filter to the FIR filtered data
        sequential_filtered = self.apply_kalman_filter()
        
        # Restore the original audio data
        self.audio_data = original_data
        
        return sequential_filtered


# Main execution
if __name__ == "__main__":

    # pink noise
    # Initialize the filter class with the audio file
    filter_processor = Filters('Sound/pink_noise_30sec.wav')
    
    # Load the audio
    sample_rate, original_audio = filter_processor.load_audio()
    print(f"Loaded audio file: {sample_rate} Hz, {len(original_audio)} samples")
    
    # Apply FIR low-pass filter
    cutoff_freq = 1000  # Hz
    filter_order = 101
    fir_filtered_audio = filter_processor.create_lowpass_fir(filter_order, cutoff_freq)
    
    # Apply Kalman filter
    kalman_filtered_audio = filter_processor.apply_kalman_filter()
    
    # Apply sequential filtering (FIR followed by Kalman)
    sequential_filtered_audio = filter_processor.apply_sequential_filtering(filter_order, cutoff_freq)
    
    # Save the original audio (in case it was normalized)
    filter_processor.audio_data = original_audio
    filter_processor.save_audio('Sound/Pink_Noise/pink_noise_original.wav')
    
    # Save the FIR filtered audio
    filter_processor.audio_data = fir_filtered_audio
    filter_processor.save_audio('Sound/Pink_Noise/pink_noise_lowpass.wav')
    
    # Save the Kalman filtered audio
    filter_processor.audio_data = kalman_filtered_audio
    filter_processor.save_audio('Sound/Pink_Noise/pink_noise_kalman.wav')
    
    # Save the sequential filtered audio
    filter_processor.audio_data = sequential_filtered_audio
    filter_processor.save_audio('Sound/Pink_Noise/pink_noise_sequential.wav')
    
    # Plot frequency responses
    filter_processor.plot_frequency_response(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original Signal', 'Low-Pass FIR Filtered Signal', 'Kalman Filtered Signal', 'Sequential Filtered Signal'],
        ['blue', 'green', 'purple', 'red'],
        'Sound/Pink_Noise/all_filters_frequency_response.png',
        cutoff_freq
    )
    
    # Plot theoretical filter response
    filter_processor.plot_filter_response(
        filter_order, 
        cutoff_freq, 
        'Sound/Pink_Noise/lowpass_filter_response.png'
    )
    
    # Plot spectrograms
    filter_processor.plot_spectrograms(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original', 'Low-Pass FIR Filtered', 'Kalman Filtered', 'Sequential Filtered'],
        'Sound/Pink_Noise/all_filters_spectrogram.png'
    )

    #monologue
    # Initialize the filter class with the audio file
    filter_processor = Filters('Sound/monologue.wav')
    
    # Load the audio
    sample_rate, original_audio = filter_processor.load_audio()
    print(f"Loaded audio file: {sample_rate} Hz, {len(original_audio)} samples")
    
    # Apply FIR low-pass filter
    cutoff_freq = 1000  # Hz
    filter_order = 101
    fir_filtered_audio = filter_processor.create_lowpass_fir(filter_order, cutoff_freq)
    
    # Apply Kalman filter
    kalman_filtered_audio = filter_processor.apply_kalman_filter()
    
    # Apply sequential filtering (FIR followed by Kalman)
    sequential_filtered_audio = filter_processor.apply_sequential_filtering(filter_order, cutoff_freq)

    # Save the original audio (in case it was normalized)
    filter_processor.audio_data = original_audio
    filter_processor.save_audio('Sound/Monologue/monologue_original.wav')
    
    # Save the FIR filtered audio
    filter_processor.audio_data = fir_filtered_audio
    filter_processor.save_audio('Sound/Monologue/monologue_lowpass.wav')
    
    # Save the Kalman filtered audio
    filter_processor.audio_data = kalman_filtered_audio
    filter_processor.save_audio('Sound/Monologue/monologue_kalman.wav')
    
    # Save the sequential filtered audio
    filter_processor.audio_data = sequential_filtered_audio
    filter_processor.save_audio('Sound/Monologue/monologue_sequential.wav')

    # Plot frequency responses
    filter_processor.plot_frequency_response(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original Signal', 'Low-Pass FIR Filtered Signal', 'Kalman Filtered Signal', 'Sequential Filtered Signal'],
        ['blue', 'green', 'purple', 'red'],
        'Sound/Monologue/all_filters_frequency_response.png',
        cutoff_freq
    )

    # Plot theoretical filter response
    filter_processor.plot_filter_response(
        filter_order, 
        cutoff_freq, 
        'Sound/Monologue/lowpass_filter_response.png'
    )   
    
    # Plot spectrograms
    filter_processor.plot_spectrograms(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original', 'Low-Pass FIR Filtered', 'Kalman Filtered', 'Sequential Filtered'],
        'Sound/Monologue/all_filters_spectrogram.png'
    )

    # instrumental
    # Initialize the filter class with the audio file
    filter_processor = Filters('Sound/instrumental.wav')
    
    # Load the audio
    sample_rate, original_audio = filter_processor.load_audio()
    print(f"Loaded audio file: {sample_rate} Hz, {len(original_audio)} samples")    
    
    # Apply FIR low-pass filter
    cutoff_freq = 1000  # Hz
    filter_order = 101
    fir_filtered_audio = filter_processor.create_lowpass_fir(filter_order, cutoff_freq) 
    
    # Apply Kalman filter
    kalman_filtered_audio = filter_processor.apply_kalman_filter()
    
    # Apply sequential filtering (FIR followed by Kalman)
    sequential_filtered_audio = filter_processor.apply_sequential_filtering(filter_order, cutoff_freq)

    # Save the original audio (in case it was normalized)
    filter_processor.audio_data = original_audio
    filter_processor.save_audio('Sound/Instrumental/instrumental_original.wav') 
    
    # Save the FIR filtered audio
    filter_processor.audio_data = fir_filtered_audio
    filter_processor.save_audio('Sound/Instrumental/instrumental_lowpass.wav')
    
    # Save the Kalman filtered audio
    filter_processor.audio_data = kalman_filtered_audio
    filter_processor.save_audio('Sound/Instrumental/instrumental_kalman.wav')   
    
    # Save the sequential filtered audio
    filter_processor.audio_data = sequential_filtered_audio
    filter_processor.save_audio('Sound/Instrumental/instrumental_sequential.wav')

    # Plot frequency responses
    filter_processor.plot_frequency_response(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original Signal', 'Low-Pass FIR Filtered Signal', 'Kalman Filtered Signal', 'Sequential Filtered Signal'],
        ['blue', 'green', 'purple', 'red'],    
        'Sound/Instrumental/all_filters_frequency_response.png',
        cutoff_freq
    )

    # Plot theoretical filter response
    filter_processor.plot_filter_response(
        filter_order, 
        cutoff_freq, 
        'Sound/Instrumental/lowpass_filter_response.png'
    )

    # Plot spectrograms
    filter_processor.plot_spectrograms(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original', 'Low-Pass FIR Filtered', 'Kalman Filtered', 'Sequential Filtered'],   
        'Sound/Instrumental/all_filters_spectrogram.png'
    ) 

    # Interesting
    # Initialize the filter class with the audio file
    filter_processor = Filters('Sound/interesting.wav')
    
    # Load the audio
    sample_rate, original_audio = filter_processor.load_audio()
    print(f"Loaded audio file: {sample_rate} Hz, {len(original_audio)} samples")    
    
    # Apply FIR low-pass filter
    cutoff_freq = 1000  # Hz
    filter_order = 101
    fir_filtered_audio = filter_processor.create_lowpass_fir(filter_order, cutoff_freq)
    
    # Apply Kalman filter
    kalman_filtered_audio = filter_processor.apply_kalman_filter()
    
    # Apply sequential filtering (FIR followed by Kalman)
    sequential_filtered_audio = filter_processor.apply_sequential_filtering(filter_order, cutoff_freq)

    # Save the original audio (in case it was normalized)
    filter_processor.audio_data = original_audio
    filter_processor.save_audio('Sound/Interesting/interesting_original.wav')
    
    # Save the FIR filtered audio
    filter_processor.audio_data = fir_filtered_audio
    filter_processor.save_audio('Sound/Interesting/interesting_lowpass.wav')
    
    # Save the Kalman filtered audio
    filter_processor.audio_data = kalman_filtered_audio
    filter_processor.save_audio('Sound/Interesting/interesting_kalman.wav')     
    
    # Save the sequential filtered audio
    filter_processor.audio_data = sequential_filtered_audio
    filter_processor.save_audio('Sound/Interesting/interesting_sequential.wav')

    # Plot frequency responses
    filter_processor.plot_frequency_response(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original Signal', 'Low-Pass FIR Filtered Signal', 'Kalman Filtered Signal', 'Sequential Filtered Signal'],
        ['blue', 'green', 'purple', 'red'],    
        'Sound/Interesting/all_filters_frequency_response.png', 
        cutoff_freq
    )

    # Plot theoretical filter response
    filter_processor.plot_filter_response(
        filter_order, 
        cutoff_freq, 
        'Sound/Interesting/lowpass_filter_response.png'
    )

    # Plot spectrograms
    filter_processor.plot_spectrograms(
        [original_audio, fir_filtered_audio, kalman_filtered_audio, sequential_filtered_audio],
        ['Original', 'Low-Pass FIR Filtered', 'Kalman Filtered', 'Sequential Filtered'],   
        'Sound/Interesting/all_filters_spectrogram.png'
    )

    print("Analysis complete. Generated frequency response and spectral analysis graphs in the Sound folder.")
