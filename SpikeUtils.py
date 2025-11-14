import datetime
import math
import os
import threading
from scipy import signal
from scipy.signal import find_peaks
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import h5py
import torch


class SpikeDetection:
    def __init__(self, params, output_directory):
        self.filter_type = params.get('filter')
        self.filter_low = params.get('filter_low')
        self.filter_high = params.get('filter_high')
        self.filter_order = params.get('filter_order')
        self.threshold = params.get('threshold')
        self.detect_interval = params.get('detect_interval')
        self.num_channels = params.get('num_channels')
        self.sample_rate = params.get('sample_rate')
        self.is_electrode_correlation = params.get('is_electrode_correlation')
        self.spike_window = int(self.sample_rate * 0.003)
        self.chunk = self.spike_window // 3
        self.filtered_data = 0
        self.output_directory = output_directory
        if params.get('amplifier_filtered_filename') is None:
            self.amplifier_filtered_filepath = None
        else:
            self.amplifier_filtered_filepath = os.path.join(self.output_directory, params.get('amplifier_filtered_filename'))
        self.spikeInfo_filepath = os.path.join(self.output_directory, params.get('spikeInfo_filename'))
        self.num_chunks = params.get('num_chunks')
        self.adc_to_uV = params.get('adc_to_uV')  
        self.define_threshold = params.get('define_threshold')
        self.pos_neg_detect = params.get('pos_neg_detect')
        self.max_workers_detect = params.get('max_workers_detect')
        self.max_workers_preprocess = params.get('max_workers_preprocess')
        self.file_lock = threading.Lock()

    def run_preprocess(self, data):
        if self.amplifier_filtered_filepath is not None:
            if os.path.exists(self.amplifier_filtered_filepath):
                os.remove(self.amplifier_filtered_filepath)
                print(f"The old file {self.amplifier_filtered_filepath} is delete.")
            else:
                print(f"{self.amplifier_filtered_filepath} is not exits, will create this file, this will make sorting slow.")
            chunk_len = int(data.shape[1] / self.num_chunks)
            start = datetime.datetime.now()
            if self.filter_type == 'bandpass':
                self.bandpass_filter(data, chunk_len, num_chunks=self.num_chunks)
            end = datetime.datetime.now()
            preprocess_time = (end - start).total_seconds()
            print(f"Filter data cost {preprocess_time} seconds. Filtered data is saved in {self.amplifier_filtered_filepath}.")
            return preprocess_time
        else:
            print(f"This is fast mode, will not save preprocessed data.")
            self.filtered_data = np.zeros_like(data, dtype=np.float32)
            chunk_len = int(data.shape[1] / self.num_chunks)
            start = datetime.datetime.now()
            if self.filter_type == 'bandpass':
                self.bandpass_filter(data, chunk_len, num_chunks=self.num_chunks)
            end = datetime.datetime.now()
            preprocess_time = (end - start).total_seconds()
            print(f"Filter data cost {preprocess_time} seconds.")
            return preprocess_time

    def run_detection(self):
        if self.amplifier_filtered_filepath is not None:
            start = datetime.datetime.now()
            with h5py.File(self.amplifier_filtered_filepath, 'r') as ff:
                data = ff['preprocess_data']
                with ThreadPoolExecutor(max_workers=self.max_workers_detect) as executor:
                    futures = []
                    for c in range(self.num_channels):
                        futures.append(
                            executor.submit(self.process_channel, data, c)
                        )
                    results = [future.result() for future in as_completed(futures)]
                spike_times = np.concatenate([res[0] for res in results]) if results else np.array([])
                spike_electrodes = np.concatenate([res[1] for res in results]) if results else np.array([])
                non_empty_waveforms = [res[2] for res in results if res[2].size > 0]
                spike_waveforms = np.vstack(non_empty_waveforms) if non_empty_waveforms else np.empty((0, self.spike_window))
                ff.close()
            end = datetime.datetime.now()
            detection_time = (end - start).total_seconds()
            print(f"Detect spikes cost {detection_time} seconds, spike times and corresponding electrodes saved in {self.spikeInfo_filepath}.")
            return detection_time, spike_times, spike_electrodes, spike_waveforms
        else:
            start = datetime.datetime.now()
            with ThreadPoolExecutor(max_workers=self.max_workers_detect) as executor:
                futures = []
                for c in range(self.num_channels):
                    futures.append(
                        executor.submit(self.process_channel, self.filtered_data, c)
                    )
                results = [future.result() for future in as_completed(futures)]
            spike_times = np.concatenate([res[0] for res in results]) if results else np.array([])
            spike_electrodes = np.concatenate([res[1] for res in results]) if results else np.array([])
            non_empty_waveforms = [res[2] for res in results if res[2].size > 0]
            spike_waveforms = np.vstack(non_empty_waveforms) if non_empty_waveforms else np.empty((0, self.spike_window))
            end = datetime.datetime.now()
            detection_time = (end - start).total_seconds()
            print(f"Detect spikes cost {detection_time} seconds.")
            return detection_time, spike_times, spike_electrodes, spike_waveforms

    def process_channel(self, output_data, c):
        channel_data = output_data[c, :]
        if self.define_threshold is not None:
            if self.define_threshold < 0:
                peak_indices = self.detect_spikes(channel_data, self.define_threshold, 0)
            else:
                peak_indices = self.detect_spikes(channel_data, 0, self.define_threshold)
        else:
            threshold = self.threshold * np.median(np.abs(channel_data)) / 0.6745
            if self.pos_neg_detect > 0:
                peak_indices = self.detect_spikes(channel_data, 0, threshold)
            else:
                peak_indices = self.detect_spikes(channel_data, threshold * -1, 0)
 
        if len(peak_indices) > 0:
            spike_times = np.array(peak_indices, dtype=np.int32)
            electrodes = np.full(len(peak_indices), int(c), dtype=np.int32)
            waveforms = []
            
            for spike_time in spike_times:
                window_start = spike_time - self.chunk
                window_end = spike_time + self.chunk * 2
                waveforms.append(channel_data[window_start:window_end])
            waveforms = np.array(waveforms, dtype=np.float32)
                
        else:
            spike_times = np.array([], dtype=np.int32)
            electrodes = np.array([], dtype=np.int32)
            waveforms = np.array([], dtype=np.float32)
        del channel_data
        gc.collect()
        return spike_times, electrodes, waveforms

    def detect_spikes(self, data, threshold_low, threshold_high):
        detect_interval_samples = int(self.detect_interval * self.sample_rate)
        chunk = int(detect_interval_samples / 4)
        min_interval = 10  
        spike_times = []
        if threshold_low != 0:
            # neg spikes
            potential_spikes = np.where(data < threshold_low)[0]
            print(f'{len(potential_spikes)} spikes intially...')
           
            if len(potential_spikes) > 0:
                if potential_spikes[0] - self.chunk >= 0 and (potential_spikes[0] + self.chunk * 2) <= len(data):
                    spike_times.append(potential_spikes[0])

                for i in range(1, len(potential_spikes)):
                    if potential_spikes[i] - potential_spikes[i - 1] > min_interval and (potential_spikes[i] - self.chunk >= 0) and (potential_spikes[i] + self.chunk * 2) <= len(data):
                        spike_times.append(potential_spikes[i])
            del potential_spikes
            
            filtered_spike_times = []
            for spike_time in spike_times:
                window_start = spike_time
                window_end = spike_time + chunk * 3
                spike_window = data[window_start:window_end]
                
                peak_value = np.max(spike_window)  

                half_max = peak_value / 2
                indices = np.where(spike_window >= half_max)[0]
                if indices.size > 1:
                    left_idx = indices[0]
                    right_idx = indices[-1]
                    peak_width = right_idx - left_idx
                else:
                    peak_width = 0

                if chunk//2 <= peak_width < chunk * 3:
                    filtered_spike_times.append(spike_time)
        else:
            # pos spikes
            potential_spikes = np.where(data > threshold_high)[0]
            if len(potential_spikes) > 0:
                if potential_spikes[0] - self.chunk >= 0 and (potential_spikes[0] + self.chunk * 2) <= len(data):
                    spike_times.append(potential_spikes[0])

                for i in range(1, len(potential_spikes)):
                    if potential_spikes[i] - potential_spikes[i - 1] > min_interval and (potential_spikes[i] - self.chunk >= 0) and (potential_spikes[i] + self.chunk * 2) <= len(data):
                        spike_times.append(potential_spikes[i])
            del potential_spikes
            
            filtered_spike_times = []
            for spike_time in spike_times:
                
                window_start = spike_time - chunk
                window_end = spike_time + chunk * 2
                spike_window = data[window_start:window_end]

              
                peak_value = np.max(spike_window)  

                half_max = peak_value / 2
                indices = np.where(spike_window >= half_max)[0]
                if indices.size > 1:
                    left_idx = indices[0]
                    right_idx = indices[-1]
                    peak_width = right_idx - left_idx
                else:
                    peak_width = 0

                if chunk//2 <= peak_width < chunk * 3:
                    filtered_spike_times.append(spike_time)
        del spike_times
        return filtered_spike_times

    def bandpass_filter(self, data, chunk_len, num_chunks):
        nyquist = self.sample_rate / 2.0
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        b, a = signal.butter(self.filter_order, [low, high], btype='bandpass')
        if self.amplifier_filtered_filepath is not None:
            with h5py.File(self.amplifier_filtered_filepath, 'w') as hf:
                maxshape = (self.num_channels, data.shape[1])
                hf.create_dataset('preprocess_data', shape=(self.num_channels, 0), maxshape=maxshape)
                hf['preprocess_data'].resize(maxshape)
                del maxshape
                hf.flush()
                fd = hf.id.get_vfd_handle()
                os.fsync(fd)
                hf.close()
            with ThreadPoolExecutor(max_workers=self.max_workers_preprocess) as executor:
                futures = []
                for i in range(num_chunks):
                    start = i * chunk_len
                    end = (i + 1) * chunk_len if i < num_chunks - 1 else data.shape[1]
                    data_chunk = data[:, start:end]
                    
                    futures.append(
                        executor.submit(self.bandpass_filter_chunk, b, a, data_chunk, self.adc_to_uV, start, end))
                    gc.collect()
                for future in as_completed(futures):
                    future.result()  
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers_preprocess) as executor:
                futures = []
                for i in range(num_chunks):
                    start = i * chunk_len
                    end = (i + 1) * chunk_len if i < num_chunks - 1 else data.shape[1]
                    data_chunk = data[:, start:end]
                    
                    futures.append(
                        executor.submit(self.bandpass_filter_chunk, b, a, data_chunk, self.adc_to_uV, start, end))
                    gc.collect()

                for future in as_completed(futures):
                    chunk_result, start_idx, end_idx = future.result()
                    self.filtered_data[:, start_idx:end_idx] = chunk_result

                    del chunk_result
                    gc.collect()


    def bandpass_filter_chunk(self, b, a, data_chunk, adc_to_uV, start_idx, end_idx):
        car_data = self.remove_CAR(data_chunk)
        filtered_chunk = signal.filtfilt(b, a, car_data.astype(np.float32) * adc_to_uV, axis=1)

        if self.amplifier_filtered_filepath is None:
            return filtered_chunk, start_idx, end_idx
        else:
            self.file_lock.acquire()
            try:
                with h5py.File(self.amplifier_filtered_filepath, 'a') as hf:
                    dataset = hf['preprocess_data']
                    dataset[:, start_idx:end_idx] = filtered_chunk
                    hf.flush()
                    fd = hf.id.get_vfd_handle()
                    os.fsync(fd)
                    hf.close()
            finally:
                self.file_lock.release()

            del filtered_chunk
            gc.collect()

    def remove_CAR(self, data):
        """remove CAR, common average reference by median"""
        mean = np.mean(data, axis=1, keepdims=True).astype(data.dtype)
        data = data - mean
        if data.shape[0] != 1:
            median = np.median(data, axis=0, keepdims=True).astype(data.dtype)
            data = data - median
        
        return data


    