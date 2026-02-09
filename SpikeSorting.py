import math
import h5py
import numpy as np
import umap
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import datetime
from AttenModel import ModelConfig
from NeuroSort import CustomDataset, ModelEncoder, ModelDecoder, SpikeSort, set_seed, find_consecutive_indices, \
    find_indices
from ContrasAug import my_transform
from SpikeUtils import SpikeDetection
import torch
import torch.optim as optim
import os
import sys
import time
import threading
import psutil

def _run_and_monitor_cpu_peak(func, *args, **kwargs):
    result = [None]
    def run():
        result[0] = func(*args, **kwargs)
    t = threading.Thread(target=run)
    t.start()
    max_cpu = 0.0
    max_rss_mb = 0.0
    proc = psutil.Process() if psutil else None
    while t.is_alive():
        time.sleep(0.2)
        if psutil:
            max_cpu = max(max_cpu, psutil.cpu_percent())
            max_rss_mb = max(max_rss_mb, proc.memory_info().rss / (1024 * 1024))
    t.join()
    return result[0], max_cpu, max_rss_mb

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_device(0)

# Configuration parameters
params = {
    # set preprocessing and detection parameters 
    'filter': 'bandpass',
    'filter_low': 250,
    'filter_high': 7000,
    'filter_order': 3,
    'threshold': 7,  # between 3 and 7 is commonly used, but this can vary depending on the application.
    'pos_neg_detect': -1,  # 1 means positive detection, -1 means negative detection
    'define_threshold': None,  # If this is none, the 'threshold' will be used to adaptively detect spikes.
    'detect_interval': 0.002,
    'adc_to_uV': 0.195,    # neuropixel use 0.195 to convert adc to uV
    'num_channels': 100,   
    'sample_rate': 20000,  
    'is_electrode_correlation': False,  # If False, do not need to sort channel number
    'directory': '../dataset',  # Dir of the raw data
    'filename': 'demo.dat',    # Filename of the raw data
    'num_chunks': 1,  # Number of chunks for Multithreaded processing
    'max_workers_preprocess': 1,
    'max_workers_detect': 1,
    # set training parameters of the spike sorting algorithm 
    'windowForTrain': 48,   # the waveform length for training, it must be divisible by 4
    'epoch': 20,
    'batch_size': 256,
    'patience': 2,  # tolerance, means after several epochs the MI stopped training without a significant increase
    'lossVariance_threshold': 1e-4,  # variance threshold for iic_loss
    'seed': 1,
    'model_output_dictionary': 'model',  # Dir for saving model
    'encoder_filename': None,  # If set the filename like 'encoder_parameters.pth', the model paramenters will be saved
    'decoder_filename': None,  
    'amplifier_filtered_filename': None,  #If not None, .h5 file will be used to save preprocessed data, not fast mode
    'spikeInfo_filename': 'spikeInfo.h5',  # used to save results, including spike time, channel
    'save_waveform': False
}

directory = params.get('directory')
filename = params.get('filename')
# Save the output data to subdirectory
output_directory = os.path.join(directory, 'results')
if params.get('encoder_filename') is not None or params.get('decoder_filename') is not None:
    model_directory = os.path.join(output_directory, params.get('model_output_dictionary'))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
spikeInfo_filepath = os.path.join(output_directory, params.get('spikeInfo_filename'))

if __name__ == '__main__':
    s = params.get('seed')
    spike_window = int(params.get('sample_rate') * 0.003)
    chunk = int(spike_window // 3)
    max_num_units = 1 << (params.get('num_channels') - 1).bit_length()
    if os.path.exists(spikeInfo_filepath):
        os.remove(spikeInfo_filepath)
        print(f"The old file {spikeInfo_filepath} is delete.")

    if not os.path.exists(spikeInfo_filepath): 
        print(f"{spikeInfo_filepath} is not exits, will create this file.")
        # Input the location of your electrophysiology data
        file_path = os.path.join(directory, filename)
        data = np.memmap(file_path, dtype=np.int16, mode='r')
        data = data.reshape(-1, params.get('num_channels')).T
        # Initialize the SpikeDetection object with the parameters
        detector = SpikeDetection(params, output_directory)
        # return the detected spikes
        preprocess_ret, cpu_peak, rss_peak_mb = _run_and_monitor_cpu_peak(detector.run_preprocess, data)
        preprocess_time = preprocess_ret
        print(f'Preprocess: peak CPU {cpu_peak:.1f}%, peak RAM {rss_peak_mb:.1f} MB')
        del data
        det_ret, cpu_peak, rss_peak_mb = _run_and_monitor_cpu_peak(detector.run_detection)
        detection_time, time, electrode, spike = det_ret
        print(f'Detection: peak CPU {cpu_peak:.1f}%, peak RAM {rss_peak_mb:.1f} MB')

        spike = spike[:,
                chunk - params.get('windowForTrain') // 4: chunk + params.get('windowForTrain') // 4 * 3]
        print(
            f'Training data: spike shape {spike.shape}, time shape {time.shape}, electrode shape {electrode.shape}.')

    with h5py.File(spikeInfo_filepath, 'a') as f2:
        group_name = 'spikeWindow_' + str(params.get('windowForTrain'))
        if group_name in f2:
            windowInfo = f2[group_name]
            spike = windowInfo['waveform']
            time = windowInfo['time']
            electrode = windowInfo['electrode']
            spike = spike[:,
                    chunk - params.get('windowForTrain') // 4: chunk + params.get('windowForTrain') // 4 * 3]
            print(
                f'Training data: spike shape {spike.shape}, time shape {time.shape}, electrode shape {electrode.shape}.')
        else:
            windowInfo = f2.create_group(group_name)
        if 'sortInfo' in windowInfo:
            del windowInfo['sortInfo']
        sortInfo = windowInfo.create_group('sortInfo')
        for ss in range(1):
            sss = s + ss
            set_seed(sss)
            seed_name = f'seed_{sss}'
            seedInfo = sortInfo.create_group(seed_name)

            # spike sorting
            tree_depth = max(int(math.log2(max_num_units / 16)), 1)
        
            print("Depth of tree is %d." % tree_depth)
            batch_size = params.get('batch_size', 512)
            num_nodes_time = int(8192 * (2 ** int(round(math.log2(batch_size / 512)))))
            config = ModelConfig(feature_dim=spike.shape[1], num_channels=params.get('num_channels'),
                                tree_depth=tree_depth, num_nodes_time=num_nodes_time)

            independent_rng = torch.Generator()
            independent_rng.manual_seed(torch.initial_seed())  

            dataset = CustomDataset(spike, electrode, time, config, transform=my_transform)
            train_dataloader = DataLoader(dataset, params.get('batch_size'), shuffle=True, num_workers=4)
            test_dataloader = DataLoader(dataset, params.get('batch_size'), shuffle=False, num_workers=4)

            spike_info = [np.max(spike), np.min(spike)]
            time_info = [np.max(time), np.min(time)]

            # initialize the model
            encoder = ModelEncoder(config, independent_rng, params.get('is_electrode_correlation'), spike_info, time_info).to('cuda')
            decoder = ModelDecoder(config, output_dim=max_num_units).to('cuda')
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
            spikeSort = SpikeSort(params)

            # save parameters of encoder and decoder
            if params.get('encoder_filename') is not None:
                encoder_filepath = os.path.join(model_directory, 'seed_' + str(sss) + params.get('encoder_filename'))
            if params.get('decoder_filename') is not None:
                decoder_filepath = os.path.join(model_directory, 'seed_' + str(sss) + params.get('decoder_filename'))
            start = datetime.datetime.now()
            torch.cuda.reset_peak_memory_stats()
            raw_label = spikeSort.spikeSorting(encoder, decoder, encoder_optimizer,
                                                        decoder_optimizer,
                                                        train_dataloader, test_dataloader)
            end = datetime.datetime.now()
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f'Clustering: peak GPU memory {peak_gpu_mb:.1f} MB')
            sort_time = end - start
            print('Spike sorting time lasting:', sort_time)

            if 'sort_runtime' in seedInfo:
                del seedInfo['sort_runtime']

            seedInfo.create_dataset('sort_runtime', data=sort_time.total_seconds())
            del sort_time
            if params.get('encoder_filename') is not None:
                torch.save(encoder.state_dict(), encoder_filepath)
            if params.get('decoder_filename') is not None:
                torch.save(decoder.state_dict(), decoder_filepath)

            unique_elements, inverse_indices = torch.unique(raw_label, sorted=True, return_inverse=True)
            units_tacsort_raw = unique_elements.size(0)

            # check
            start = datetime.datetime.now()
            label = max_num_units
            if params.get('is_electrode_correlation'):
                for i in range(max_num_units):
                    indices = np.where(raw_label == i)[0]
                    if len(electrode[indices]) != 0:
                        new_indices = find_consecutive_indices(electrode[indices])
                        if len(new_indices) > 1:
                            for j in range(len(new_indices) - 1):
                                raw_label[indices[new_indices[j + 1]]] = label
                                label += 1
                unique_elements, new_pre_label_ = torch.unique(raw_label, sorted=True, return_inverse=True)
                
                new_pre_label = new_pre_label_ + 1
                units = np.unique(new_pre_label)
                max_unit = units[-1]
                for u in units:
                    index = np.where(new_pre_label == u)[0]
                    index_time = time[index]
                    index_elec = electrode[index]
                    index_spike = spike[index]

                    unique_e = np.unique(index_elec)

                    for e in unique_e:
                        idx_e = np.where(index_elec == e)[0]
                        e_time = index_time[idx_e]
                        e_spike = index_spike[idx_e]

                        sorted_indices = np.argsort(e_time)
                        sorted_index_time = e_time[sorted_indices]

                        segment_start = 0
                        while segment_start < len(sorted_index_time):
                            segment_end = segment_start
                            while segment_end < len(sorted_index_time) - 1 and (
                                    sorted_index_time[segment_end + 1] - sorted_index_time[
                                segment_end] <= params.get('sample_rate') * 0.0006):
                                segment_end += 1

                            segment_indices = sorted_indices[segment_start:segment_end + 1]
                            if len(segment_indices) > 1:
                                
                                max_spike_index = np.argmax(
                                    np.max(np.abs(index_spike[segment_indices, :]), axis=1))
                                middle_index = segment_indices[max_spike_index]
                                
                                for idx in segment_indices:
                                    if idx != middle_index:
                                        new_pre_label[index[idx]] = -u
                            
                            segment_start = segment_end + 1
            else:
                for i in range(max_num_units):
                    indices = np.where(raw_label == i)[0]
                    new_indices = find_indices(electrode[indices].reshape(-1))
                    if len(new_indices) > 1:
                        for j in range(len(new_indices) - 1):
                            raw_label[indices[new_indices[j + 1]]] = label
                            label += 1
                unique_elements, new_pre_label = torch.unique(raw_label, sorted=True, return_inverse=True)
            

            end = datetime.datetime.now()
            check_time = (end - start).total_seconds()
            print(f'Checking time lasting {check_time} seconds. Update {len(unique_elements)} units.')
            seedInfo.create_dataset('check_runtime', data=check_time)

            if 'pre_label' in seedInfo:
                del seedInfo['pre_label']
            seedInfo.create_dataset('pre_label', data=new_pre_label)
            print(f'Spike sorting result is saved.')
        if 'time' not in windowInfo:
            windowInfo.create_dataset('time', data=time)
            windowInfo.create_dataset('electrode', data=electrode)
            if params.get('save_waveform'):
                windowInfo.create_dataset('waveform', data=spike)






