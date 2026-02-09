# NeuroSort

A deep learning-based spike sorting pipeline.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Overview

NeuroSort is an automated spike sorting tool. It combines traditional signal processing with deep learning to achieve accurate and efficient spike detection and clustering.

## ✨ Key Features

- **🧠 Advanced Spike Detection**: Adaptive threshold-based detection with waveform characterization
- **🤖 Deep Learning Clustering**: Encoder-decoder architecture for automatic feature learning
- **🔬 High-Density Array Support**: Optimized for Neuropixels (384 channels) and Neuroscroll (1024 channels) probe
- **⚡ High Performance**: Multi-threading and GPU acceleration support
- **📊 Visualization Ready**: Compatible with Phy for manual curation
- **🔧 Highly Configurable**: Flexible parameters for various experimental setups

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
git clone https://github.com/NeuroAILand/NeuroSort.git
cd NeuroSort
conda env create -f environment.yaml
conda activate pytorch_gpu
```

## 🧪 Demo Dataset

We provide a simulated dataset `dataset/demo.dat` to help you quickly test the pipeline. This dataset was generated using real neural data as the foundation:

### Data Source
The underlying neural waveforms are derived from the **publicly available extracellular dataset with known ground truth** collected by Matthew G. Perich, available through CRCNS (https://crcns.org/data-sets/motor-cortex/pmd-1/about-pmd-1).

### Simulation Method
1. **Realistic Waveform Embedding**: Taking spike waveforms from real neural units and embedding them at their occurrence times across channels
2. **Realistic Noise**: Adding bandpass-filtered noise at approximately 30 µV RMS
3. **Quantization**: Converting to int16 format matching Neuropixels data acquisition

### Download Demo Data

Download from Google Drive: [demo.dat.gz](https://drive.google.com/file/d/1K5CyS1lPZrEDx6pAYNFJpJXRNeAEcSuw/view?usp=sharing) (165 MB compressed)

After downloading:
```bash
mv ~/Downloads/demo.dat.gz dataset/  # Move to dataset folder
gzip -d dataset/demo.dat.gz          # Decompress

### Demo Configuration
To run the demo, use these parameters in `SpikeSorting.py`:

```python
params = {
    'directory': '../dataset',
    'filename': 'demo.dat',
    'num_channels': 100,
    'sample_rate': 20000,
    'threshold': 7,  # Demo-specific threshold
    'is_electrode_correlation': False,
    'batch_size': 256,  # Smaller batch for demo
    'num_chunks': 1,  # Single chunk processing
    'max_workers_preprocess': 1,
    'max_workers_detect': 1,
    'patience': 2,
    # ... other parameters
}
```

## ⏱️ Performance Benchmarks

### Expected Run Time (Demo Dataset)
On a "normal" desktop computer with:
- **CPU**: Intel Core i7-12700K or equivalent
- **GPU**: NVIDIA RTX 3080 (12GB VRAM)
- **RAM**: 32GB DDR4
- **Storage**: NVMe SSD

The demo pipeline completes in approximately **35 seconds**:

| Stage | Time | Peak Memory |
|-------|------|-------------|
| **Data Filtering** | 7.06 seconds | 4.2 GB RAM |
| **Spike Detection** | 8.86 seconds | 1.0 GB RAM |
| **Spike Sorting** | 19.38 seconds | 900 MB GPU RAM |
| **Validation** | 0.01 seconds | Minimal |
| **Total** | **35.31 seconds** | **4.2 GB RAM max** |

**Note**: Performance scales with data size. Full Neuropixels recordings (1-2 hours) typically take 10-30 minutes depending on spike density.

### Typical Install Time
- **Environment Setup**: 3-5 minutes (conda environment creation)
- **Dependency Installation**: 2-4 minutes (PyTorch + dependencies)
- **Total**: **5-9 minutes** on a standard desktop with good internet connection

## 🖥️ Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores (Intel i5 / AMD Ryzen 5 or better)
- **RAM**: 8 GB (16 GB recommended for large datasets)
- **GPU**: NVIDIA GPU with 4+ GB VRAM and CUDA support
- **Storage**: 10 GB free space

### Recommended Configuration
- **CPU**: 8+ cores (Intel i7 / AMD Ryzen 7)
- **RAM**: 16-32 GB
- **GPU**: NVIDIA RTX 3060+ with 8+ GB VRAM
- **Storage**: NVMe SSD for optimal I/O performance


## 🔬 Demo Validation

After running the demo, you should expect:

1. **Output Files**: `spikeInfo.h5` containing:
   - Detected spike times, detection channels and waveforms
   - Automatic cluster assignments

2. **Visual Validation**: Use the provided tutorial to load results into Phy:
   ```bash
   python tutorials/load_result.py
   phy template-gui params.py
   ```


## 📈 Scaling to Larger Datasets

For full experimental recordings, consider these adjustments:

```python
params = {
    'num_chunks': 4,  # Parallel processing
    'max_workers_preprocess': 4,
    'max_workers_detect': 4,
    'batch_size': 4096,  # Larger batches for efficiency
    'threshold': 5,  # Standard threshold
    # ... other parameters
}
```


## 📖 Quick Start

### 1. Configure Your Data

Update the parameters in `SpikeSorting.py`:

```python
params = {
    'directory': '/path/to/your/data',
    'filename': 'continuous.dat',
    'num_channels': 384,
    'sample_rate': 30000,
    # ... other parameters
}
```

### 2. Run Spike Sorting

```bash
python SpikeSorting.py
```

### 3. Visualize Results (Optional)

Use the provided conversion script to prepare data for Phy:

```bash
python tutorials/load_result.py
phy template-gui params.py
```

## ⚙️ Configuration

### Essential Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `threshold` | Spike detection threshold (× RMS) | 5 |
| `filter_low/high` | Bandpass filter range (Hz) | 250-7000 |
| `batch_size` | Training batch size | 4096 |
| `epoch` | Training epochs | 20 |

### Data Paths

```python
params = {
    'directory': '/spikesorting/neuropixel',  # Raw data directory
    'filename': 'continuous.dat',             # Raw data file
    'spikeInfo_filename': 'spikeInfo.h5',     # Output file
}
```

## 📊 Input Data Format

### Raw Data
- **Format**: Binary file (`.dat`)
- **Data type**: `int16`
- **Neuropixels conversion**: 0.195 μV/ADC

### Output Structure
Results are saved in HDF5 format containing:
- `spike_times`: Spike timestamps
- `spike_electrodes`: Detection channels  
- `spike_waveforms`: Spike waveforms
- `cluster_labels`: Cluster assignments

## 🏗️ Pipeline Architecture

1. **Preprocessing**
   - Bandpass filtering (250-7000 Hz)
   - Adaptive spike detection
   - Waveform extraction and alignment

2. **Feature Learning**
   - Encoder: Learns compact spike representations
   - Decoder: Generates cluster assignments

3. **Post-processing**
   - Electrode correlation validation

## 📁 Project Structure

```
NeuroSort/
├── SpikeSorting.py         # Main entry point
├── NeuroSort.py            # Core algorithm modules
├── AttenModel.py           # Model architecture
├── SpikeUtils              # Utility functions for Preprocessing and Spike detection
├── ContrasAug.py           # Data augmentation
├── dataset/                # New directory for demo dataset
│   └── demo.dat            # Simulated demo dataset, Download dataset and Decompress here
├── tutorials/
│   └── load_result.ipynb   # Phy conversion utility
└── environment.yaml        # Dependencies
```

## 🔧 Customization

### For Different Electrode Arrays

Modify the electrode geometry in `create_full_neuropixels_layout()`:

```python
def create_full_neuropixels_layout(n_channels):
    # Adjust these parameters for your probe:
    vertical_spacing = 20    # µm between rows
    horizontal_spacing = 32  # µm between columns
    row_offset = 16          # µm horizontal shift
    # ... implementation
```

### For Different Data Types

Update the `dtype` in 'SpikeSorting.py' and `create_params_file()`:

```python
params_content = f'''
dtype = 'int16'  # Change to `uint16', `int32', `float32' or your data type
'''
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Support

- 📧 Email: LXL517@student.bham.ac.uk
- 🐛 Issues: [GitHub Issues](https://github.com/NeuroAILand/NeuroSort/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/NeuroAILand/NeuroSort/discussions)

---

**Note**: Make sure to adjust electrode geometry parameters in `create_full_neuropixels_layout` for different probe types.
