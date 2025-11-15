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

## 📖 Quick Start

### 1. Configure Your Data

Update the parameters in `main.py`:

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
