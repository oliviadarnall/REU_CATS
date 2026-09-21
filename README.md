# DCATVD: Domestic Cat Age Transition Vocalization Dataset

This repository accompanies the paper "DCATVD: A Large-Scale Longitudinal Dataset for Analysis of Domestic Cat Vocal Development".

The data is available at this link: zenodo

A comprehensive machine learning pipeline for analyzing domestic cat vocalizations. This project encompasses audio preprocessing, cat-call separation, feature extraction, clustering, and classification of cat vocalizations into cat-call vocalizations.

This repository also introduces a thoroughly curated Domestic Cat Age Transition Vocalization Dataset (DCATVD), which includes accurate metadata for 156 individual cats from 12 different breeds. This metadata includes precise birth-date, breed, age group, and individual cat ID in accordance with each of the 9,135 meow units (totaling 4.42 hours). This is the largest open-source cat dataset of this type, designed to enable novel and robust research regarding feline vocal development.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Modules](#modules)
  - [Channel Selection](#channel-selection)
  - [Video Filtering](#video-selection)
  - [Denoising](#denoising)
  - [Separation](#separation)
  - [Meow Unit Filtering](#mu-filtering)
  - [Classification](#classification)
  - [Clustering](#clustering)
  - [Statistics](#stats)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)

---

## Overview - TO EDIT

This project analyzes corvid vocalizations using state-of-the-art deep learning models and traditional signal processing techniques. The pipeline processes raw audio recordings, separates individual calls, extracts acoustic features, and identifies call types across multiple corvid species including:

- **American Crow** (*Corvus brachyrhynchos*)
- **Common Raven** (*Corvus corax*)
- **Hooded Crow** (*Corvus cornix*)
- **Carrion Crow** (*Corvus corone*)
- **Fish Crow** (*Corvus ossifragus*)

The research aims to understand vocal repertoires, identify species-specific and cross-species call types, and analyze acoustic communication patterns.

---

## Project Structure

```
dcatvd/
├── dataset/
│   ├── raw_audio/
│   ├── denoised_audio/
│   ├── meow_units/
│   ├── ecmus/
├── code/
│   ├── channel_selection/  # Selecting channels from YT and IG
│   ├── video_filtering/    # Filtering high-quality YT videos
│   ├── denoising/          # Audio preprocessing and noise reduction
│   ├── separation/         # Call detection and segmentation
│   ├── mu_filtering/       # Additional filtering and feline-Vczn classification
│   ├── classification/     # Call-Type classification
│   ├── clustering/         # Feature extraction and clustering
│   └── statistics/         # Statistical analysis of results
```

## Pipeline Workflow

The analysis pipeline follows these stages:

```
[1. CHANNE SELETCTION] → Find high-quality candidate cat channels
    ↓
[2. VIDEO FILTERING] → Keep only videos containing cat calls
    ↓
Raw Audio
    ↓
[3. DENOISING] → Remove background noise
    ↓
[4. SEPARATION] → Detect and extract individual cat calls
    ↓
[5. ADDITIONAL FILTERING] → Keep only Meow Units containing cat calls
    ↓
[6. CLASSIFICATION] → Identify cat call-types
    ↓
[7. FEATURE EXTRACTION] → Extract acoustic embeddings
    ↓
[8. CLUSTERING] → Group similar calls
    ↓
[9. STATISTICAL ANALYSIS] → Identify patterns in cat calls
```

---

REST AFTER THIS HAS NOT BEEN EDITED YET ....


## Modules - TO EDIT

### Denoising

**Purpose**: Remove background noise from raw audio recordings to improve downstream analysis.

**Key Files**:
- `run_noisereduce.py` - Main noise reduction script using spectral gating
- `conversion.py` - Audio format conversion utilities
- `renaming.py` - Batch file renaming tools

**How it works**:
- Loads raw audio files (WAV, MP3, FLAC)
- Applies spectral noise reduction using `noisereduce` library
- Estimates noise profile from quieter portions
- Subtracts noise while preserving signal integrity
- Outputs cleaned audio for separation

**Configuration**:
- `INPUT_FOLDER` - Raw audio directory
- `OUTPUT_FOLDER` - Cleaned audio destination
- `USE_CHUNKING` - Process large files in chunks to save memory

**Usage**:
```python
python run_noisereduce.py
```

---

### Separation

**Purpose**: Detect and segment individual crow calls from continuous audio recordings.

**Key Files**:
- `split_calls.py` - PANNs-based call detection and segmentation
- `split_calls_dual_model.py` - Ensemble approach with two models
- `split_calls_panns_finetuned.py` - Uses fine-tuned PANNs model
- `split_sentences.py` - Segments longer call sequences
- `reformat_model_annotations.py` - Process model outputs

**Detection Approach**:

1. **PANNs Model** (Pretrained Audio Neural Networks):
   - Loads CNN14 architecture trained on AudioSet
   - Generates framewise confidence scores (10ms resolution)
   - Target labels: "Crow", "Bird", "Caw", "Raven", "Corvid", "Rattle", etc.

2. **Frame-level Detection**:
   - Sample rate: 32kHz (PANNs requirement)
   - Hop size: 320 samples (10ms frames)
   - Confidence threshold: 0.05 (adjustable)
   - Min call duration: 3 frames (30ms)

3. **Segmentation Logic**:
   ```python
   for each frame:
       if confidence > threshold:
           mark as "crow call"
       
   merge consecutive active frames into segments
   add padding (e.g., 3 frames) to avoid cutoffs
   extract and save individual call segments
   ```

4. **Dual Model Approach** (`split_calls_dual_model.py`):
   - Uses two models for consensus
   - Reduces false positives
   - Handles edge cases better

**Configuration**:
- `CONF_THRESHOLD` - Detection sensitivity (0.01-0.5)
- `MIN_CALL_FRAMES` - Minimum call length
- `MAX_GAP_FRAMES` - Tolerance for brief silences within calls
- `NUM_PADDING_FRAMES` - Buffer around detected calls

**Outputs**:
- Individual call WAV files
- CSV annotations with timestamps and confidence scores
- Framewise confidence plots for visualization

---

### Clustering

**Purpose**: Extract acoustic features and group similar calls to discover call types.

#### Feature Extraction Methods

The project implements multiple feature extraction approaches:

**Recommended**: Use MATES-inspired features (based on Mates et al., 2015) for best clustering and classification performance. Other methods are provided for comparison and ensemble approaches.

##### 1. **AudioMAE (Audio Masked Autoencoder)** 

**Files**: `run_AudioMAE.py`, `run_audiomae_resize.py`, `run_AudioMAE_posttrained.py`

**Details**:
- Pre-trained Vision Transformer adapted for audio
- Input: 16kHz audio → Log mel-spectrogram (128 mel bins)
- Window: 1024 frames (~10.24s)
- Hop: 512 frames (50% overlap)
- Normalization: Mean=-4.27, Std=4.57
- Output: 768-dimensional embeddings per window
- Aggregation: Mean or attention pooling across windows

##### 2. **AVES (Avian Vocalization Encoder)**

**File**: `aves_features.ipynb`

**Details**:
- Specialized for bird vocalizations
- Pre-trained on large-scale bird audio dataset
- Direct application to crow calls
- Outputs species-aware embeddings

##### 3. **ResNet-based Spectrogram Embeddings**

**Files**: `resnet_embedder.py`, `resnet_embedder_max_pool_sliding_window.py`

**Details**:
- Treats mel-spectrogram as image
- Pre-trained ResNet18 (ImageNet weights)
- Extracts 512-D features from last conv layer
- Sliding window with max pooling for temporal modeling

##### 4. **MFCC (Mel-Frequency Cepstral Coefficients)**

**File**: `mfcc_vectorization.py`

**Details**:
- Traditional signal processing approach
- 13 MFCC coefficients per frame
- Statistical aggregation: mean, std, min, max
- PCA dimensionality reduction
- Fast, interpretable, baseline method

##### 5. **MATES-Inspired Features**
*Best performing method for corvid vocalization analysis*

**Files**: `crow_features_mates.py`, `crow_features_mates_extended.py`, `crow_features_mates_swipe.py`

**Details**:
- Custom acoustic analysis pipeline based on Mates et al. (2015)
- Pitch extraction using Praat (parselmouth)
- Harmonic energy analysis (12 harmonics)
- Temporal envelope statistics
- Spectral centroid, bandwidth, rolloff
- Formant tracking
- Pitch range: 100-1200 Hz (crow-specific)
- **Superior performance**: Outperforms deep learning approaches for corvid call clustering and classification

**Reference**: This approach builds on the acoustic profiling methodology introduced by Mates et al. (2015), who demonstrated that crow caws encode information on caller sex, identity, and behavioral context through specific acoustic features.

**Features Extracted**:
```python
- Pitch statistics: mean, std, min, max, range, CV
- Harmonic energy ratios (H1-H12)
- Harmonic trends over time
- Spectral moments (1st-4th)
- Temporal envelope: attack, decay, sustain
- Formant frequencies and bandwidths
```

**Why MATES-Inspired Features are Best**:
- **Domain-specific**: Tailored for corvid vocal characteristics based on established bioacoustic research (Mates et al., 2015)
- **Interpretable**: Features correspond to known acoustic properties relevant to bird vocalizations
- **Robust**: Explicitly models harmonics and pitch, which are fundamental to corvid calls
- **Empirically validated**: Consistently produces clearer cluster separation and more biologically meaningful groupings than general-purpose deep learning embeddings
- **Computationally efficient**: Faster than transformer-based models while achieving superior results
- **Biologically grounded**: Based on features proven to encode caller identity and behavioral context in American crows

##### 6. **VGGish & PANNs Embeddings**

**Files**: `vggish_embedder.py`, `panns_embedder.py`

**Details**:
- General audio embeddings
- VGGish: 128-D embeddings from YouTube-8M
- PANNs: 2048-D from AudioSet
- Pre-trained on diverse sound events

##### 7. **OpenL3**

**File**: `openl3_embedder.py`

**Details**:
- Self-supervised learned embeddings
- Environmental sound representations
- 512-D or 6144-D versions available

#### Clustering Algorithms

##### **Gaussian Mixture Models (GMM)**

**Files**: `gaussian_call_clusters.py`, `cross_species_clustering_gmm.py`

**Approach**:
```python
1. Load feature embeddings from CSV
2. Standardize features (optional)
3. UMAP dimensionality reduction (optional)
4. Fit GMM with diagonal or full covariance
5. Compute soft cluster probabilities
6. Evaluate with silhouette score
7. Visualize with UMAP scatter plots
```

**Advantages**:
- Soft clustering (probabilistic assignments)
- Models uncertainty
- Captures elliptical cluster shapes
- BIC/AIC for model selection

**Configuration**:
```python
n_components = 20-70  # Number of call types
covariance_type = 'diag' or 'full'
```

**Bayesian GMM** (`gaussian_call_clusters_bic.py`):
- Automatic component selection
- Dirichlet process prior
- Prevents overfitting

##### **K-Means Clustering**

**File**: `kmeans_clusters.py`

**Approach**:
- Hard cluster assignments
- Faster than GMM
- Good for baseline
- UMAP visualization

##### **HDBSCAN**

**File**: `visualize_hdbscan.py`

**Approach**:
- Density-based clustering
- Automatically determines number of clusters
- Handles noise and outliers
- Good for discovering natural groupings

#### Cross-Species Analysis

**File**: `cross_species_clustering_gmm.py`

**Purpose**: Compare vocal repertoires across 5 corvid species

**Workflow**:
```python
1. Load features from multiple species CSVs
2. Concatenate into single dataframe
3. Extract species labels from filenames
4. Run GMM clustering on combined data
5. Analyze cluster composition by species
6. Identify shared vs. species-specific calls
```

**Analysis Questions**:
- Which call types are species-specific?
- Which are shared across corvids?
- How do American crows differ from ravens?
- Evidence for convergent evolution?

#### Visualization Tools

- `umap_projection.py` - UMAP scatter plots
- `tsne_visualization.py` - t-SNE embeddings
- `pca_visualization.py` - PCA projections
- `pacmap_visualization.py` - PaCMAP (preserves local and global structure)
- `plot_species.py` - Species-colored scatter plots
- `plot_density.py` - Density heatmaps
- `visualize_confusion_matrix.py` - Cluster confusion matrices

#### Sequence Analysis

**Markov Models**:

**Files**: `markov_transition_order_1.py`, `markov_transition_order_2.py`

**Purpose**: Model call sequences and syntax

**Approach**:
```python
1. Load soft cluster probabilities for calls
2. Group calls by recording/individual
3. Build transition matrices:
   - Order 1: P(call_t | call_{t-1})
   - Order 2: P(call_t | call_{t-1}, call_{t-2})
4. Analyze transition probabilities
5. Identify common call sequences
```

**Questions Addressed**:
- Do crows follow grammar rules?
- Which call transitions are common?
- Context-dependent call usage?

**Other Sequence Tools**:
- `summary_sequence_stats.py` - Sequence length, diversity metrics
- `call_distributions.py` - Call type frequency distributions

#### Subclustering

**File**: `subcluster.py`

**Purpose**: Hierarchical refinement of call types

**Workflow**:
```python
1. Run initial clustering (e.g., 20 clusters)
2. For each cluster:
   - Extract calls assigned to that cluster
   - Run secondary clustering (e.g., 5 sub-clusters)
3. Create hierarchical labels (e.g., "Cluster 3, Sub-cluster 2")
```

**Use Case**: Discover subtle variations within major call types

---

### Advanced Clustering

**Purpose**: Fine-tune deep learning models for crow-specific acoustic analysis.

#### AudioMAE Post-training

**Files**: 
- `audiomae_post_training.py` - Attention pooling
- `audiomae_post_training_2.py` - Contrastive learning

**Approach 1: Attention Pooling**

**File**: `audiomae_post_training.py`

**Motivation**: Pre-trained AudioMAE uses mean pooling across time, losing temporal detail. Attention pooling learns to weight important time frames.

**Architecture**:
```python
class SimpleMAE(nn.Module):
    encoder: AudioMAE (frozen)
    decoder: Linear layers for reconstruction
    attention_pool: Multi-head attention pooling
    
    forward(mel_spectrogram):
        patches = encoder.patch_embed(mel)
        tokens = encoder.blocks(patches)
        pooled = attention_pool(tokens)  # Weighted average
        return pooled  # 768-D embedding
```

**Training**:
- Unsupervised reconstruction loss
- Input: Mel-spectrogram
- Target: Reconstruct original from embeddings
- Optimizer: Adam, lr=1e-4
- Epochs: 10-20
- Augmentation: SpecAugment (freq/time masking)

**Approach 2: Contrastive Learning**

**File**: `audiomae_post_training_2.py`

**Motivation**: Learn embeddings where similar calls are close, dissimilar calls are far.

**Method**:
```python
1. For each call:
   - Create positive pair (same call, different augmentation)
   - Sample negative pairs (other calls)

2. Contrastive Loss (NT-Xent):
   - Pull positive pairs together
   - Push negative pairs apart
   
3. Projection head:
   - encoder → projector → normalized embeddings
   - Contrastive loss on projections
   - Use encoder embeddings for downstream tasks
```

**Augmentations**:
- Time masking (mask random time segments)
- Frequency masking (mask mel bins)
- Gaussian noise addition
- Time stretching

**Benefits**:
- Learns discriminative features
- Robust to noise and variations
- Improves clustering quality

#### AudioMAE Fine-tuning for Classification

**File**: `finetune_audiomae.py`

**Purpose**: Train AudioMAE to classify crow call types

**Workflow**:
```python
1. Load labeled data (e.g., from CrowTools classifier)
2. Freeze AudioMAE encoder (transfer learning)
3. Add classification head:
   encoder → attention_pool → linear → softmax

4. Handle class imbalance:
   - Weighted sampling (oversample rare classes)
   - OVERSAMPLE_FACTOR = 5
   
5. Train with cross-entropy loss
6. Save fine-tuned model
```

**Data Format** (`crowtools_classified_updated.csv`):
```csv
filename,rattle,mob,contact,scold,...
brachyrhynchos_00001_1.wav,1,0,0,0,...
```

**Output**: `audiomae_finetuned.pth` - Model checkpoint for classification

**Usage**: `AudioMAE_finetuned_embedder.py` - Extract embeddings with fine-tuned model

#### Call Classification

**File**: `classify_calls.py`

**Purpose**: Apply CrowTools classifier to identify call functions

**Dependencies**: `crow-tools/classifier/` (external repository)

**Workflow**:
```python
1. Load embeddings (e.g., AVES base bio)
2. Load cluster probabilities
3. Concatenate features
4. Load pre-trained CrowClassifier
5. Predict call types: rattle, mob, contact, scold, etc.
6. Save predictions to CSV
```

**Call Types** (from crow-tools):
- **Rattle**: Aggressive, territorial
- **Mob**: Anti-predator, group response
- **Contact**: Communication between individuals
- **Scold**: Warning, distress

#### Label Extraction

**File**: `extract_labels.py`

**Purpose**: Create subsets of labeled data for training

**Example**:
```python
# Extract all rattle and mob calls
filtered_df = df[(df['rattle']) | (df['mob'])]

# Copy corresponding audio files
# Save filtered CSV for model training
```

---

### Counting

**Purpose**: Estimate the number of calls in long recordings without full segmentation.

**Files**:
- `save_gaussian_peaks.py` - Main peak detection algorithm
- `plot_peaks.py` - Visualization

**Algorithm**:

1. **Envelope Extraction**:
   ```python
   y, sr = librosa.load(audio_file)
   envelope = librosa.feature.rms(y=y)[0]  # Root mean square energy
   ```

2. **Smoothing**:
   ```python
   smoothed = gaussian_filter1d(envelope, sigma=GAUSSIAN_SIGMA)
   # Reduces noise, highlights major peaks
   ```

3. **Normalization**:
   ```python
   env_norm = envelope / max(envelope)
   # Scale to [0, 1]
   ```

4. **Peak Detection**:
   ```python
   peaks, _ = find_peaks(
       env_norm,
       height=RELATIVE_PEAK_HEIGHT,  # e.g., 0.3
       distance=PEAK_DISTANCE         # min frames between peaks
   )
   num_calls = len(peaks)
   ```

**Configuration**:
- `GAUSSIAN_SIGMA` - Smoothing kernel width (default: 2)
- `PEAK_DISTANCE` - Minimum frames between peaks (default: 5)
- `RELATIVE_PEAK_HEIGHT` - Threshold for peak detection (default: 0.3)

**Output**: CSV with filename and call count per file

**Use Case**: Quick call rate estimation for large datasets without full separation

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for deep learning models)
- FFmpeg (for audio I/O)

### Dependencies

```bash
# Core scientific computing
numpy
pandas
scipy
scikit-learn

# Audio processing
librosa
soundfile
torchaudio
noisereduce
parselmouth  # Praat Python wrapper
pysptk       # Speech signal processing

# Deep learning
torch
torchvision
timm         # PyTorch Image Models (for AudioMAE)
panns-inference

# Visualization
matplotlib
seaborn
umap-learn
pacmap

# Clustering
hdbscan
```

### Setup

```bash
# Clone repository
git clone https://github.com/UTA-ACL2/corvids_vocal_repertoire.git
cd corvids_vocal_repertoire

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
# AudioMAE: automatic on first run
# PANNs: automatic on first run
# CrowTools classifier: clone from separate repository
```

---

## Usage

### End-to-End Pipeline

```bash
# 1. Denoise raw audio
cd main/denoising
python run_noisereduce.py

# 2. Separate individual calls
cd ../separation
python split_calls.py

# 3. Extract features
cd ../clustering
python run_AudioMAE.py

# 4. Cluster calls
python cross_species_clustering_gmm.py

# 5. Classify call types
cd ../advanced_clustering
python classify_calls.py
```

### Individual Tasks

**Extract AudioMAE features**:
```bash
cd main/clustering
# Edit configuration in run_AudioMAE.py:
# - input_dir: your call directory
# - output_csv: output feature file
python run_AudioMAE.py
```

**Run GMM clustering**:
```bash
# Edit cross_species_clustering_gmm.py:
# - csv_files: list of feature CSVs
# - n_components: number of clusters
python cross_species_clustering_gmm.py
```

**Visualize clusters**:
```bash
# UMAP projection
python umap_projection.py

# Species-colored plot
python plot_species_from_coordinates.py
```

**Analyze sequences**:
```bash
python markov_transition_order_1.py
python summary_sequence_stats.py
```

---

## Species Supported

The pipeline has been tested on the following corvid species:

| Common Name | Scientific Name | Dataset Code |
|-------------|----------------|--------------|
| American Crow | *Corvus brachyrhynchos* | `brachyrhynchos` |
| Common Raven | *Corvus corax* | `corax` |
| Hooded Crow | *Corvus cornix* | `cornix` |
| Carrion Crow | *Corvus corone* | `corone` |
| Fish Crow | *Corvus ossifragus* | `ossifragus` |

**Note**: The same pipeline can be adapted for other bird species by adjusting:
- Frequency ranges in MATES feature extraction
- Detection labels in PANNs separation
- Species-specific model fine-tuning

---

## Technical Details

### AudioMAE Architecture

AudioMAE adapts Vision Transformers for audio:

1. **Input**: Log mel-spectrogram (time × frequency)
2. **Patching**: Divide into 16×16 patches
3. **Embedding**: Linear projection + positional encoding
4. **Transformer**: 12 layers, 768-D, 12 attention heads
5. **Pooling**: Mean or attention-weighted average
6. **Output**: 768-D embedding per audio window

**Pre-training**: Masked autoencoding on AudioSet (2M clips)

### PANNs (Pretrained Audio Neural Networks)

**Architecture**: CNN14 (14-layer convolutional neural network)

**Training**: AudioSet (632 classes, 2M 10-second clips)

**Output**: Framewise predictions (10ms resolution)

**Advantages**:
- Real-time capable
- High temporal resolution
- Generalizes to novel sounds

### Feature Aggregation Strategies

For variable-length calls, features are aggregated:

1. **Mean Pooling**: Average across time
2. **Max Pooling**: Max value per dimension
3. **Attention Pooling**: Learned weighted average
4. **Statistical**: Mean, std, min, max, quantiles

### Dimensionality Reduction

**UMAP (Uniform Manifold Approximation and Projection)**:
- Preserves local and global structure
- Non-linear, manifold-based
- Faster than t-SNE for large datasets
- Parameters: `n_neighbors`, `min_dist`, `metric`

**PCA (Principal Component Analysis)**:
- Linear, fast
- Interpretable components
- Good for preprocessing before clustering

**PaCMAP (Pairwise Controlled Manifold Approximation)**:
- Balances local and global structure
- Stable across parameter choices

### Clustering Evaluation

**Silhouette Score**: Measures cluster separation
- Range: [-1, 1]
- Higher is better
- Considers intra-cluster and inter-cluster distances

**Bayesian Information Criterion (BIC)**: Model selection for GMM
- Lower is better
- Penalizes model complexity
- Helps choose optimal number of clusters

---

## Data Organization

Expected directory structure:

```
corvids_vocal_repertoire/
├── data/
│   ├── brachyrhynchos/
│   │   ├── raw/              # Original recordings
│   │   ├── denoised/         # After noise reduction
│   │   └── calls/            # Separated calls
│   ├── corax/
│   ├── cornix/
│   └── ...
├── main/
│   ├── clustering/
│   │   └── features/
│   │       ├── brachyrhynchos/
│   │       │   └── crow_features_audiomae_brachyrhynchos.csv
│   │       └── ...
│   └── ...
└── models/
    ├── Cnn14_mAP=0.431.pth        # PANNs checkpoint
    ├── audiomae_finetuned.pth     # Fine-tuned AudioMAE
    └── ...
```

---

## Output Files

### Feature CSVs
Format: `filename, feature_1, feature_2, ..., feature_N`

Example:
```csv
filename,feat_0,feat_1,...,feat_767
brachyrhynchos_00001_1.wav,-0.234,0.567,...,0.123
brachyrhynchos_00001_2.wav,0.145,-0.432,...,-0.067
```

### Cluster Assignments

**Hard Clustering** (`kmeans_clusters.csv`):
```csv
filename,cluster,umap_1,umap_2
brachyrhynchos_00001_1.wav,3,12.5,-8.3
```

**Soft Clustering** (`soft_cluster_probabilities.csv`):
```csv
filename,Cluster_0,Cluster_1,...,Cluster_26
brachyrhynchos_00001_1.wav,0.05,0.82,...,0.01
```

### Annotations

**Separation Annotations**:
```csv
filename,start_sec,end_sec,confidence,label
recording_01.wav,5.23,5.67,0.89,crow
recording_01.wav,12.45,13.01,0.76,crow
```

### Classification Results

```csv
filename,rattle,mob,contact,scold,cluster_id
brachyrhynchos_00001_1.wav,0.95,0.02,0.01,0.02,3
brachyrhynchos_00001_2.wav,0.03,0.87,0.05,0.05,7
```

---

## Experimental Features

### Super Embeddings

**File**: `super_embeddings.py`

Concatenate multiple feature types:
```python
features = [AudioMAE, AVES, MFCC, MATES]
super_embedding = concatenate(features)
```

**Benefit**: Combines complementary information

**Challenge**: High dimensionality, requires careful normalization

### CrowTools Integration

**Directory**: `advanced_clustering/crow-tools/`

Integration with external crow call classifier:
- Pre-trained on annotated American crow calls
- Identifies call functions (alarm, contact, aggression)
- Can be fine-tuned with new data

---

## References

### Models & Methods

- **AudioMAE**: [Masked Autoencoders for Audio Spectrogram Transformers](https://arxiv.org/abs/2203.16691)
- **PANNs**: Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880-2894.
- **AVES**: [AVES: Animal Vocalization Encoder based on Self-Supervision](https://github.com/earthspecies/aves)
- **UMAP**: [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)
- **Noise Reduction**: Sainburg, T. (2019). timsainb/noisereduce: v1.0 (Version db94fe2). Zenodo. https://doi.org/10.5281/zenodo.3243139
- **AudioSet**: Gemmeke, J., Ellis, D., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., & Ritter, M. (2017). Audio Set: An ontology and human-labeled dataset for audio events. *IEEE ICASSP*, 776-780.
- **Markov Models for Birdsong**: Katahira, K., Suzuki, K., Okanoya, K., & Okada, M. (2011). Complex Sequencing Rules of Birdsong Can be Explained by Simple Hidden Markov Processes. *PLOS ONE*, 6(9), e24516.

### Corvid Vocal Communication

- **American Crow Vocalizations**: Chamberlain & Cornwell (1971)
- **Acoustic Profiling in American Crows**: Mates, E. A., Tarter, R. R., Ha, J. C., Clark, A. B., & McGowan, K. J. (2015). Acoustic profiling in a complexly social species, the American crow: caws encode information on caller sex, identity and behavioural context. *Bioacoustics*, 24(1), 63-80.
- **Vocal Communication in Corvids**: Wascher, C., & Reynolds, S. (2025). Vocal communication in corvids: a systematic review. *Animal Behaviour*, 221, 123073.
- **Raven Syntax**: Enggist-Dueblin & Pfister (2002)
- **Corvid Cognition**: Clayton & Emery (2015)
- **Crow Numerical Competence**: Liao, D. A., Brecht, K. F., Veit, L., & Nieder, A. (2024). Crows "count" the number of self-generated vocalizations. *Science*, 384(6698), 874-877.
- **Volitional Control**: Brecht, K. F., Hage, S. R., Gavrilov, N., & Nieder, A. (2019). Volitional control of vocalizations in corvid songbirds. *PLOS Biology*, 17(8), e3000375.
- **Social Learning**: Cornell, H. N., Marzluff, J. M., & Pecoraro, S. (2012). Social learning spreads knowledge about dangerous humans among American crows. *Proceedings of the Royal Society B*, 279(1728), 499-508.

---

## License

Research and educational use. Please cite if you use this code in your work.

---

## Acknowledgments

- National Science Foundation REU Program
- This is a research project developed at the University of Texas at Arlington REU program.
- Pre-trained model authors (AudioMAE, PANNs, AVES teams)
- Cornell Lab of Ornithology (Macaulay Library audio datasets)
- Xeno-canto bird sound database
- Mates et al. (2015) for foundational acoustic profiling methodology

---

## Bibliography

For a complete list of references cited in this project, see the key citations below:

**Primary Methodology Reference**:
- Mates, E. A., Tarter, R. R., Ha, J. C., Clark, A. B., & McGowan, K. J. (2015). Acoustic profiling in a complexly social species, the American crow: caws encode information on caller sex, identity and behavioural context. *Bioacoustics*, 24(1), 63-80. https://doi.org/10.1080/09524622.2014.933446

**Additional Key References**:
- Kong, Q., et al. (2020). PANNs: Large-scale pretrained audio neural networks. *IEEE/ACM TASLP*, 28, 2880-2894.
- Wascher, C., & Reynolds, S. (2025). Vocal communication in corvids: a systematic review. *Animal Behaviour*, 221, 123073.
- Liao, D. A., et al. (2024). Crows "count" the number of self-generated vocalizations. *Science*, 384(6698), 874-877.
- Gemmeke, J., et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. *IEEE ICASSP*, 776-780.
- Sainburg, T. (2019). noisereduce: v1.0. Zenodo. https://doi.org/10.5281/zenodo.3243139

For the full bibliography with all citations, please refer to the project documentation.

---

## Future Directions

1. **Real-time Detection**: Optimize for live field recording analysis
2. **Mobile Deployment**: Port models to edge devices
3. **Multi-modal Analysis**: Integrate video (posture, context)
4. **Temporal Context**: Model long-term call patterns (hours/days)
5. **Individual Recognition**: Voice fingerprinting for crow individuals
6. **Ecological Applications**: Population monitoring, behavior studies
7. **Cross-species Transfer**: Apply to other bird families

---

## Known Issues

- **Memory Usage**: AudioMAE feature extraction requires significant GPU memory (16GB+ recommended)
- **Processing Time**: Separation and feature extraction can be slow for large datasets (parallelize with `ProcessPoolExecutor`)
- **PANNs Sensitivity**: May detect non-crow vocalizations; fine-tuning recommended for production use
- **Pitch Extraction**: Occasional failures on very short or noisy calls (filtered out)

---

## Tips & Best Practices

1. **Start Small**: Test pipeline on subset (e.g., 100 calls) before full dataset
2. **Check Audio Quality**: Visualize spectrograms to ensure proper preprocessing
3. **Tune Thresholds**: Adjust detection thresholds based on your dataset
4. **Validate Clusters**: Manually inspect cluster prototypes (see `extract_top_examples.py`)
5. **Save Checkpoints**: Cache intermediate results (features, clusters) to avoid recomputation
6. **Document Parameters**: Keep a log of configurations for reproducibility