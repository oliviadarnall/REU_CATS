# DCATVD: Domestic Cat Age Transition Vocalization Dataset

This repository accompanies the paper "DCATVD: A Large-Scale Longitudinal Dataset for Analysis of Domestic Cat Vocal Development".

The data is available at this link: X

A comprehensive machine learning pipeline for analyzing domestic cat vocalizations. This project encompasses audio preprocessing, cat-call separation, feature extraction, clustering, and classification of cat vocalizations into cat-call vocalizations.

This repository also introduces a thoroughly curated Domestic Cat Age Transition Vocalization Dataset (DCATVD), which includes accurate metadata for 156 individual cats from 12 different breeds. This metadata includes precise birth-date, breed, age group, and individual cat ID in accordance with each of the 9,135 meow units (totaling 4.42 hours). This is the largest open-source cat dataset of this type, designed to enable novel and robust research regarding feline vocal development.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Modules](#modules)
  - [Data Selection](#channel-selection)
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

This project analyzes corvid vocalizations using state-of-the-art deep learning models and traditional signal processing techniques. The pipeline processes raw audio recordings, separates individual calls, extracts acoustic features, and identifies call types across multiple domestic cat breeds including:

- **Abyssinian**
- **Bengal**
- **Birman** 
- **British Shorthair**
- **Devon Rex**
- **Maine Coon**
- **Persian**
- **Ragdoll**
- **Siamese**
- **Siberian**
- **Sphynx**
- **Tonkinese**

The research aims to understand vocal repertoires and identify and analyze acoustic communication patterns of domestic cats of varying ages and breeds.

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
│   ├── data_selection/     # Selection of data from social media
│   ├── denoising/          # Audio preprocessing and noise reduction
│   ├── separation/         # Call detection and segmentation
│   ├── classification/     # Feline vzcn and call-type classification
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