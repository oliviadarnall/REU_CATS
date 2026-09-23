# DCATVD: Domestic Cat Age Transition Vocalization Dataset

This repository accompanies the paper "DCATVD: A Large-Scale Longitudinal Dataset for Analysis of Domestic Cat Vocal Development," currently under review for the ICLR 2027 Conference.

This paper describes a machine learning pipeline for analyzing domestic cat vocalizations, encompassing audio preprocessing, cat-call separation, feature extraction, classification of cat vocalizations into cat-call vocalizations, and the clustering of cat vocalizations into elemental units.

This repository also introduces a thoroughly curated Domestic Cat Age Transition Vocalization Dataset (DCATVD), which includes accurate metadata for vocalizations from 135 individual cats from 12 different breeds. This metadata includes precise birth-date, breed, age group, and individual cat ID in accordance with each of the 9,135 meow units (totaling 4.42 hours). This is the largest open-source cat dataset of this type, designed to enable novel and robust research regarding feline vocal development.

Upon acceptance, the full dataset and code will be released. Currently, this repository contains a sample of 10 cat vocalizations selected at random from the full dataset, as well as sample code for the cat call SED, meow unit separation, and cat call-type classifier. 

## Table of Contents

- [Overview]
- [Project Structure]
- [Pipeline Workflow]

---

## Overview

Our pipeline processes raw audio recordings, separates individual cat calls, extracts acoustic features, and identifies call-types across multiple domestic cat breeds and life stages including:

Breeds:
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

Life Stages:
- **Kitten (birth - 6 months)**
- **Junior (7 months - 2 years)**
- **Prime (3 years - 6 years)**
- **Mature (7 years - 10 years)**
- **Senior (11 years - 14 years)**
- **Geriatric (15 years or older)**

We classified individual cat vocalizations into 9 cat-call classes:
- **Chatter/Chirp**
- **Growl**
- **Hiss/Spit**
- **Meow**
- **Purr**
- **Pain-Cry**
- **Trill**
- **Trill-Meow**
- **Yowl/Howl**

We also perform unsupervised clustering on smaller 25ms units from each vocalization.

The research aims to understand vocal repertoires and identify and analyze acoustic communication patterns of domestic cats of varying ages and breeds.

---

## Project Structure

At present, this repository contains a sample of 10 cat vocalizations selected at random from the full dataset, as well as sample code for the cat call SED, meow unit separation, and cat call-type classifier.

```
dcatvd_sample/
├── sample_code/
│   ├── classification/     # Cat call-type classification
│   ├── separation/         # Call detection and segmentation
├── sample_dataset/
│   ├── sample_audio/       # Random sample of 10 cat vocalizations
```

Upon acceptance, we will release the full dataset and code used in the project.
The structure of the repository will be as follows:

```
dcatvd/
├── dataset/
├── code/
│   ├── data_selection/     # Selection of data from social media
│   ├── denoising/          # Audio preprocessing and noise reduction
│   ├── separation/         # Call detection and segmentation
│   ├── classification/     # Binary feline vocalization and call-type classification
│   ├── clustering/         # Feature extraction and clustering
│   └── statistics/         # Statistical analysis of results
```

## Pipeline Workflow

Our analysis pipeline follows these stages:

```
[1. CHANNEL SELETCTION] → Find high-quality candidate cat channels
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
