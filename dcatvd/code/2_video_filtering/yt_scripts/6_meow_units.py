"""
Extract the Meow Units (MUs) from the BEATS-SED Meowseqs

INPUT
wav_folder/
beats_output.jsonl

OUTPUT
meow_units/
audio/
spectrograms/
waveforms/
metadata.csv
metadata.jsonl
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from tqdm import tqdm

#Configure the input and output folders
WAV_FOLDER = Path("downloads/dataset_filtered/accepted/wav")
JSONL_FILE = Path("beats_output1.jsonl")
OUTPUT_DIR = Path("meow_units")

#Output subfolder directories
AUDIO_DIR = OUTPUT_DIR / "audio"
SPEC_DIR = OUTPUT_DIR / "spectrograms"
WAVE_DIR = OUTPUT_DIR / "waveforms"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)
WAVE_DIR.mkdir(parents=True, exist_ok=True)

#sr = sampling rate, number of audio samples per seconds
TARGET_SR = 16000

#Minimum duration for each meow unit in seconds
MIN_MU_DURATION = 0.1

metadata_rows = []

existing_metadata_file = (OUTPUT_DIR / "metadata.csv")
processed_files = set()

if existing_metadata_file.exists():
    try:
        existing_df = pd.read_csv(existing_metadata_file)

        if "source_file" in existing_df.columns:
            processed_files = set(
                existing_df["source_file"]
                .dropna()
                .astype(str)
            )

        metadata_rows = (existing_df.to_dict("records"))
        mu_counter = len(metadata_rows)

        print(
            f"Found existing metadata with "
            f"{len(metadata_rows)} MUs"
        )

        print(
            f"Skipping "
            f"{len(processed_files)} "
            f"already processed videos"
        )

    except Exception as e:
        print(f"Could not load existing metadata: {e}")
        mu_counter = 0
else:
    mu_counter = 0

MU_BUFFER_SEC = 0.25
MIN_SAVE_DURATION = 1.0

from collections import defaultdict

print("Indexing WAV files")

wav_lookup = defaultdict(list)
for wav_file in WAV_FOLDER.rglob("*.wav"):
    wav_lookup[wav_file.stem].append(wav_file)

print(
    f"Indexed {sum(len(v) for v in wav_lookup.values())} "
    f"WAV files"
)

#--------------------------------------------------------------------------------
#Compute the local signal variability feature, 𝐹std (𝑘), the standard deviation of
#amplitudes in overlapping frames (𝐿feat = 1024, 𝐻feat = 256) of the input audio

L_FEAT = 1024 #Length of a frame
H_FEAT = 256 #Hop length (number of audio samples between the start of consecutive analysis frames)

def compute_variability_feature(audio):
    #Not enough data to fill one frame
    if len(audio) < L_FEAT:
        return np.array([])

    #slice a data array into overlapping frames
    frames = librosa.util.frame(
        audio,
        frame_length=L_FEAT,
        hop_length=H_FEAT
    )

    return np.std(frames, axis=0)

#--------------------------------------------------------------------------------
#The candidates are validated by confirming an energy drop using 
#a local amplitude envelope 𝐸(𝑡). 𝐸(𝑡) is computed from frames (𝐿env = 256, 𝐻env = 64)
#using maximum absolute amplitude (Eq. 3) and smoothed with a
#median filter (𝑀env = 5). 

#Envelope = audio signal's onset strength, traces the peak volume over time

L_ENV = 256
H_ENV = 64
MEDIAN_SIZE = 5

def compute_envelope(audio):
    #Not enough data to fill one frame
    if len(audio) < L_ENV:
        return np.array([])

    #slice a data array into overlapping frames
    frames = librosa.util.frame(
        audio,
        frame_length=L_ENV,
        hop_length=H_ENV
    )

    #Get the maximum of the volume over the frames
    env = np.max(np.abs(frames), axis=0)

    #remove noise from the signal, smooth with the median filter
    env = median_filter(env,  size=MEDIAN_SIZE)
    return env


#--------------------------------------------------------------------------------
#Segmentation of the MUs

GAUSSIAN_SIGMA = 2.0
PROMINENCE = 0.0001

#Find the boundaries of the MUs
def find_mu_boundaries(audio, sr):

    #Compute features
    F_std = compute_variability_feature(audio)
    if len(F_std) == 0:
        return []

    #𝐹std (𝑘) is smoothed with a Gaussian filter (𝜎smooth = 2.0) to 𝐹smooth(𝑘)
    #reducing noise and highlighting sustained variability changes
    #following established audio segmentation techniques
    F_smooth = gaussian_filter1d(F_std, sigma=GAUSSIAN_SIGMA)

    #BU boundaries (𝐵cand) are hypothesized at local minima in 𝐹smooth(𝑘)
    #identified via peakfinding on −𝐹smooth(𝑘) (𝑝 prominence = 0.0001). 
    candidates, _ = find_peaks(-F_smooth, prominence=PROMINENCE)

    #Confirm candidates from an energy drop using a local amplitude envelope 𝐸(𝑡)
    env = compute_envelope(audio)
    if len(env) == 0:
        return []
    env_times = (np.arange(len(env)) * H_ENV / sr)

    valid_boundaries = []

    for c in candidates:
        boundary_time = c * H_FEAT / sr

        env_idx = np.argmin(np.abs(env_times - boundary_time))

        #Get the left and right indexes
        left_idx = max(0, env_idx - 50)
        right_idx = min(len(env), env_idx + 50)

        #Diregard if they are too close together
        if right_idx - left_idx < 10:
            continue

        boundary_amp = env[env_idx]

        #Get the leftmost and rightmost peaks from the envelope
        left_peak = np.max(env[left_idx:env_idx + 1])
        right_peak = np.max(env[env_idx:right_idx])

        if left_peak <= 0 or right_peak <= 0:
            continue
        
        #If the amplitude at the boundaries are under the dip ratio 
        #then valid boundary, add to list of boundaries
        if (
            boundary_amp < LOCAL_DIP_RATIO * left_peak
            and
            boundary_amp < LOCAL_DIP_RATIO * right_peak
        ):
            valid_boundaries.append(boundary_time)

    return valid_boundaries

#--------------------------------------------------------------------------------
#MU Refinement
#Initial BUs (𝐵𝑖 ∈ 𝐵initial), whether from
#detected Inter-BU Pauses or overall Barkseq limits, are refined
#by trimming leading/trailing low-energy segments.

#Tuning specific to cats
LOCAL_DIP_RATIO = 0.40
TRIM_RATIO = 0.20

def refine_mu(audio, sr):

    #For each 𝐵𝑖’s audio A𝐵𝑖(𝑡′), a segment-specific 
    #local amplitude envelope 𝐸local(𝑡′) is computed 
    #(using 𝐿env, 𝐻env, 𝑀env parameters identical to 𝐸(𝑡)’s)
    env = compute_envelope(audio)
    if len(env) == 0:
        return None

    peak = np.max(env)
    if peak <= 0:
        return None

    #A trimming threshold, 𝑇ℎtrim(𝐵𝑖), is defined as 𝑟trim = 0.2 times the
    #peak of this local envelope, 𝐸local, peak (𝐵𝑖) (Eqs. 5, 6). The segment
    #is then trimmed to span from the first point 𝑡′start to the last 𝑡′ end
    #where 𝐸local(𝑡′) ≥ 𝑇ℎtrim(𝐵𝑖)
    threshold = (TRIM_RATIO * peak)
    keep = np.where(env >= threshold)[0]

    if len(keep) == 0:
        return None

    #Verify the start and end of the Meow Unit
    start_env = keep[0]
    end_env = keep[-1]

    buffer_samples = int(MU_BUFFER_SEC * sr)

    start_sample = (int(start_env * H_ENV) - buffer_samples)
    end_sample = (int((end_env + 1) * H_ENV) + buffer_samples)

    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    duration = (end_sample - start_sample) / sr

    if duration < MIN_SAVE_DURATION:
        deficit = int((MIN_SAVE_DURATION - duration) * sr)

        left_expand = deficit // 2
        right_expand = deficit - left_expand

        start_sample = max( 0, start_sample - left_expand)
        end_sample = min(len(audio),end_sample + right_expand)

    #Trim the audio based on the start and end of the MU
    trimmed = audio[start_sample:end_sample]

    return (
        trimmed,
        start_sample / sr,
        end_sample / sr
    )

#Visualize the Meow Units for type identification later

#Save the audio into a spectrogram image
def save_spectrogram(audio, sr, output_path):
    D = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(audio)
        ),
        ref=np.max
    )
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

#Save the audio into a waveform image
def save_waveform(audio, sr, output_path):
    t = (np.arange(len(audio)) / sr)
    plt.figure( figsize=(6, 2))
    plt.plot(t, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150
    )
    plt.close()

#Main loop to process the files

with open(JSONL_FILE) as infile:
    for line in tqdm(infile):
        record = json.loads(line)

        matches = wav_lookup.get(record["filename"],[])

        if len(matches) == 0:
            print(
                f"Missing WAV: "
                f"{record['filename']}"
            )
            continue

        if len(matches) > 1:
            print(
                f"Duplicate filename: "
                f"{record['filename']}"
            )
            continue

        wav_path = matches[0]
        source_filename = wav_path.name
        if source_filename in processed_files:
            print(
                f"Skipping already processed: "
                f"{source_filename}"
            )
            continue
        if not wav_path.exists():
            print( f"Missing WAV: {wav_path}")
            continue

        try:
            #Load the audio and sample rate with librosa
            audio, sr = librosa.load(
                wav_path,
                sr=TARGET_SR,
                mono=True
            )

        except Exception as e:
            print(
                f"Failed loading "
                f"{wav_path}: {e}"
            )
            continue
        
        #Get the Meowseq events identified by the BEATS-SED
        events = record.get("events", [])

        #For each event get the saved information
        for event_idx, event in enumerate(events):
            onset = float(event["onset"])
            offset = float(event["offset"])
            score = float(event.get( "score", -1))
            start_sample = int(onset * sr)
            end_sample = int(offset * sr)
            meowseq = audio[start_sample:end_sample]
        
            #Any segment with duration < 𝑑min = 0.05 s (before or
            #after trimming) is discarded.

            #Disregard meowseq if too short
            if (len(meowseq) / sr < MIN_MU_DURATION):
                continue

            #Find the boundaries for the MUs in the meowseq
            boundaries = (find_mu_boundaries(meowseq,sr))
            split_points = ([0.0] + boundaries + [len(meowseq)/ sr])

            #Calculate the start and end split points and split the MU audio
            for i in range( len(split_points) - 1):
                seg_start = ( split_points[i])
                seg_end = (split_points[i + 1])
                mu_audio = meowseq[int(seg_start * sr): int(seg_end * sr)]

                if (len(mu_audio) / sr < MIN_MU_DURATION):
                    continue

                #Disregard if the refinement returns nothing
                refined = refine_mu(mu_audio, sr)
                if refined is None:
                    continue

                (trimmed_audio, trim_start,  trim_end) = refined
                duration = (len(trimmed_audio) / sr)

                #Disregard if the refined MU is too short
                if (duration< MIN_MU_DURATION):
                    continue
                
                #ID Number for the MU to be saved as
                mu_id = (
                    f"{wav_path.stem}"
                    f"_MU_{mu_counter:04d}"
                )

                channel_id = wav_path.parent.name
                audio_dir = (AUDIO_DIR /  channel_id)
                spec_dir = (SPEC_DIR /  channel_id)
                wave_dir = (WAVE_DIR /  channel_id)
                audio_dir.mkdir(parents=True,  exist_ok=True)
                spec_dir.mkdir(parents=True,  exist_ok=True)
                wave_dir.mkdir(parents=True,  exist_ok=True)
                
                #Get the paths
                audio_path = (audio_dir /   f"{mu_id}.wav")
                spec_path = (spec_dir /f"{mu_id}.png")
                wave_path = (wave_dir /   f"{mu_id}.png")

                try:
                    #Save the MU
                    sf.write(audio_path, trimmed_audio, sr)
                    save_spectrogram( trimmed_audio, sr, spec_path)
                    save_waveform(trimmed_audio, sr, wave_path)

                except Exception as e:
                    print(
                        f"Failed saving "
                        f"{mu_id}: {e}"
                    )
                    continue

                metadata = {
                    "mu_id": mu_id,
                    "channel_id": channel_id,
                    "source_file": wav_path.name,
                    "event_index": event_idx,
                    "event_score": score,
                    "meowseq_start": onset,
                    "meowseq_end": offset,
                    "mu_start":onset + seg_start + trim_start,
                    "mu_end": onset + seg_start + trim_end,
                    "duration": duration,
                    "peak_amplitude": float(np.max(np.abs(trimmed_audio))),
                    "audio_path": str(audio_path.relative_to(OUTPUT_DIR)),
                    "spectrogram_path": str(spec_path.relative_to(OUTPUT_DIR)),
                    "waveform_path": str(wave_path.relative_to(OUTPUT_DIR)),
                    "label": ""
                }
                metadata_rows.append(metadata)
                mu_counter += 1

#Save the Metdata as csv and jsonl)

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv(
    OUTPUT_DIR /
    "metadata.csv",
    index=False
)

with open(
    OUTPUT_DIR /
    "metadata.jsonl",
    "w"
) as outfile:
    for row in metadata_rows:
        outfile.write(
            json.dumps(row)
            + "\n"
        )

print()
print(
    f"Extracted "
    f"{len(metadata_rows)} "
    f"Meow Units"
)

print(
    f"Saved to "
    f"{OUTPUT_DIR}"
)