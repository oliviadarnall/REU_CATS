#!/usr/bin/env python3

"""
Cat SED Inference Pipeline - Fully Automatic & Architecture-Agnostic

USAGE EXAMPLES:

  # 1. Basic: folder input, auto-detect model from checkpoint
  python inference_cat_sed_final_updated.py \
      --checkpoint /path/to/best.ckpt \
      --input_folder /path/to/audio/ \
      --output_jsonl results.jsonl

  # 2. With audio normalization (recommended if input differs from training)
  python inference_cat_sed_final_updated.py \
      --checkpoint /path/to/best.ckpt \
      --input_folder /path/to/audio/ \
      --output_jsonl results.jsonl \
      --normalize --target_lufs -23.0

  # 3. Using JSONL input (from a prior SED first-cut pipeline)
  python inference_cat_sed_final_updated.py \
      --checkpoint /path/to/best.ckpt \
      --input_jsonl /path/to/clips.jsonl \
      --output_jsonl results.jsonl

  # 4. Override config (e.g., point to different data/model paths)
  python inference_cat_sed_final_updated.py \
      --checkpoint /path/to/best.ckpt \
      --input_folder /path/to/audio/ \
      --output_jsonl results.jsonl \
      --conf_file /path/to/override.yaml

  # 5. Custom threshold, median filter, batch size, device
  python inference_cat_sed_final_updated.py \
      --checkpoint /path/to/best.ckpt \
      --input_folder /path/to/audio/ \
      --output_jsonl results.jsonl \
      --threshold 0.5 --median_window 3 \
      --batch_size 32 --device cuda:1

  # 6. Multi-GPU inference (use run_sed_multigpu.py wrapper)
  python run_sed_multigpu.py \
      --checkpoint /path/to/best.ckpt \
      --input_folder /path/to/audio/ \
      --output_dir /path/to/output/ \
      --gpus 0,1,2,3

SUPPORTED ENCODERS:
  - beats : BEATs (frozen or partially unfrozen) + CNN + Conformer/Zipformer/BiGRU
  - atst  : ATST-Frame + CNN + BiGRU (original architecture)
  - eat   : EAT (Efficient Audio Transformer) + CNN + Conformer/Zipformer/BiGRU

  The encoder type is auto-detected from the checkpoint's saved config
  (encoder_type field).  No manual selection needed.

ARCHITECTURE:
  All encoder types share the same pipeline:
    Raw Audio -> Encoder (frozen) -> frame embeddings [B, N, 768]
    Mel Spec  -> CNN front-end     -> frame features  [B, T, C]
    [CNN, Encoder] -> Linear Fusion -> Temporal Encoder -> strong + weak logits

  Temporal backends: Conformer, Zipformer, or BiGRU (config-selectable)
"""

import sys
import os

# Add train directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
train_dir = os.path.join(parent_dir, "train")
sys.path.insert(0, train_dir)
print(f"Script directory: {script_dir}")
print(f"Train directory: {train_dir}")
print(f"Python path (first 3): {sys.path[:3]}")

import json
import yaml
from tqdm import tqdm
import librosa
import torch
import numpy as np
import scipy.ndimage
from pathlib import Path

from inference_utils import (
    normalize_audio_for_inference,
    find_audio_files,
    load_audio_paths_from_jsonl,
    chunk_audio,
    merge_events_across_chunks,
    load_processed_files,
    append_to_jsonl,
    load_audio_robust,
)

from desed_task.utils.encoder import ManyHotEncoder
from local.classes_dict import get_classes_labels

# MODEL LOADING - FULLY AUTOMATIC FROM CHECKPOINT

def load_sed_model(checkpoint, device, conf_file=None):
    """Load SED model automatically from checkpoint.

    Args:
        checkpoint: Path to model checkpoint (.ckpt file)
        device: torch device
        conf_file: Optional config file (only needed to override data/model paths)

    Returns:
        sed_task: InferenceWrapper with model, mel_spec, scaler
        sed_teacher: The model itself
        config: Full configuration dict
        encoder: ManyHotEncoder for labels
        encoder_type: String indicating encoder type (beats/atst/eat)
        has_temporal: Boolean indicating if model has temporal module
    """
    assert os.path.exists(checkpoint), f"Checkpoint not found: {checkpoint}"

    print(f"\n{'='*80}")
    print("LOADING MODEL FROM CHECKPOINT")
    print(f"{'='*80}")

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Assertions on checkpoint structure
    assert "hyper_parameters" in ckpt, "Checkpoint missing hyper_parameters"
    assert "state_dict" in ckpt, "Checkpoint missing state_dict"
    assert "epoch" in ckpt, "Checkpoint missing epoch"
    assert "global_step" in ckpt, "Checkpoint missing global_step"
    assert ckpt["epoch"] >= 0, f"Invalid epoch: {ckpt['epoch']}"

    # Use config from checkpoint
    config = ckpt["hyper_parameters"]

    # Optional: Override data/model paths if external config provided
    if conf_file is not None:
        assert os.path.exists(conf_file), f"Config file not found: {conf_file}"
        print(f"✓ Overriding paths from: {conf_file}")
        with open(conf_file) as f:
            user_config = yaml.safe_load(f)
        if "data" in user_config:
            config["data"] = user_config["data"]
            print("  - Overrode data paths")
        if "ultra" in user_config:
            config["ultra"].update(user_config["ultra"])
            print("  - Overrode model paths")
    else:
        print("✓ Using configuration from checkpoint (no external config)")

    # Assertions on config structure
    assert "net" in config, "Checkpoint config missing 'net' section"
    assert "feats" in config, "Checkpoint config missing 'feats' section"
    assert "scaler" in config, "Checkpoint config missing 'scaler' section"
    assert "data" in config, "Checkpoint config missing 'data' section"
    assert "encoder_type" in config, "Config missing encoder_type"

    # Extract model metadata
    net_cfg = config["net"]
    encoder_type = config.get("encoder_type", "atst").lower()
    stage_type = net_cfg.get("stage_type", "stage1")
    rnn_type = net_cfg.get("rnn_type", "BGRU")

    # Validate encoder type
    assert encoder_type in ["beats", "atst", "eat"], \
        f"Unknown encoder_type in checkpoint: {encoder_type}. Supported: beats, atst, eat"

    # Net config assertions
    assert "nclass" in net_cfg, "net config missing nclass"
    assert "rnn_type" in net_cfg, "net config missing rnn_type"

    # Resolve class labels from checkpoint config or default
    active_classes_labels = get_classes_labels(config)
    assert net_cfg["nclass"] == len(active_classes_labels), \
        f"Class mismatch: model expects {net_cfg['nclass']}, got {len(active_classes_labels)}"

    # Feature config assertions
    feat_params = config["feats"]
    assert "sample_rate" in feat_params, "feats missing sample_rate"
    assert "n_mels" in feat_params, "feats missing n_mels"
    assert "hop_length" in feat_params, "feats missing hop_length"
    assert "n_window" in feat_params, "feats missing n_window"
    assert feat_params["sample_rate"] == 16000, \
        f"Unexpected sample rate: {feat_params['sample_rate']}"

    print(f"\n✓ Model Metadata:")
    print(f"  Encoder type: {encoder_type.upper()}")
    print(f"  Stage: {stage_type}")
    print(f"  RNN type: {rnn_type}")
    print(f"  Trained epochs: {ckpt['epoch']}")
    print(f"  Global step: {ckpt['global_step']}")
    print(f"  Classes: {net_cfg['nclass']}")

    # Dynamic model import based on encoder type
    print(f"\n✓ Loading {encoder_type.upper()} model architecture...")

    ultra_cfg = config.get("ultra", {})
    opt_cfg = config.get("opt", {})

    if encoder_type == "beats":
        from desed_task.nnet.CRNN_beats import CRNN
        encoder_init_key = "beats_init"
        encoder_attr = "BEATs_model"
    elif encoder_type == "atst":
        from desed_task.nnet.CRNN_e2e import CRNN
        encoder_init_key = "atst_init"
        encoder_attr = "atst_model"
    elif encoder_type == "eat":
        from desed_task.nnet.CRNN_eat import CRNN
        encoder_init_key = "eat_checkpoint_path"
        encoder_attr = "EAT_model"
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}. Supported: beats, atst, eat")

    # Extract architecture params
    unfreeze_atst_layer = opt_cfg.get("tfm_trainable_layers", 0)
    atst_dropout = ultra_cfg.get("atst_dropout", 0.0)

    # Build model kwargs dynamically
    model_kwargs = {
        "unfreeze_atst_layer": unfreeze_atst_layer,
        **net_cfg,
        "model_init": None,  # Don't reinitialize from pretrained
        "atst_dropout": atst_dropout,
    }

    # Add encoder-specific init path
    if encoder_type == "beats":
        model_kwargs["beats_init"] = ultra_cfg.get("beats_init")
    elif encoder_type == "atst":
        model_kwargs["atst_init"] = ultra_cfg.get("atst_init")
    elif encoder_type == "eat":
        model_kwargs["eat_model_dir"] = ultra_cfg.get("eat_model_dir")
        model_kwargs["eat_checkpoint_path"] = ultra_cfg.get("eat_checkpoint_path")
        model_kwargs["eat_norm_mean"] = ultra_cfg.get("eat_norm_mean", -7.8449)
        model_kwargs["eat_norm_std"] = ultra_cfg.get("eat_norm_std", 4.4862)
        model_kwargs["eat_num_mel_bins"] = ultra_cfg.get("eat_num_mel_bins", 128)

    # Instantiate model
    sed_teacher = CRNN(**model_kwargs)

    # Validate model structure
    assert hasattr(sed_teacher, 'cnn'), "Model missing CNN module"
    assert hasattr(sed_teacher, encoder_attr), \
        f"{encoder_type.upper()} model missing {encoder_attr} attribute"

    # Detect architecture: Does it have a temporal module?
    has_temporal = hasattr(sed_teacher, 'temporal')

    if has_temporal:
        print("  ✓ Architecture: Encoder + CNN + Conformer/Transformer + BiGRU")
    else:
        print("  ✓ Architecture: Encoder + CNN + BiGRU (direct)")
        # For simple architecture, validate BiGRU/RNN exists
        assert hasattr(sed_teacher, 'rnn'), "Simple model missing RNN module"

    # Load weights
    print("\n✓ Loading model weights from checkpoint...")
    state_dict = ckpt["state_dict"]
    sd_teacher = {
        k.replace("sed_teacher.", ""): v
        for k, v in state_dict.items()
        if "sed_teacher" in k
    }

    assert len(sd_teacher) > 0, "No sed_teacher weights found in checkpoint"

    missing, unexpected = sed_teacher.load_state_dict(sd_teacher, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys (showing first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  ✓ All weights loaded successfully")

    sed_teacher = sed_teacher.to(device)
    sed_teacher.eval()

    # Load scaler
    print("\n✓ Loading scaler...")
    from desed_task.utils.scaler import TorchScaler

    scaler_config = config["scaler"]
    assert "statistic" in scaler_config, "scaler config missing statistic"
    assert "normtype" in scaler_config, "scaler config missing normtype"
    assert "dims" in scaler_config, "scaler config missing dims"

    scaler = TorchScaler(
        scaler_config["statistic"],
        scaler_config["normtype"],
        scaler_config["dims"],
    )

    # Extract scaler state from checkpoint
    scaler_state = {
        k.replace("scaler.", ""): v
        for k, v in state_dict.items()
        if k.startswith("scaler.")
    }

    if scaler_state:
        print("  ✓ Loading fitted scaler from checkpoint")
        print(f"    Scaler keys: {list(scaler_state.keys())}")
        # Manually register buffers
        for key, value in scaler_state.items():
            scaler.register_buffer(key, value.to(device))
            print(f"    Registered buffer: {key} with shape {value.shape}")
    else:
        # Try loading from savepath
        scaler_path = scaler_config.get("savepath", "./scaler.ckpt")
        if os.path.exists(scaler_path):
            print(f"  ✓ Loading scaler from {scaler_path}")
            saved_scaler = torch.load(
                scaler_path, map_location=device, weights_only=False
            )
            if hasattr(saved_scaler, "mean"):
                scaler.register_buffer("mean", saved_scaler.mean.to(device))
            if hasattr(saved_scaler, "std"):
                scaler.register_buffer("std", saved_scaler.std.to(device))
            if hasattr(saved_scaler, "mean_squared"):
                scaler.register_buffer(
                    "mean_squared", saved_scaler.mean_squared.to(device)
                )
        else:
            raise RuntimeError(
                "No scaler found in checkpoint and savepath doesn't exist: "
                f"{scaler_path}"
            )

    scaler = scaler.to(device)
    scaler.eval()

    # Verify scaler
    assert hasattr(scaler, "mean"), "Scaler must have 'mean' attribute after loading"
    print(
        f"  ✓ Scaler ready: statistic={scaler_config['statistic']}, "
        f"normtype={scaler_config['normtype']}"
    )
    print(f"    mean shape: {scaler.mean.shape}, device: {scaler.mean.device}")

    # Create mel spectrogram transform
    print("\n✓ Creating mel spectrogram transform...")
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    mel_spec = MelSpectrogram(
        sample_rate=feat_params["sample_rate"],
        n_fft=feat_params["n_window"],
        win_length=feat_params["n_window"],
        hop_length=feat_params["hop_length"],
        f_min=feat_params["f_min"],
        f_max=feat_params["f_max"],
        n_mels=feat_params["n_mels"],
        window_fn=torch.hamming_window,
        wkwargs={"periodic": False},
        power=1,
    ).to(device)

    print(f"  Sample rate: {feat_params['sample_rate']} Hz")
    print(f"  N-FFT: {feat_params['n_window']}")
    print(f"  Hop length: {feat_params['hop_length']}")
    print(f"  Mel bins: {feat_params['n_mels']}")

    # take_log function
    def take_log(mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5
        return amp_to_db(mels).clamp(min=-50, max=80)

    # Create wrapper
    class InferenceWrapper:
        def __init__(self, teacher, mel_spec, scaler, take_log_fn, device, encoder_type, has_temporal):
            self.sed_teacher = teacher
            self.mel_spec = mel_spec
            self.scaler = scaler
            self.take_log = take_log_fn
            self.device = device
            self.encoder_type = encoder_type
            self.has_temporal = has_temporal

            assert self.sed_teacher is not None
            assert self.mel_spec is not None
            assert self.scaler is not None
            assert hasattr(self.scaler, "mean"), "Scaler must have 'mean' attribute"

        def eval(self):
            self.sed_teacher.eval()
            self.mel_spec.eval()
            self.scaler.eval()

    sed_task = InferenceWrapper(sed_teacher, mel_spec, scaler, take_log, device, encoder_type, has_temporal)
    sed_task.eval()

    # Create encoder
    encoder = ManyHotEncoder(
        list(active_classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    print(f"\n{'='*80}")
    print("✓ MODEL LOADED SUCCESSFULLY")
    print(f"{'='*80}\n")

    return sed_task, sed_teacher, config, encoder, encoder_type, has_temporal


# ============================================================================
# INFERENCE - ARCHITECTURE-AGNOSTIC (handles both with/without temporal)
# ============================================================================

def run_sed_inference_batch(
    y_chunks,
    sr,
    sed_task,
    sed_teacher,
    config,
    device,
    encoder_type,
    has_temporal,
    threshold=0.5,
    median_window=3,
    debug=False,
    chunk_sec=10.0,
):
    """
    Run SED inference on a batch of chunks (architecture-agnostic).

    Handles BOTH:
      - Simple: Encoder + CNN + BiGRU
      - Advanced: Encoder + CNN + Conformer/Transformer + BiGRU

    Args:
        y_chunks: list of numpy arrays, each of length chunk_sec*sr
        sr: sample rate
        sed_task: InferenceWrapper containing model and transforms
        sed_teacher: The model itself
        config: Configuration dict
        device: torch device
        encoder_type: Type of encoder (beats/atst/eat)
        has_temporal: Boolean, True if model has temporal module
        threshold: Detection threshold
        median_window: Median filter window size
        debug: Enable debug output
        chunk_sec: Chunk duration in seconds

    Returns:
        list of tuples: [(events, weak_score), ...] for each chunk
    """
    batch_size = len(y_chunks)

    # Ensure all chunks are exactly `chunk_sec` seconds
    target_samples = int(chunk_sec * sr)
    processed_chunks = []

    for i, y_chunk in enumerate(y_chunks):
        if len(y_chunk) < target_samples:
            if debug and i == 0:
                print(
                    f"[DEBUG] Padding chunk from {len(y_chunk)} "
                    f"to {target_samples} samples"
                )
            y_chunk = np.pad(
                y_chunk,
                (0, target_samples - len(y_chunk)),
                mode="constant",
                constant_values=0,
            )
        elif len(y_chunk) > target_samples:
            if debug and i == 0:
                print(
                    f"[DEBUG] Truncating chunk from {len(y_chunk)} "
                    f"to {target_samples} samples"
                )
            y_chunk = y_chunk[:target_samples]

        assert (
            len(y_chunk) == target_samples
        ), f"Chunk {i} must be exactly {target_samples} samples, got {len(y_chunk)}"
        processed_chunks.append(y_chunk)

    if debug:
        print(f"\n[DEBUG] Encoder type: {encoder_type}")
        print(f"[DEBUG] Architecture: {'With Temporal Module' if has_temporal else 'Direct CNN->RNN'}")
        print(f"[DEBUG] Batch size: {batch_size}")
        print(f"[DEBUG] Input chunk shape: {processed_chunks[0].shape}, sr: {sr}")
        print(f"[DEBUG] Chunk duration: {len(processed_chunks[0])/sr:.3f}s")

    # Prepare batch audio tensor [B, samples]
    audio = torch.stack([
        torch.tensor(chunk, dtype=torch.float32) for chunk in processed_chunks
    ]).to(device)

    assert audio.ndim == 2, f"Audio should be [B, samples], got {audio.shape}"
    assert (
        audio.shape[1] == target_samples
    ), f"Audio should have {target_samples} samples, got {audio.shape[1]}"
    assert not torch.isnan(audio).any(), "NaN in input audio"

    if debug:
        print(f"[DEBUG] Audio tensor shape: {audio.shape}")

    with torch.no_grad():
        # Mel -> log-mel -> scaler
        mel = sed_task.mel_spec(audio)
        assert mel.ndim == 3, f"Mel should be [B, n_mels, T], got {mel.shape}"
        assert mel.shape[0] == batch_size, f"Mel batch mismatch: {mel.shape[0]} vs {batch_size}"
        assert not torch.isnan(mel).any(), "NaN in mel spectrogram"

        if debug:
            print(f"[DEBUG] Mel spectrogram shape: {mel.shape}")

        log_mel = sed_task.take_log(mel)
        assert not torch.isnan(log_mel).any(), "NaN in log-mel"

        norm_mel = sed_task.scaler(log_mel)
        assert not torch.isnan(norm_mel).any(), "NaN in normalized mel"

        if debug:
            print(f"[DEBUG] Normalized mel shape: {norm_mel.shape}")

        # Get embeddings based on encoder type
        if encoder_type == "beats":
            embeddings = sed_teacher.BEATs_model(audio)
        elif encoder_type == "atst":
            embeddings = sed_teacher.atst_model(audio)
        elif encoder_type == "eat":
            embeddings = sed_teacher.EAT_model(audio)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        assert embeddings.ndim == 3, (
            f"Embeddings should be [B, patches/time, dim], "
            f"got {embeddings.shape}"
        )
        assert embeddings.shape[0] == batch_size, f"Embeddings batch mismatch: {embeddings.shape[0]} vs {batch_size}"
        assert not torch.isnan(embeddings).any(), "NaN in embeddings"

        if debug:
            print(f"[DEBUG] {encoder_type.upper()} embeddings shape: {embeddings.shape}")

        # CNN features
        x = norm_mel.transpose(1, 2).unsqueeze(1)  # [B, 1, T, n_mels]
        assert x.ndim == 4, f"CNN input should be [B, 1, T, n_mels], got {x.shape}"

        x = sed_teacher.cnn(x)
        bs, chan, frames, freq = x.size()
        assert freq == 1, f"CNN output freq should be 1, got {freq}"
        assert not torch.isnan(x).any(), "NaN after CNN"

        x = x.squeeze(-1).permute(0, 2, 1)  # [B, T, C]

        if debug:
            print(f"[DEBUG] CNN output shape: {x.shape}")

        # Align embeddings to CNN frames
        embeddings = torch.nn.functional.adaptive_avg_pool1d(
            embeddings.transpose(-1, -2), x.shape[1]
        ).transpose(-1, -2)

        assert embeddings.shape[1] == x.shape[1], "Embeddings alignment failed"
        assert not torch.isnan(embeddings).any(), "NaN in aligned embeddings"

        if debug:
            print(f"[DEBUG] Aligned embeddings shape: {embeddings.shape}")

        # Fusion
        x_fused = torch.cat((x, embeddings), dim=-1)
        x_fused = sed_teacher.cat_tf(x_fused)
        assert not torch.isnan(x_fused).any(), "NaN in fused features"

        if debug:
            print(f"[DEBUG] Fused features shape: {x_fused.shape}")

        # Architecture-specific path
        if has_temporal:
            # Advanced: Conformer/Transformer + BiGRU
            # temporal module returns (strong, weak) logits directly
            strong_logits, weak_logits = sed_teacher.temporal(x_fused)

            if debug:
                print(f"[DEBUG] Using temporal module (Conformer/Transformer)")
                print(f"[DEBUG] Strong logits shape: {strong_logits.shape}")
                print(f"[DEBUG] Weak logits shape: {weak_logits.shape}")
        else:
            # Simple: Direct BiGRU (CRNN_e2e path)
            # Mirrors CRNN_e2e.forward(): RNN -> dropout -> dense -> attention-weighted weak
            x_rnn = sed_teacher.rnn(x_fused)
            x_rnn = sed_teacher.dropout(x_rnn)

            # Get logits (before sigmoid)
            strong_logits = sed_teacher.dense(x_rnn)  # [B, T, C]
            strong_logits = strong_logits.transpose(1, 2)  # [B, C, T]

            # Weak prediction via attention-weighted pooling
            # CRNN_e2e always uses dense_softmax attention (no conditional)
            sof = sed_teacher.dense_softmax(x_rnn)  # [B, T, C]
            sof = sed_teacher.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            strong_probs_for_weak = torch.sigmoid(strong_logits.transpose(1, 2))
            weak_probs = (strong_probs_for_weak * sof).sum(1) / sof.sum(1)  # [B, C]
            # Convert back to logits (inverse sigmoid)
            eps = 1e-7
            weak_probs = weak_probs.float().clamp(eps, 1.0 - eps)
            weak_logits = torch.log(weak_probs) - torch.log1p(-weak_probs)

            if debug:
                print(f"[DEBUG] Using direct RNN path (no temporal module)")
                print(f"[DEBUG] Strong logits shape: {strong_logits.shape}")
                print(f"[DEBUG] Weak logits shape: {weak_logits.shape}")

        # Strong logits are [B, C, T], weak logits are [B, C]
        assert strong_logits.shape[0] == batch_size, f"Strong logits batch mismatch"
        assert weak_logits.shape[0] == batch_size, f"Weak logits batch mismatch"
        assert strong_logits.shape[1] == config["net"]["nclass"], (
            f"Class mismatch: {strong_logits.shape[1]} vs {config['net']['nclass']}"
        )
        assert not torch.isnan(strong_logits).any(), "NaN in strong logits"
        assert not torch.isnan(weak_logits).any(), "NaN in weak logits"

        # Convert to probabilities
        strong_probs = torch.sigmoid(strong_logits).cpu().numpy()  # [B, C, T]
        weak_probs = torch.sigmoid(weak_logits).cpu().numpy()  # [B, C]

        if debug:
            print(f"[DEBUG] Strong probs shape: {strong_probs.shape}")
            print(f"[DEBUG] Weak probs shape: {weak_probs.shape}")

    # Process each chunk in the batch
    results = []
    nclass = strong_probs.shape[1]

    # Resolve class names from config (for multi-label event labelling)
    class_names_list = config.get("data", {}).get("class_names", None)
    if class_names_list is None:
        class_names_list = list(get_classes_labels(config).keys())

    for b in range(batch_size):
        all_events = []
        weak_scores = {}

        for c in range(nclass):
            class_name = class_names_list[c] if c < len(class_names_list) else f"class_{c}"
            frame_probs = strong_probs[b, c, :]  # [T]
            weak_score = float(weak_probs[b, c])
            weak_scores[class_name] = weak_score

            assert frame_probs.ndim == 1, (
                f"Frame probs should be 1D, got {frame_probs.shape}"
            )
            assert not np.isnan(frame_probs).any(), "NaN in frame probabilities"

            if debug and b == 0 and c == 0:
                print(f"[DEBUG] Frame probs shape: {frame_probs.shape}")
                print(
                    "[DEBUG] Frame probs range: "
                    f"[{frame_probs.min():.3f}, {frame_probs.max():.3f}]"
                )

            # Median filter
            if median_window > 1:
                win = median_window if median_window % 2 == 1 else median_window + 1
                filtered_probs = scipy.ndimage.median_filter(
                    frame_probs, size=win, mode="reflect"
                )
                assert filtered_probs.shape == frame_probs.shape, (
                    "Median filter changed shape"
                )
                assert not np.isnan(filtered_probs).any(), (
                    "NaN after median filtering"
                )
            else:
                filtered_probs = frame_probs.copy()

            pred_binary = (filtered_probs >= threshold).astype(np.float32)

            if debug and b == 0 and c == 0:
                print(
                    f"[DEBUG] Positive frames: "
                    f"{pred_binary.sum()}/{len(pred_binary)}"
                )

            # Event extraction - frame rate consistent with encoder/training
            frame_rate = (
                config["data"]["fs"]
                / config["feats"]["hop_length"]
                / config["data"]["net_subsample"]
            )

            if debug and b == 0 and c == 0:
                print(f"[DEBUG] Frame rate: {frame_rate:.2f} fps")

            events = []
            in_event = False
            onset = None

            for i, val in enumerate(pred_binary):
                t = i / frame_rate
                if val > 0 and not in_event:
                    onset = t
                    in_event = True
                elif val == 0 and in_event:
                    offset = t
                    onset_frame = int(onset * frame_rate)
                    offset_frame = int(offset * frame_rate)
                    if offset_frame > onset_frame:
                        score = float(
                            frame_probs[onset_frame:offset_frame].mean()
                        )
                    else:
                        score = float(frame_probs[onset_frame])
                    events.append(
                        {"onset": onset, "offset": offset, "score": score, "event_label": class_name}
                    )
                    in_event = False

            # Handle event extending to end of chunk
            if in_event:
                offset = len(pred_binary) / frame_rate
                onset_frame = int(onset * frame_rate)
                score = float(frame_probs[onset_frame:].mean())
                events.append({"onset": onset, "offset": offset, "score": score, "event_label": class_name})

            all_events.extend(events)

            if debug and b == 0:
                print(f"[DEBUG] Class '{class_name}': {len(events)} events, weak={weak_score:.4f}")

        # For single-class backward compat: return scalar weak_score
        if nclass == 1:
            result_weak = weak_scores[class_names_list[0]]
        else:
            result_weak = weak_scores

        results.append((all_events, result_weak))

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cat SED Inference - Automatic & Architecture-Agnostic"
    )

    parser.add_argument(
        "--input_folder", default=None, help="Input audio folder (recursive)"
    )
    parser.add_argument(
        "--input_jsonl", default=None,
        help="Input JSONL file with audio paths"
    )
    parser.add_argument(
        "--output_jsonl", required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--conf_file", default=None,
        help="Optional config YAML file (only needed to override data/model paths)"
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=10.0,
        help="Chunk length in seconds (must be 10.0 for this model)",
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Sample rate"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.64,
        help="Frame-level detection threshold",
    )
    parser.add_argument(
        "--median_window",
        type=int,
        default=1,
        help="Median filter window (frames)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.0,
        help="Minimum event duration (seconds)",
    )
    parser.add_argument(
        "--merge_gap",
        type=float,
        default=0.064,
        help="Merge events closer than this (seconds)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save to JSONL every N files",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device (cuda:0 or cpu)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply loudness normalization (same as training)",
    )
    parser.add_argument(
        "--target_lufs",
        type=float,
        default=-23.0,
        help="Target loudness in LUFS",
    )
    parser.add_argument(
        "--force_normalize",
        action="store_true",
        help="Force normalization even if already normalized",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of chunks to process in parallel",
    )

    args = parser.parse_args()

    # Check that either input_folder or input_jsonl is provided
    if not args.input_folder and not args.input_jsonl:
        parser.error("Must provide either --input_folder or --input_jsonl")
    if args.input_folder and args.input_jsonl:
        parser.error("Cannot provide both --input_folder and --input_jsonl")

    # Enforce 10s chunks, as per training
    if abs(args.chunk_sec - 10.0) > 1e-6:
        print(
            f"⚠️  chunk_sec={args.chunk_sec} but model was trained on 10s chunks."
        )
        print("   Overriding to 10.0 seconds to match training.")
        args.chunk_sec = 10.0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model automatically from checkpoint
    sed_task, sed_teacher, config, encoder, encoder_type, has_temporal = load_sed_model(
        args.checkpoint, device, conf_file=args.conf_file
    )

    if args.normalize:
        print(f"✓ Audio normalization enabled: {args.target_lufs} LUFS")
        print(f"  Force normalization: {args.force_normalize}")
    else:
        print("⚠️  Audio normalization DISABLED - ensure input matches training!")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load already processed files
    processed_files = load_processed_files(args.output_jsonl)

    # Discover audio files
    if args.input_folder:
        print(f"\nScanning {args.input_folder} for audio files...")
        all_files = list(find_audio_files(args.input_folder))
        # Convert to (basename, path) tuples for consistency
        all_files = [(os.path.splitext(os.path.basename(f))[0], f) for f in all_files]
        print(f"Found {len(all_files)} audio files")
    else:
        print(f"\nLoading audio paths from {args.input_jsonl}...")
        all_files = load_audio_paths_from_jsonl(args.input_jsonl)
        print(f"Found {len(all_files)} audio files with non-empty clips")

    # Filter out already processed
    files_to_process = [
        (basename, path)
        for basename, path in all_files
        if basename not in processed_files
    ]

    print(
        f"Files to process: {len(files_to_process)} "
        f"(skipping {len(all_files) - len(files_to_process)} already processed)\n"
    )

    assert len(all_files) > 0, f"No audio files found"

    # Process files with incremental saving
    total_events = 0
    is_first_file = True
    batch_records = []
    processed_count = 0
    error_count = 0
    normalized_count = 0
    already_normalized_count = 0

    for file_idx, (basename, file_path) in enumerate(
        tqdm(files_to_process, desc="Processing files")
    ):
        temp_file_created = False
        normalized_path = file_path

        try:
            # Normalize audio if requested
            if args.normalize:
                try:
                    (
                        normalized_path,
                        temp_file_created,
                    ) = normalize_audio_for_inference(
                        file_path,
                        target_lufs=args.target_lufs,
                        tolerance=3.0,
                        force_normalize=args.force_normalize,
                    )
                    if temp_file_created:
                        normalized_count += 1
                        if is_first_file:
                            print(f"\n[DEBUG] Normalized {file_path}")
                            print(f"[DEBUG] Temp file: {normalized_path}")
                    else:
                        already_normalized_count += 1
                        if is_first_file:
                            print(
                                f"\n[DEBUG] Audio already normalized: {file_path}"
                            )
                except Exception as norm_error:
                    print(
                        f"\n⚠️  Normalization failed for {file_path}: {norm_error}"
                    )
                    print("   Attempting to process without normalization...")
                    normalized_path = file_path
                    temp_file_created = False

            # Load audio (from normalized path if normalization was applied)
            y, sr = load_audio_robust(normalized_path, target_sr=args.sr, mono=True)
            # Skip very short files
            if len(y) < int(0.1 * sr):
                if is_first_file:
                    print(
                        f"\n⚠️  Skipping {file_path}: too short "
                        f"({len(y)} samples, {len(y)/sr:.3f}s)"
                    )
                error_count += 1
                continue

            assert len(y) > 0, f"Empty audio file: {file_path}"
            assert sr == args.sr, f"Sample rate mismatch: {sr} != {args.sr}"

            audio_duration = len(y) / sr

            if is_first_file:
                print(f"\n[DEBUG] First file: {file_path}")
                print(
                    f"[DEBUG] Audio shape: {y.shape}, sr: {sr}, "
                    f"duration: {audio_duration:.3f}s"
                )

            # Chunk audio
            chunks = chunk_audio(y, sr, args.chunk_sec)
            assert len(chunks) > 0, f"No chunks created for {file_path}"

            if is_first_file:
                print(f"[DEBUG] Created {len(chunks)} chunks")

            # Accumulate events from all chunks
            all_chunk_events = []
            all_weak_scores = []  # Collect weak scores from all chunks

            # Process chunks in batches
            for batch_start in range(0, len(chunks), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                # Extract just the audio arrays for batch processing
                y_batch = [y_chunk for y_chunk, t0, t1 in batch_chunks]

                # Run inference on batch
                debug_mode = is_first_file and batch_start == 0
                batch_results = run_sed_inference_batch(
                    y_batch,
                    sr,
                    sed_task,
                    sed_teacher,
                    config,
                    device,
                    encoder_type,
                    has_temporal,
                    threshold=args.threshold,
                    median_window=args.median_window,
                    debug=debug_mode,
                    chunk_sec=args.chunk_sec,
                )

                # Unpack results for each chunk in batch
                for idx, (events, weak_score) in enumerate(batch_results):
                    chunk_idx = batch_start + idx
                    y_chunk, t0, t1 = batch_chunks[idx]
                    all_weak_scores.append(weak_score)

                    # Filter by min_duration
                    if args.min_duration > 0:
                        events = [
                            e
                            for e in events
                            if (e["offset"] - e["onset"]) >= args.min_duration
                        ]

                    # Add events with absolute timestamps
                    for event in events:
                        assert event["onset"] >= 0
                        assert event["offset"] > event["onset"]
                        assert 0 <= event["score"] <= 1
                        ev_dict = {
                            "onset": event["onset"] + t0,
                            "offset": event["offset"] + t0,
                            "score": event["score"],
                            "duration": event["offset"] - event["onset"],
                        }
                        if "event_label" in event:
                            ev_dict["event_label"] = event["event_label"]
                        all_chunk_events.append(ev_dict)

            # Merge events across chunks
            if all_chunk_events:
                merged_events = merge_events_across_chunks(
                    all_chunk_events, min_gap=args.merge_gap
                )
            else:
                merged_events = []

            # Clip event offsets to actual audio duration (padding creates spurious detections)
            clipped_events = []
            for ev in merged_events:
                onset = ev["onset"]
                offset = min(ev["offset"], audio_duration)

                # Skip events that start after audio ends (from padding)
                if onset >= audio_duration:
                    continue

                # Skip events that become too short after clipping
                if offset - onset < 0.01:
                    continue

                clipped_events.append({
                    "onset": onset,
                    "offset": offset,
                    "score": ev["score"],
                    "duration": offset - onset,
                    **({"event_label": ev["event_label"]} if "event_label" in ev else {}),
                })

            merged_events = clipped_events
            total_events += len(merged_events)

            # Aggregate weak scores across chunks (use max for clip-level confidence)
            if all_weak_scores and isinstance(all_weak_scores[0], dict):
                # Multi-label: aggregate per class
                file_weak_score = {}
                for key in all_weak_scores[0]:
                    file_weak_score[key] = max(ws[key] for ws in all_weak_scores)
            else:
                file_weak_score = max(all_weak_scores) if all_weak_scores else 0.0

            # Create record for this file
            file_record = {
                "filename": basename,
                "file_path": os.path.abspath(file_path),
                "duration": audio_duration,
                "sample_rate": sr,
                "num_chunks": len(chunks),
                "num_events": len(merged_events),
                "weak_score": file_weak_score,  # Clip-level cat detection confidence
                "events": merged_events,
            }

            batch_records.append(file_record)
            processed_count += 1

            # Incremental save
            if processed_count % args.save_every == 0:
                append_to_jsonl(args.output_jsonl, batch_records)
                print(
                    f"\n✓ Saved {len(batch_records)} records "
                    f"(total processed: {processed_count}, "
                    f"errors: {error_count})"
                )
                batch_records = []

            is_first_file = False

        except Exception as e:
            if is_first_file or error_count < 10:  # Show first 10 errors
                print(f"\n⚠️  Error processing {file_path}: {e}")
                import traceback
                if is_first_file:
                    traceback.print_exc()
            error_count += 1
            continue

        finally:
            # Clean up temp file if created
            if (
                temp_file_created
                and normalized_path
                and os.path.exists(normalized_path)
            ):
                try:
                    os.remove(normalized_path)
                except Exception as cleanup_error:
                    if is_first_file:
                        print(
                            f"[DEBUG] Could not remove temp file "
                            f"{normalized_path}: {cleanup_error}"
                        )

    # Save remaining
    if batch_records:
        append_to_jsonl(args.output_jsonl, batch_records)
        print(f"\n✓ Saved final {len(batch_records)} records")

    print(f"\n{'='*80}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Files processed: {processed_count}")
    if args.normalize:
        print(f"Files normalized: {normalized_count}")
        print(f"Files already normalized (skipped): {already_normalized_count}")
    print(f"Files with errors (skipped): {error_count}")
    print(f"Total events detected: {total_events}")
    print(f"\nOutput JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()