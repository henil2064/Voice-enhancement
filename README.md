# Voice Enhancement Pipeline: Noise Suppression, Speaker Separation and Hindi ASR

`voice_enhancement .ipynb` is a Kaggle notebook that benchmarks a speech front-end for noisy, multi-speaker Indian-language audio. It builds synthetic test sets from public datasets, runs them through noise suppression (DeepFilterNet3), gain control and two-speaker separation (Conv-TasNet), and scores each variant by transcribing it with **SraVaani** (a Parakeet-based Indic ASR model) and computing WER/CER.

At the end it provides a reusable `process_pipeline()` function:

```
input audio ─► DeepFilterNet3 denoise ─► AGC amplify ─► Conv-TasNet separate ─► SraVaani transcribe (per speaker)
```

---

## What the notebook does

| Cells | Stage | Details |
|---|---|---|
| 0–2 | Environment setup | Installs Rust (needed to build DeepFilterNet), then `deepfilternet`, `asteroid`, `datasets`, `jiwer`, `pystoi`, `silero-vad`, etc. Pins `numpy<2.1`. |
| 3 | Imports + compatibility shim | Patches the removed `torchaudio.backend.common.AudioMetaData` so DeepFilterNet imports on new torchaudio (2.10). |
| 4 | Config | Language, clip counts, test SNRs, output paths, and the denoising variants to compare. |
| 5 | Clean speech | Streams `N_SPEECH` clips + transcripts from **Google FLEURS** (`test` split) for `LANG`. |
| 6 | Real-world noise | Streams `N_NOISE` clips from **ARTPARK-IISc/Vaani-Noise-Event-Dataset**, filtered to human non-speech, phone/alarm, appliance/machine and baby/child noise. |
| 7 | Noisy test set | Mixes each clean clip with 60% babble (3 other speakers) + 40% ambient noise at **15, 5 and 0 dB SNR**, giving 45 clips. |
| 8–10 | Models & helpers | Loads DeepFilterNet3 (48 kHz), and defines resampling, `dfn3_pass`, `agc_pass` (RMS gain control) and `si_sdr`. Cells 8 and 9 are duplicates, so one can be removed. |
| 11–12 | Denoising run | Processes every clip with each variant, logs the real-time factor (RTF), and shows spectrograms + audio players for one clip. |
| 13 | ASR setup | Downloads the `parakeet.cpp` v0.5.0 CPU binary and `Henil1/Sravaani-1.0-GGUF` (`sravaani-q8_0.gguf`, ~660 MB), and defines text normalisation and WER helpers, including permutation-invariant WER for two speakers. |
| 14 | **Experiment 1: denoising vs. WER** | Transcribes all 4 variants × 45 clips and reports mean WER/CER. |
| 15–17 | **Experiment 2: overlapping speakers** | Sums pairs of FLEURS clips (7 mixtures), separates them with `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`, and compares WER before and after separation. |
| 18–21 | **Experiment 3: noise + overlap** | Adds Vaani noise at 8 dB to the overlapping mixtures and compares three orderings: denoise→separate, separate→denoise, denoise→amplify→separate. Prints a worked example with ground truth. |
| 22–23 | **Production pipeline** | `process_pipeline(audio_path)` returns per-speaker transcripts, per-stage timings and RTF. |
| 24–26 | **Experiment 4: loudness matching** | Rebuilds the overlap mixtures with both speakers RMS-matched to test whether loudness imbalance explains separation failures. |

---

## Results (Hindi, `hi_in`; recorded run)

### 1. Noise suppression only (45 clips, SNR 15/5/0 dB)

| Variant | `atten_lim_db` | WER | CER | Mean RTF |
|---|---|---|---|---|
| raw (no processing) | – | 0.496 | 0.459 | – |
| dfn3_full | unlimited | 0.511 | 0.453 | 0.052 |
| **dfn3_atten12** | 12 dB | **0.432** | **0.385** | 0.008 |
| dfn3_atten6 | 6 dB | 0.448 | 0.404 | 0.008 |

**Takeaway:** use a limited attenuation (12 dB). Full-strength suppression removes speech cues and is slightly *worse* than doing nothing for ASR.

### 2. Overlapping speakers (7 two-speaker mixtures)

| Condition | Mean WER | Median WER |
|---|---|---|
| Raw mixture | 0.713 | 0.782 |
| **Separated (Conv-TasNet)** | **0.401** | **0.361** |
| Separated, loudness-matched inputs | 0.418 | – |

Loudness matching did not help on average, so speaker loudness imbalance is not the main cause of the remaining errors.

### 3. Noise + overlapping speakers (8 dB SNR)

| Pipeline | Mean WER | Median WER |
|---|---|---|
| Raw | 0.927 | 0.987 |
| denoise → separate | 0.716 | 0.765 |
| separate → denoise | 0.716 | 0.800 |
| **denoise → amplify → separate** | **0.703** | 0.791 |

All three orderings perform about the same, and each is roughly 22 WER points better than raw.

### 4. End-to-end latency (`process_pipeline`, 9.12 s clip)

| Stage | Time |
|---|---|
| Denoise | 0.09 s |
| Amplify | 0.00 s |
| Separate | 0.11 s |
| Transcribe (2 streams, CPU) | 4.48 s |
| **Total** | **4.68 s (RTF 0.51)** |

Transcription is the bottleneck. The enhancement stages together take about 2% of total runtime.

---

## Requirements

- **Platform:** Kaggle notebook with a GPU (the run used CUDA). Paths are hard-coded to `/kaggle/working/`.
- **Internet:** must be enabled. The notebook downloads rustup, pip packages, HF datasets/models and a GitHub release.
- **Hugging Face token:** add a Kaggle secret named `HF_TOKEN`. Accept the terms of the gated **Vaani Noise Event Dataset** (and FLEURS if prompted) on huggingface.co first.
- **Python:** 3.12, torch/torchaudio 2.10, `numpy>=1.26,<2.1`.

## How to run

1. Upload the notebook to Kaggle, enable **GPU** and **Internet**, and add the `HF_TOKEN` secret.
2. Optionally edit the config in cell 4:
   ```python
   LANG = "hi_in"        # any FLEURS code: ta_in, te_in, bn_in, mr_in, kn_in, ml_in, gu_in, pa_in, ...
   N_SPEECH = 15         # clean FLEURS clips
   N_NOISE  = 10         # Vaani noise clips
   TEST_SNRS = [15, 5, 0]
   VARIANTS = {"raw": "raw", "dfn3_full": None, "dfn3_atten12": 12, "dfn3_atten6": 6}
   ```
3. Run all cells top to bottom. Cell 14 takes about 8–10 minutes (180 CPU transcriptions).
4. To process your own recording, run the setup cells (0–13 and 16), then:
   ```python
   result = process_pipeline("/kaggle/input/my-data/meeting.wav")
   result["transcripts"]   # list of 2 strings, one per speaker stream
   result["timings"]       # per-stage seconds + RTF
   ```

## Output layout (`/kaggle/working/`)

```
dummy/
  clean/                      FLEURS clean clips
  noise_pool/                 Vaani noise clips
  noisy_input/                45 babble+noise mixtures (+ refs_noisy.csv)
  overlap_input/              7 two-speaker mixtures (+ overlap_pairs.csv)
  overlap_input_normalized/   loudness-matched mixtures
  separated/, separated_normalized/
  combo_input/                overlap + noise mixtures
  combo_out/<pipeline>/       per-pipeline separated streams
out/<variant>/                denoised 16 kHz PCM-16 outputs for Experiment 1
parakeet_bin/                 parakeet.cpp CLI
```

## Models & data

| Component | Source |
|---|---|
| Noise suppression | [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) |
| Speaker separation | [`JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k) via Asteroid |
| ASR | [`Henil1/Sravaani-1.0-GGUF`](https://huggingface.co/Henil1/Sravaani-1.0-GGUF) run with [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (TDT decoder, CPU) |
| Clean speech | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) |
| Noise | [`ARTPARK-IISc/Vaani-Noise-Event-Dataset`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) |

## Known issues and limitations

- **Separator is English-trained.** Conv-TasNet was trained on LibriMix (English, clean). On Hindi it often leaves one stream nearly empty. The ASR then hallucinates on that stream; in the worked example (cell 21) it outputs a repeated **Telugu** sentence. Fine-tuning on Indic mixtures, or gating streams with VAD (`silero-vad` is already installed but unused), would help.
- **Small sample.** There are 15 source clips and 7 overlap mixtures, so the differences between pipelines in Experiment 3 (about 1 WER point) are within noise.
- **Always 2 streams.** The separator assumes exactly two speakers, even for single-speaker input.
- **Dependency conflicts.** `deepfilternet 0.5.6` declares `numpy<2.0`, and pip warns about this. It works in practice with the pinned `numpy<2.1`.
- **Leftovers.** Cells 8 and 9 are duplicates. `MED_TERMS` in cell 14 is defined but not used. `si_sdr` and `pystoi` are imported but not reported. The notebook filename has a trailing space (`voice_enhancement .ipynb`), which you may want to rename.
