# Gravitational Anomaly Detector

A Python-based machine-learning pipeline that detects unusual patterns in LIGO gravitational-wave strain data using a convolutional autoencoder and validates candidates across the Livingston and Hanford detectors.

Rather than treating every unusual reconstruction as a detection, the system progressively filters candidates using detector-specific anomaly thresholds, H1/L1 coincidence, fine-time correlation, physical propagation constraints, and empirical background comparison. The project combines simulated Advanced LIGO noise, real GWOSC strain data, PyTorch anomaly detection, PyCBC signal processing, and physical interpretation.

## Demo

![Gravitational Anomaly Detector Demo](assets/demo.gif)

Example result:

```text
Initial L1 anomalies: 5
Cross-detector candidates: 1
Surviving candidates: 0

No candidates survived validation.
```

---

## Overview

Gravitational Anomaly Detector is an experimental anomaly-detection system designed to identify unusual structure in public LIGO interferometer strain data.

The system uses a one-dimensional convolutional autoencoder trained first on simulated Advanced LIGO background noise and then on quality-filtered real Livingston strain. During testing, windows that reconstruct poorly are identified as initial anomalies because they differ from the background patterns learned by the model.

An autoencoder anomaly alone is not treated as evidence of a gravitational signal. Livingston candidates are cross-checked against independent Hanford measurements, analyzed at finer temporal resolution, constrained by the physical propagation time between detectors, and compared against correlations occurring naturally in quality-filtered background data.

The goal of the project is to explore how machine learning, signal processing, detector coincidence, statistical screening, and physical constraints can be combined into a conservative anomaly-detection pipeline for real scientific data.

## Features

- **Convolutional autoencoder** — Learns background strain structure and detects unusual windows through reconstruction error.
- **Hybrid training** — Trains first on simulated Advanced LIGO noise and then continues training on quality-filtered real LIGO data.
- **Real GWOSC data** — Processes public calibrated strain from the LIGO Livingston and Hanford detectors.
- **Quality filtering** — Removes periods that fail available detector-quality requirements or contain deliberate hardware injections.
- **Detector-specific thresholds** — Calculates independent reconstruction-error thresholds for L1 and H1 background data.
- **Cross-detector validation** — Requires Livingston anomalies to have independently anomalous Hanford data nearby.
- **Fine-time correlation** — Searches relative H1/L1 time shifts to locate the strongest conditioned-strain correlation.
- **Physical delay screening** — Rejects candidates whose detector delay exceeds the approximate physical H1/L1 propagation range.
- **Empirical background testing** — Compares candidate correlation against quality-passing detector background.
- **Physical interpretation** — Converts calibrated strain into an equivalent LIGO differential arm-length perturbation for validated candidates.

## How It Works

The application processes LIGO strain through several stages:

1. **Generate simulated background**  
   PyCBC generates noise based on the Advanced LIGO design sensitivity power spectral density.

2. **Filter real LIGO data**  
   GWOSC HDF5 files are inspected using data-quality and hardware-injection masks to retain suitable detector background.

3. **Train on simulated strain**  
   A one-dimensional convolutional autoencoder learns to reconstruct simulated LIGO-like background.

4. **Fine-tune on real strain**  
   The same model continues training on quality-filtered real Livingston data.

5. **Calculate detector thresholds**  
   Reconstruction errors across L1 and H1 background data are used to establish independent 99th-percentile anomaly thresholds.

6. **Scan unseen Livingston data**  
   Previously unseen one-second windows exceeding the L1 reconstruction-error threshold become initial candidates.

7. **Cross-check Hanford data**  
   Each L1 candidate is checked for independently anomalous H1 strain in the same or adjacent GPS second.

8. **Perform fine-time correlation**  
   Surviving candidates are band-limited and compared across relative detector time shifts.

9. **Apply physical and statistical validation**  
   Candidates must have a physically plausible H1/L1 delay and correlation sufficiently unusual compared with quality-passing background.

10. **Characterize surviving candidates**  
    Candidates passing every validation stage proceed to calibrated strain and physical analysis.

## Architecture

The project separates machine-learning anomaly detection from cross-detector validation and physical interpretation.

The processing pipeline is:

**GWOSC Strain → Quality Filtering → Hybrid Autoencoder Training → Reconstruction-Error Screening → H1/L1 Coincidence → Fine-Time Correlation → Empirical Background Test → Classification → Physical Analysis**

The autoencoder determines whether a strain window is unusual relative to learned background behavior. Separate validation stages then determine whether that anomaly is independently supported by both detectors and unusual compared with ordinary cross-detector background.

This prevents a high reconstruction error from automatically being interpreted as evidence of a gravitational event.

### Autoencoder

Each model input contains one second of strain sampled at 4096 Hz.

The encoder uses three one-dimensional convolutional layers:

```text
4096 samples
    ↓
Conv1d: 1 → 16
    ↓
Conv1d: 16 → 32
    ↓
Conv1d: 32 → 64
```

The decoder mirrors the encoder using transposed convolutions:

```text
64 → 32 → 16 → 1
```

Each strain window is independently standardized before entering the model. Training minimizes mean squared reconstruction error using the Adam optimizer.

A window that differs substantially from learned background structure should be more difficult to reconstruct and therefore produce a larger reconstruction error.

### Hybrid Training

Training occurs in two stages.

The first stage uses simulated noise generated from the Advanced LIGO design sensitivity curve. This gives the autoencoder an initial representation of LIGO-like detector background.

The same model then continues training on quality-filtered real Livingston strain. This adapts the initial representation to characteristics present in real interferometer data.

Example training progression:

```text
Stage 1: simulated background training
Epoch 1/10 - reconstruction loss: 0.842634
...
Epoch 10/10 - reconstruction loss: 0.193096

Stage 2: real LIGO background training
Epoch 1/5 - reconstruction loss: 0.102629
...
Epoch 5/5 - reconstruction loss: 0.014295
```

### Cross-Detector Validation

Livingston and Hanford are independent detectors, so unusual behavior in only one instrument is not sufficient for a candidate to continue.

The system calculates independent anomaly thresholds for each detector and checks every L1 anomaly against the same and adjacent H1 seconds.

Candidates appearing in both detectors undergo fine-time analysis. Two seconds of strain are band-limited to approximately 30–500 Hz and compared across relative time shifts.

The strongest correlation must occur within the approximate ±10 ms physical propagation range between Hanford and Livingston.

### Background Comparison

A physically plausible correlation can still occur naturally in detector background.

The system therefore selects up to 200 quality-passing two-second H1/L1 background regions and processes them using the same correlation procedure as the candidate.

Candidate correlation strength is compared against this empirical background distribution:

```text
p = (background correlations >= candidate + 1)
    / (background comparisons + 1)
```

The current project screening criterion is `p < 0.05`.

This value is an empirical project-level screening statistic and is not equivalent to the formal false-alarm rate or astrophysical significance used by LIGO.

### Physical Analysis

Candidates surviving validation are analyzed using calibrated strain that has not been standardized.

LIGO strain is related to differential interferometer arm length by:

```text
h = ΔL / L
```

Using LIGO's approximately 4 km arms, the system calculates:

```text
ΔL = h × 4000 m
```

The physics module also calculates gravity-only time-dilation reference values at sea level, ISS altitude, low Earth orbit, GPS altitude, and geostationary altitude using the Schwarzschild metric.

These time-dilation values are general-relativistic reference calculations and are not interpreted as effects caused by a detected strain anomaly.

## Results

The pipeline was tested on previously unseen L1 and H1 strain beginning at GPS time `1126076416`.

Five Livingston windows initially exceeded the L1 anomaly threshold:

```text
L1 candidate anomalies: 5
GPS: 1126077385 | Error: 0.048695
GPS: 1126077807 | Error: 0.048207
GPS: 1126078534 | Error: 0.057253
GPS: 1126079216 | Error: 0.060190
GPS: 1126079431 | Error: 0.055075
```

Only one candidate also had anomalous Hanford data nearby:

```text
L1 GPS: 1126077807
H1 GPS: 1126077808
H1 reconstruction error: 0.016617
```

Fine-time analysis found:

```text
Best H1-L1 delay: -2.197 ms
Correlation: 0.293114
```

The delay was physically plausible, but background testing showed that the correlation was not unusual:

```text
Quality-passing paired regions: 2803
Background comparisons: 200
Background regions >= candidate: 86
Empirical p-value: 0.432836
```

The candidate was therefore rejected as compatible with ordinary background.

The complete reduction was:

**5 L1 anomalies → 1 cross-detector candidate → 0 surviving candidates**

This demonstrates the intended behavior of the pipeline: initial machine-learning anomalies are treated as candidates to investigate rather than automatic detections.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Core detection and analysis pipeline |
| **PyTorch** | Convolutional autoencoder and reconstruction-error analysis |
| **PyCBC** | LIGO noise simulation and strain signal processing |
| **GWOSC** | Public LIGO strain data and detector metadata |
| **h5py / HDF5** | Reading strain, quality masks, and hardware-injection metadata |
| **Conv1d / ConvTranspose1d** | Temporal encoding and reconstruction of strain windows |
| **FFT** | Frequency-domain candidate characterization |

## Project Structure

| File | Purpose |
| --- | --- |
| `simulation.py` | Generates simulated Advanced LIGO background and one-second strain windows. |
| `model.py` | Defines and trains the convolutional autoencoder using simulated and real strain. |
| `Main.py` | Locates GWOSC data and prepares quality-filtered real LIGO training strain. |
| `anomolydetection.py` | Performs anomaly detection, H1/L1 validation, correlation testing, and candidate classification. |
| `physics.py` | Performs physical interpretation of validated strain candidates and time-dilation reference calculations. |
| `data/` | Stores local GWOSC HDF5 files and is excluded from version control. |

## Installation

### Prerequisites

- Python 3
- `pip`
- Internet access for retrieving GWOSC data
- Local GWOSC HDF5 strain files used by the configured training and test intervals

### Clone the Repository

```bash
git clone https://github.com/CarterNeisz10/Gravitational-Anomaly-Detector.git
cd Gravitational-Anomaly-Detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

GWOSC strain files and trained model weights are excluded from version control because of their size.

## Usage

### Train the Model

Run:

```bash
python model.py
```

The model first trains on simulated Advanced LIGO background and then continues training on quality-filtered real Livingston strain.

The trained parameters are saved locally as:

```text
strain_autoencoder.pth
```

### Run Anomaly Detection

Run:

```bash
python anomolydetection.py
```

The application calculates detector thresholds, scans unseen strain, cross-checks candidates between L1 and H1, performs fine-time and empirical background validation, and displays the final result.

## Example Behaviors

### Initial Anomaly Detection

```text
Scanning unseen L1 data...
L1 candidate anomalies: 5
```

The autoencoder identifies windows whose reconstruction error exceeds the detector-specific background threshold.

### Cross-Detector Candidate

```text
L1 candidate GPS 1126077807
H1 anomaly found nearby:
  GPS: 1126077808
```

An independently anomalous H1 window allows the candidate to proceed to fine-time validation.

### Candidate Rejection

```text
Best conditioned H1-L1 delay: -2.197 ms
Conditioned correlation: 0.293114
Empirical p-value: 0.432836

REJECTED
Reason: cross-detector correlation is compatible with ordinary background.
```

Although the candidate passes the physical timing check, its correlation is sufficiently common in ordinary background that it is rejected.

## Design Decisions

### Unsupervised Anomaly Detection

The autoencoder learns background strain rather than being trained to recognize specific gravitational-wave classes.

This allows reconstruction error to act as a general measure of how strongly an unseen window differs from learned detector background.

### Hybrid Simulated and Real Training

Simulated Advanced LIGO noise provides a controlled initial training distribution, while real quality-filtered strain exposes the model to characteristics of actual interferometer data.

Training sequentially on both combines these advantages without reinitializing the model between stages.

### Independent Detector Thresholds

Livingston and Hanford are separate instruments and produce different reconstruction-error distributions.

Each detector therefore receives its own threshold rather than assuming that one numerical anomaly score has identical meaning across both detectors.

### Separate Detection and Validation

A high reconstruction error determines that a window is unusual, not that it contains a gravitational event.

Cross-detector coincidence, physical timing, and empirical background comparison are separate stages specifically designed to test whether an initial anomaly deserves further analysis.

### Empirical Background Screening

Cross-detector correlation is evaluated relative to correlations naturally occurring in quality-passing detector background.

This prevents a nonzero or physically timed correlation from being treated as significant without a reference distribution.

## Limitations

Gravitational Anomaly Detector is an experimental anomaly-detection system and is not a replacement for official LIGO/Virgo/KAGRA gravitational-wave search pipelines.

Current limitations include:

- **Limited training data** — The autoencoder is trained on a relatively small subset of public detector strain.
- **Simple model architecture** — The convolutional autoencoder is intentionally compact and does not use more advanced temporal architectures.
- **One-second windows** — Initial anomaly localization is limited by the fixed window duration.
- **Heuristic anomaly threshold** — The 99th-percentile reconstruction threshold is a project-level screening choice.
- **Limited background sample** — Empirical testing currently uses up to 200 background comparisons.
- **Simplified significance testing** — The empirical p-value is not equivalent to formal LIGO false-alarm or astrophysical significance calculations.
- **Basic signal conditioning** — The project uses broad frequency filtering rather than a complete professional gravitational-wave preprocessing pipeline.
- **No source classification** — The system does not attempt to classify surviving anomalies as particular astrophysical source types.
- **Gravity-only time dilation** — Altitude calculations do not include special-relativistic effects caused by orbital velocity.

These constraints are intentionally surfaced rather than hidden: the project demonstrates an end-to-end machine-learning and validation pipeline using real gravitational-wave detector data, not a production astrophysical search system.

## Future Improvements

Potential extensions include:

- Training on larger amounts of Livingston and Hanford background data
- Evaluating additional LIGO observing periods
- Increasing the number of empirical background comparisons
- Using overlapping windows for finer anomaly localization
- Comparing additional autoencoder architectures
- Adding more advanced strain conditioning and whitening
- Evaluating the detector against known gravitational-wave events
- Testing sensitivity using controlled signal injections
- Adding visualizations of reconstruction error, strain, and cross-detector correlation