# Capsulorhexis Analysis Project

This repository contains a **complete, end-to-end AI-powered pipeline** for analyzing **capsulorhexis** quality in cataract-surgery videos. The primary goal is to **detect** the capsulorhexis creation phase, **segment** the relevant instruments (such as **Capsulorhexis Forceps**), **track** the tear boundary, and **classify** the final capsulorhexis as “ideal” or “at-risk.” The system is designed to highlight videos or segments that may lead to **Posterior Capsule Opacification (PCO)** if the capsulorhexis is off-center, too large, or irregular.

---

## Why Capsulorhexis Analysis Matters

- **Capsulorhexis** (the circular tear in the anterior capsule of the lens) is a critical step in cataract surgery.  
- A **well-centered, ~5 mm** capsulorhexis that overlaps the intraocular lens (IOL) optic helps reduce PCO rates.  
- An **eccentric** or **overly large** tear may leave gaps, allowing lens epithelial cells to migrate behind the lens, **increasing PCO risk**.

By **automating** the size, shape, and centering measurement of the capsulorhexis, surgeons and researchers can:

1. **Identify** best practices for improved long-term patient outcomes.  
2. **Quantify** metrics for surgical training and skill evaluation.  
3. **Efficiently** analyze large sets of videos without manually checking each tear.

---

## High-Level Architecture

1. **Phase Recognition**  
   - **Purpose**: Find the **time interval** where capsulorhexis occurs in each surgery video.  
   - **Model**: We use a **pretrained single-frame** classification approach or adapt a **Video Swin Transformer**—depending on user preference.  
   - **Benefit**: Saves computation by ignoring frames from other phases like phacoemulsification or lens implantation.  

2. **Surgical Instrument Segmentation**  
   - **Purpose**: Identify relevant instruments in each video frame—particularly the tools used to create the capsulorhexis (e.g., **Capsulorhexis Forceps**).  
   - **Data**: We have `.mp4` files plus **COCO annotations** (with polygons/bboxes) in `instances.json` for each case.  
   - **Classes**: We only retain **Capsulorhexis Forceps**, **Capsulorhexis Cystotome**, and **Katena Forceps**, ignoring all other instruments.  

3. **Non-AI Capsulorhexis Boundary Detection**  
   - **Purpose**: Rather than a neural net for capsule boundary segmentation, we use a **geometric** or “tip-tracking” approach.  
   - **Process**:  
     1. Extract the **instrument tip** across frames.  
     2. Accumulate tip points.  
     3. Detect a 360° enclosed path, signifying a completed circular tear.  

4. **Measurement & Analysis**  
   - From the tear boundary, compute:  
     - **Diameter** (via bounding box or fitted circle).  
     - **Circularity** check (detect jagged edges or incomplete tears).  
     - **Center offset** (compare tear center to pupil center or corneal vertex).  

5. **Classification** (“Ideal” or “At-Risk”)  
   - A **rule-based** system. For instance:  
     - **Diameter** within [4.5 mm, 5.5 mm].  
     - **Circularity** > 0.8.  
     - **Center offset** < 1 mm.  
   - If all checks pass → “**Ideal**.” Otherwise → “**At-Risk**,” with specific reasons (“Too large,” “Off-center,” etc.).

---

## Project Workflow

1. **Dataset Setup**  
   - **Phase** data in `Cataract-1k-Phase` (e.g., videos + CSV annotations with start/end frames).  
   - **Seg** data in `Cataract-1k-Seg` (e.g., `.mp4` plus `Annotations/Coco-Annotations/case_XXXX/instances.json`).  
   - The segmentation code reads **frames on the fly** from `.mp4` (based on the `file_name` inside the JSON, which indicates the frame).

2. **Phase Model**  
   - Trained from **single-frame** classification or loaded as a pretrained model (`phase_recognition.pth`).  
   - Quick “frame sampling” approach during inference to find the approximate capsulorhexis time interval.

3. **Segmentation Model Training**  
   - We only focus on **Capsulorhexis Forceps**, **Capsulorhexis Cystotome**, and **Katena Forceps**.  
   - Our dataset script `cataract_seg_dataset.py`:
     1. Reads **COCO** annotations.  
     2. Parses `file_name` like `"case5353_01.png"` to get `(case_id, frame_idx)`.  
     3. Opens `videos/case_5353.mp4`, seeks the frame, and builds a **mask** with relevant classes.  
   - The training script logs **loss**, **IoU**, and **Dice** metrics each epoch, saving a combined plot.

4. **Inference Pipeline**  
   - **Phase** detection → find frames where capsulorhexis is performed.  
   - **Segmentation** → identify instruments in that time window.  
   - **Tip tracking** → approximate circular tear.  
   - **Measurement** → diameter, circularity, center offset.  
   - **Classification** → “Ideal” vs. “At-Risk” with reasons.

---

## Usage Guide

### 1) Dependencies

```bash
pip install -r requirements.txt
```

### 2) Dataset Setup

- If you have `.mp4` files plus COCO `instances.json` in `Cataract-1k-Seg`, ensure they are in the correct structure:
  ```
  Cataract-1k-Seg/
  ├── videos/
  │   ├── case_0001.mp4
  │   └── case_0002.mp4
  └── Annotations/
      └── Coco-Annotations/
          ├── case_0001/
          │   └── annotations/
          │       └── instances.json
          └── case_0002/
              └── annotations/
                  └── instances.json
  ```
- Similarly for phase data (`Cataract-1k-Phase`).

### 3) Shell Scripts Overview

#### `train_segmentation.sh`  
- One-liner to train the segmentation model on the relevant classes. Example:
  ```bash
  ./train_segmentation.sh Cataract-1k-Seg
  ```
- Internally calls `train_segmentation.py`, which:
  1. Splits data into train/val.  
  2. Logs **loss**, **IoU**, and **Dice** each epoch.  
  3. Saves final model to `models/lightweight_seg.pth`.  
  4. Produces a **plot** (`seg_training_metrics.png`) of training curves.

#### `train_phase.sh`  
- One-liner to train the **phase** classification model on `Cataract-1k-Phase`. Example:
  ```bash
  ./train_phase.sh Cataract-1k-Phase
  ```
- Uses a **single-frame** classifier or your own 3D architecture.  
- Produces `phase_recognition.pth`.

#### `inference_pipeline.sh`  
- Demonstrates **end-to-end** inference:
  ```bash
  ./inference_pipeline.sh /path/to/video.mp4 \
                          /path/to/phase_recognition.pth \
                          /path/to/lightweight_seg.pth
  ```
- Loads the phase model, quickly identifies the capsulorhexis segment, and runs segmentation only on that portion.  
- Outputs final classification: “Ideal” or “At-Risk,” plus reasons (e.g. “Off-center by 1.2 mm”).

### 4) Training Segmentation with IoU & Dice

- By default, the updated `train_segmentation.py` computes and logs **IoU** and **Dice** (skipping background).  
- Final training curves are saved to **`seg_training_metrics.png`**.

### 5) Pipeline Inference

1. The **phase model** identifies frames of interest.  
2. The **segmentation model** extracts the instruments.  
3. We do a **geometry-based** approach for the tear boundary (e.g., cumulative tip coordinates).  
4. We measure the tear’s size, shape, center.  
5. We output a simple pass/fail classification with reasoning.

---

## Development Flow

1. **Phase**  
   - Single-frame or short-clip classification → learns phases from CSV.  
   - Inference → find capsulorhexis time window.

2. **Segmentation**  
   - “On-the-fly” reading of `.mp4` frames.  
   - COCO annotations for relevant classes only:
     - `Capsulorhexis Forceps`  
     - `Capsulorhexis Cystotome`  
     - `Katena Forceps`  
   - Model logs **loss** + **IoU** + **Dice** each epoch.

3. **Geometric Rhexis Tracking**  
   - Accumulate tip positions → detect 360° tear.  
   - Fit circle/bounding box → measure diameter and center.

4. **Classification**  
   - “Ideal” if diameter ~5 mm, circularity ~1, offset < 1 mm.  
   - “At-Risk” otherwise, specifying the cause.

---

## Potential Extensions

- **Instrument-Specific**: If you later want more classes (Irrigation-Aspiration, Gauge, etc.), expand the dataset code.  
- **Frame Caching**: Repeated frame extraction from `.mp4` is slow. You could **pre-extract** frames offline for faster training.  
- **Advanced Pupillometry**: Precisely track pupil boundary to measure offset.  
- **Real-Time**: Convert the pipeline for real-time feedback in the OR.

---

## Conclusion

The **Capsulorhexis Analysis Project** provides:

- **Phase detection** to locate capsulorhexis frames,  
- **Segmentation** of key instruments (forceps, cystotome, etc.),  
- **Geometric** detection of the tear,  
- **Measurement** of diameter, shape, centering,  
- **Classification** into “Ideal” or “At-Risk.”

This end-to-end workflow allows surgeons and researchers to **quickly** evaluate large volumes of cataract-surgery footage, offering **quantitative** metrics and **data-driven** feedback on the capsulorhexis creation step—potentially leading to **reduced** Posterior Capsule Opacification through better technique.