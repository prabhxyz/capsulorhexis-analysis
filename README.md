# Capsulorhexis Analysis Project

This repository contains a **complete, end-to-end AI-powered pipeline** for analyzing **capsulorhexis** quality in cataract-surgery videos. The primary goal is to **detect** the capsulorhexis creation phase, **segment** the surgical instruments, **mathematically track** the tear boundary, and **classify** the final capsulorhexis as “ideal” or “at-risk.” The system is designed to highlight videos or segments that might lead to **Posterior Capsule Opacification (PCO)** if the capsulorhexis is off-center, too large, or irregular.

---

## Why Capsulorhexis Analysis Matters

- **Capsulorhexis** (the circular tear in the anterior capsule of the lens) is a critical step in cataract surgery.
- A **well-centered, ~5 mm** capsulorhexis that evenly overlaps the intraocular lens (IOL) optic is strongly associated with **reduced PCO rates**.
- An **eccentric** or **overly large** tear can leave gaps that allow lens epithelial cells to migrate behind the lens, **increasing the risk** of PCO.

By **automatically measuring** the size, shape, and centering of the capsulorhexis, surgeons and researchers can:

1. Identify patterns leading to better long-term patient outcomes.
2. Quantify surgical performance and training metrics.
3. Efficiently review large sets of videos without manually measuring each capsulorhexis.

---

## High-Level Architecture

1. **Phase Recognition** (Video Swin Transformer or similar)  
   - **Purpose**: Identify the **time interval** in each surgery video when the capsulorhexis is created.  
   - **Benefit**: Avoids unnecessary computation on other phases (e.g., incisions, phacoemulsification).  
   - **Data**: Uses `Cataract-1k-Phase` style data, where each video (`case_XXXX.mp4`) has a corresponding annotation CSV specifying phases and frame ranges.

2. **Tool Segmentation** (Mask R-CNN)  
   - **Purpose**: Segment the relevant instruments (e.g., **Capsulorhexis Forceps**, **Capsulorhexis Cystotome**, etc.) at a pixel level.  
   - **Benefit**: Pinpoints where the surgeon is tearing the capsule.  
   - **Data**: Uses `Cataract-1k-Seg` style data, where each video has COCO-format instance annotations for the relevant instruments.

3. **Non-AI Capsulorhexis Boundary Detection**  
   - **Purpose**: Instead of using a deep-learning model to segment the capsular tear, this project uses a **geometric approach**:
     1. Track the **instrument tip** over time.  
     2. Accumulate the tip’s path.  
     3. Detect when it encloses a roughly **circular path** (i.e., a 360° tear).  
   - **Benefit**: Reduces complexity and potential error from faint visual boundaries; it’s more robust to simply watch the instrument movement that creates the tear.

4. **Measurement & Analysis**  
   - Once the **binary mask** of the capsulorhexis is approximated (or the tear boundary is computed), the system:
     - **Measures the diameter** (by bounding box or circle fitting).  
     - Checks **circularity** to detect jagged edges.  
     - Estimates **centering** by comparing tear center to the pupil center or corneal vertex (with a simple geometric reference).  
   - **Methods**: 
     - Circle or ellipse fitting to the tear outline.  
     - Reference instrument size to approximate **millimeters** per pixel.

5. **Classification** (“Ideal” vs. “At-Risk”)  
   - A **rule-based** system combining:
     - **Size** check (e.g., 4.5 mm to 5.5 mm is optimal).  
     - **Circularity** check (circularity > 0.8).  
     - **Center offset** threshold (e.g., off-center by less than 1.0 mm).  
   - Outputs a **clear reason** if “At-Risk” is triggered:  
     - “Too large: 6.0 mm”  
     - “Off-center by 1.2 mm”  
     - “Not round enough”  

---

## Detailed Steps

1. **Phase Recognition Training**  
   - **Dataset**: `<root_dir>/videos/` has `case_XXXX.mp4` files.  
   - **Annotations**: `<root_dir>/annotations/case_XXXX/case_XXXX_annotations_phases.csv` with columns like `comment`, `frame`, `endFrame`.  
   - **Model**: Video **Swin3D** or Vision Transformer that receives short **clips** (e.g., 16 frames).  
   - **Goal**: Predict which surgical phase each clip belongs to. During inference, the system locates the time window for capsulorhexis creation.

2. **Segmentation Training**  
   - **Dataset**: `<root_dir>/videos/` has frames or short videos, and `<root_dir>/Annotations/Coco-Annotations/case_XXXX/annotations/instances.json` with COCO-format polygons.  
   - **Classes**: Only relevant instruments (e.g., CapsulorhexisCystotome, CapsulorhexisForceps).  
   - **Model**: **Mask R-CNN** with a ResNet-FPN backbone.  
   - **Goal**: Predict a mask for each instrument in each frame.

3. **Capsulorhexis Detection (Non-AI)**  
   - **Input**: The segmented instrument masks.  
   - **Process**:  
     1. Identify the **tip** of the relevant instrument.  
     2. Track its (x,y) coordinates across frames.  
     3. Check if these points form a **closed circle** (or near 360° coverage) around a centroid.  
     4. When a circle is detected, build a **binary mask** approximating the rhexis.  

4. **Measurement**  
   - **Binary mask** → **Contour** → Fit bounding box or circle.  
   - Convert **pixel distance** to mm using known references (e.g., instrument width or corneal diameter).  
   - **Compute**:  
     - **Diameter**: average of bounding box height and width.  
     - **Circularity**: \( 4 \pi \cdot \text{area} / (\text{perimeter}^2) \).  
     - **Center**: moment-based centroid.

5. **Classification**  
   - **Rule-based** thresholds. Example:  
     - Diameter in [4.5 mm, 5.5 mm]?  
     - Circularity > 0.8?  
     - Center offset < 1.0 mm?  
   - If **all** conditions pass: “**Ideal**.” Otherwise: “**At-Risk**.”  
   - Provide reasons for “At-Risk” (size, shape, or centering).

---

## Usage Guide

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train Phase Recognition**  
   ```bash
   ./train_phase.sh /path/to/Cataract-1k-Phase
   ```
   - Finds subclips where capsulorhexis occurs, trains a **video classification** model.

3. **Train Segmentation**  
   ```bash
   ./train_segmentation.sh /path/to/Cataract-1k-Seg
   ```
   - Only segments relevant surgical instruments (like capsulorhexis forceps).

4. **Run Inference**  
   ```bash
   ./inference_pipeline.sh /path/to/new_surgery_video.mp4 \
                           /path/to/models/phase_recognition \
                           /path/to/models/segmentation
   ```
   - **Outputs** the classification: “Ideal” or “At-Risk,” along with any reasons (e.g., diameter out of range).

---

## Development Flow

- **Phase Recognition**:
  1. Preprocessing and **clip extraction** (16-frame windows).  
  2. 3D Swin Transformer loads these clips.  
  3. Model classifies which phase it belongs to.  
  4. During **inference**, the system extracts the **capsulorhexis time**.

- **Segmentation**:
  1. Mask R-CNN is trained on frames with **instrument annotations**.  
  2. At runtime, it identifies the **forceps/cystotome** in each frame of the relevant phase.  
  3. The **tool tip** is extracted from the segmented mask.

- **Capsule Detection (Non-AI)**:
  1. Accumulate tool tip coordinates.  
  2. Detect a complete **circle** (heuristic checks for 360° coverage).  
  3. Construct a **binary mask** of that circle, representing the final tear boundary.

- **Measurement**:
  - Use **OpenCV** contour operations to measure diameter, shape, and center.

- **Classification**:
  - If the rhexis meets **size** (≈5 mm), **circularity**, and **centering** criteria → “Ideal.”  
  - Otherwise → “At-Risk,” with specific reasons printed.

---

## Potential Extensions

- **Advanced Pupil Tracking**: Instead of using a fixed reference center, segment and track the **pupil** to more accurately measure offset.  
- **Calibration**: If you have a **scale** marker or known corneal diameter, you can refine the mm-per-pixel ratio.  
- **Domain Adaptation**: Fine-tune the segmentation model on additional data if the lighting or instruments differ from `Cataract-1k-Seg`.  
- **Uncertainty Estimation**: In high-stakes surgeries, consider Bayesian or Monte Carlo methods to highlight uncertain frames.

---

## Conclusion

The **Capsulorhexis Analysis Project** is a unified framework to:

- **Identify** the capsulorhexis step,  
- **Segment** instruments that create the tear,  
- **Derive** a final tear boundary **mathematically**,  
- **Measure** diameter and shape,  
- **Classify** outcomes as “ideal” vs. “at-risk” with clear reasons.

This workflow empowers surgeons and researchers to **quickly evaluate surgical videos**, track performance metrics, and potentially **reduce** Posterior Capsule Opacification through **data-driven** feedback on capsulorhexis quality.