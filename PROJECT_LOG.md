# Mural Restoration Project - Development Log

This document tracks all major and minor updates, milestones, and implementation steps throughout the project lifecycle.

---

## Project Overview

**Project Name:** Mural Restoration Project  
**Phase:** Phase 1  
**Objective:** Implement a comparative image restoration framework across four approaches (PDE-based, Patch-based, CNN-based, Hybrid) using MuralDH dataset (Mural512 as ground truth) and simulate damage using custom or segmentation-based masks.

**Dataset:** MuralDH/Mural512  
**Technologies:** Python 3.x, Jupyter Notebooks, OpenCV, NumPy, Matplotlib, PyTorch/TensorFlow, LPIPS, scikit-image, pandas

---

## Milestones

### Milestone 0: Project Initialization
**Date:** [To be filled]  
**Type:** Major  
**Status:** ✅ Completed

#### Steps Completed:
- [x] Project repository setup
- [x] Master prompt document created (`master_prompt_phase1.txt`)
- [x] Project structure defined
- [x] Dataset location identified (MuralDH/Mural512)
- [x] Git repository initialized with appropriate .gitignore
- [x] Project documentation framework established

---

### Milestone 1: Dataset Preparation and Mask Generation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Create folder structure (data/, methods/, notebooks/)
- [ ] Implement Notebook_1_DatasetMask.ipynb
  - [ ] Load Mural512 images
  - [ ] Generate synthetic damage masks
  - [ ] Apply masks to create damaged versions
  - [ ] Copy ground truth images to data/ground_truth/
  - [ ] Save masks to data/masks/
  - [ ] Save damaged images to data/damaged/
  - [ ] Verify mask alignment with images
- [ ] Test mask generation pipeline
- [ ] Validate data consistency

**Notes:**
- 

---

### Milestone 2: PDE-Based Restoration Implementation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_2_PDE.ipynb
  - [ ] Implement PDE-based inpainting algorithm
  - [ ] Load damaged images and masks
  - [ ] Process images through PDE pipeline
  - [ ] Save restored images to methods/PDE/results/
  - [ ] Ensure consistent naming convention
- [ ] Test PDE restoration on sample images
- [ ] Optimize parameters if needed
- [ ] Document PDE algorithm details

**Notes:**
- 

---

### Milestone 3: Patch-Based Restoration Implementation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_3_Patch.ipynb
  - [ ] Implement patch-based inpainting algorithm
  - [ ] Load damaged images and masks
  - [ ] Process images through patch-based pipeline
  - [ ] Save restored images to methods/Patch/results/
  - [ ] Ensure consistent naming convention
- [ ] Test patch-based restoration on sample images
- [ ] Optimize patch size and search parameters
- [ ] Document patch-based algorithm details

**Notes:**
- 

---

### Milestone 4: Deep Learning (CNN/U-Net) Implementation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_4_Deep.ipynb
  - [ ] Design/select CNN/U-Net architecture
  - [ ] Prepare training data (damaged images + masks)
  - [ ] Implement training loop
  - [ ] Train model on training set
  - [ ] Validate model performance
  - [ ] Test model on test images
  - [ ] Save restored images to methods/Deep/results/
  - [ ] Save trained model weights
- [ ] Hyperparameter tuning
- [ ] Model evaluation and optimization
- [ ] Document model architecture and training details

**Notes:**
- 

---

### Milestone 5: Hybrid Approach Implementation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_5_Hybrid.ipynb
  - [ ] Load PDE restoration outputs
  - [ ] Integrate CNN refinement pipeline
  - [ ] Process PDE outputs through CNN
  - [ ] Save hybrid restored images to methods/Hybrid/results/
  - [ ] Ensure consistent naming convention
- [ ] Test hybrid pipeline end-to-end
- [ ] Optimize pipeline parameters
- [ ] Document hybrid approach methodology

**Notes:**
- 

---

### Milestone 6: Quantitative Evaluation
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_6_Evaluation.ipynb
  - [ ] Load all restored images from all methods
  - [ ] Load ground truth images
  - [ ] Implement PSNR calculation
  - [ ] Implement SSIM calculation
  - [ ] Implement LPIPS calculation
  - [ ] Implement edge accuracy metric
  - [ ] Compute metrics for all methods
  - [ ] Store results in CSV format
  - [ ] Generate summary statistics
- [ ] Validate metric calculations
- [ ] Create comparison tables
- [ ] Document evaluation methodology

**Notes:**
- 

---

### Milestone 7: Visualization and Reporting
**Date:** [To be filled]  
**Type:** Major  
**Status:** ⏳ Pending

#### Steps:
- [ ] Implement Notebook_7_Visualization.ipynb
  - [ ] Create side-by-side visual comparisons
  - [ ] Generate qualitative analysis plots
  - [ ] Create metric comparison charts
  - [ ] Prepare visual reports
  - [ ] Document findings
- [ ] Generate final report
- [ ] Create presentation materials (if needed)

**Notes:**
- 

---

## Minor Updates Log

### [Date] - [Update Title]
**Type:** Minor  
**Description:** Brief description of the update

**Changes:**
- 

---

## Technical Notes

### Folder Structure
```
project/
  data/
    ground_truth/
    damaged/
    masks/
  methods/
    PDE/
      results/
    Patch/
      results/
    Deep/
      results/
    Hybrid/
      results/
  notebooks/
    Notebook_1_DatasetMask.ipynb
    Notebook_2_PDE.ipynb
    Notebook_3_Patch.ipynb
    Notebook_4_Deep.ipynb
    Notebook_5_Hybrid.ipynb
    Notebook_6_Evaluation.ipynb
    Notebook_7_Visualization.ipynb
```

### Key Requirements
- Each notebook operates independently
- Consistent naming for inputs and outputs
- Masks must align perfectly with damaged and ground truth images
- Use synthetic damage masks for controlled experimentation

### Libraries Used
- OpenCV
- NumPy
- Matplotlib
- PyTorch/TensorFlow
- LPIPS
- scikit-image
- pandas

---

## Issues and Resolutions

### [Issue Title]
**Date:** [Date]  
**Status:** [Resolved/Open/In Progress]  
**Description:**  
**Resolution:**  

---

## Future Enhancements

- [ ] Phase 2 planning
- [ ] Advanced analysis with real masks
- [ ] Additional restoration methods
- [ ] Performance optimization
- [ ] Extended dataset support

---

## Version History

| Version | Date | Description | Milestone |
|---------|------|-------------|-----------|
| 0.1.0 | [Date] | Project initialization | Milestone 0 |

---

**Last Updated:** [Date]  
**Current Phase:** Phase 1  
**Overall Progress:** 0% (0/7 major milestones completed)

