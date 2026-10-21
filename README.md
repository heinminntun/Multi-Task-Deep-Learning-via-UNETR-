# Multi-Task Deep Learning via UNETR for PSMA PET/CT in Prostate Cancer

### Simultaneous Assessment of Disease Burden, Treatment Selection, and Survival Outcomes

This repository accompanies our published research on a **multi-task deep learning framework using UNETR (Transformers for Medical Image Segmentation)** to investigate the potential of whole-body prostate-specific membrane antigen positron emission tomography/computed tomography (PSMA PET/CT) for prostate cancer assessment.

The framework explores how a single PSMA PET/CT image can support multiple complementary analytical tasks: lesion segmentation, metastatic disease burden quantification, CHAARTED and LATITUDE risk stratification, treatment category classification, and overall survival risk prediction.

Unlike conventional single-task approaches that address segmentation, classification, or prognosis independently, this research investigates a shared deep learning architecture designed to learn image representations for several clinically relevant tasks simultaneously.

**Research paper:** [Read the full publication](https://doi.org/10.1007/s10278-026-02224-3)

**Source code:** [GitHub repository](https://github.com/heinminntun/Multi-Task-Deep-Learning-via-UNETR-)

---

## Research Overview

PSMA PET/CT provides whole-body molecular imaging that can reveal primary prostate tumours and metastatic disease. However, translating these images into quantitative disease burden estimates, treatment-related classifications, and prognostic information remains a challenging problem.

This project investigates a unified deep learning workflow that combines transformer-based image representations with task-specific prediction heads to explore these related objectives.

The study used a retrospective cohort from The Brunei Cancer Centre, Jerudong Park Medical Centre, Brunei Darussalam, covering the period 2018–2024.

### Key contributions

- **Multi-task learning:** Investigates lesion segmentation, treatment category classification, and survival risk prediction within a shared UNETR-based architecture.
- **Image-based disease burden assessment:** Uses predicted lesion masks to estimate lesion distribution and derive CHAARTED and LATITUDE classifications.
- **Transformer-based image representation:** Explores Vision Transformer encoder features for learning global spatial information from PSMA PET/CT fusion images.
- **Comparative evaluation:** Compares the multi-task UNETR approach with a conventional convolutional neural network (CNN) for treatment category classification.
- **Clinically interpretable outputs:** Explores segmentation-based tumour quantification and metastatic disease burden assessment alongside predictive outputs.

## Model Architecture

The framework uses a shared image encoder and task-specific output heads to investigate multiple objectives from the same image input.

```text
             Whole-body PSMA PET/CT
                       |
                       v
              Image Preprocessing
       Cropping, Normalization, Resizing
                       |
                       v
                UNETR Encoder
             Vision Transformer
                       |
              Shared Representation
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Lesion       Treatment      Survival
    Segmentation  Classification  Risk Model
          |            |            |
          v            v            v
     Lesion Masks   Treatment     Overall
     and Burden      Category     Survival Risk
          |
          v
    CHAARTED and LATITUDE
    Disease Stratification
```

*Conceptual overview of the multi-task architecture. Disease burden stratification is derived from predicted lesion distributions.*

### Main prediction tasks

| Task | Objective | Output |
|---|---|---|
| Lesion segmentation | Identify candidate tumour regions in PSMA PET/CT images | Pixel-level segmentation probability map |
| Disease burden quantification | Estimate lesion distribution, including bone and visceral disease | Lesion counts and burden estimates |
| CHAARTED classification | Derive high-volume versus low-volume disease classification | Disease volume category |
| LATITUDE classification | Derive high-risk versus low-risk disease classification | Disease risk category |
| Treatment classification | Predict a treatment category from image features | Multiclass probabilities |
| Survival prediction | Estimate relative overall survival risk | Cox-model risk score |

## Dataset

The study used retrospectively collected clinical and imaging data from The Brunei Cancer Centre.

- **Study period:** 2018–2024.
- **Clinical cohort:** 212 prostate cancer cases with 478 time-indexed observations.
- **Imaging dataset:** 160 anonymized whole-body PSMA PET/CT images.
- **Matched analytical dataset:** 205 image–clinical records from 115 patients.
- **Image format:** Anonymized PNG images derived from PSMA PET/CT fusion displays.
- **Data splitting:** Patient-level group-wise splitting to prevent images from the same patient appearing in both training and testing sets.

The original DICOM files were excluded from the modeling workflow because of patient confidentiality considerations. The image representation used in this study therefore does not retain all quantitative information available in the original DICOM data.

**Data availability:** Patient-level clinical data and medical images are not publicly distributed with this repository because of privacy, confidentiality, and ethical restrictions.

## Image Preprocessing

The image preparation pipeline included:

1. **Bounding-box cropping** to reduce redundant background regions.
2. **Intensity normalization** using body-masked Z-score normalization, with fallback scaling when body-region detection was unsuccessful.
3. **Image preparation** for the CNN and UNETR model inputs.
4. **Weakly supervised lesion mask generation** using Otsu thresholding on PET maximum intensity projection images, followed by morphological refinement and removal of small isolated components.
5. **Patient-level data partitioning** to preserve the separation of patients between training and testing datasets.

The Otsu-derived masks were used as pragmatic weak-supervision targets. They should not be interpreted as equivalent to comprehensive, expert-annotated lesion reference masks.

## Technologies

The implementation uses Python and medical imaging/deep learning libraries, including:

- [PyTorch](https://pytorch.org/) — deep learning and model optimization.
- [MONAI](https://monai.io/) — medical imaging deep learning and the UNETR architecture.
- [scikit-learn](https://scikit-learn.org/) — preprocessing and machine learning utilities.
- [scikit-survival](https://scikit-survival.readthedocs.io/) — survival analysis utilities.
- [OpenCV](https://opencv.org/) — image processing and segmentation-mask preparation.

The UNETR model uses a Vision Transformer encoder with a U-Net-style decoder. Shared encoder representations are connected to task-specific heads for segmentation, treatment classification, and survival risk estimation.

## Experimental Results

The study evaluated the CNN and multi-task UNETR architectures using treatment classification, segmentation, and survival prediction metrics.

| Evaluation | Model | Reported result |
|---|---|---:|
| Treatment classification | CNN | Macro-average AUC = 0.91 |
| Active surveillance classification | CNN | AUC = 0.99 |
| Chemotherapy ± ADT/ARPI classification | CNN | AUC = 0.97 |
| Treatment classification | Multi-task UNETR | Weighted multiclass AUC = 0.561 |
| Overall survival prediction | Multi-task UNETR | Concordance index = 0.491 |
| Overall survival prediction | Multi-task UNETR | Mean cumulative dynamic AUC = 0.586 |

The CNN achieved stronger treatment classification discrimination in this study. The multi-task UNETR demonstrated the feasibility of producing segmentation, disease burden, treatment classification, and survival-related outputs within one architecture, although its classification and survival discrimination remained limited.

The study also observed higher mean segmentation Dice coefficients among deceased patients than survivors, but the difference was not statistically significant.

**Interpretation:** These results are preliminary and should not be interpreted as evidence of clinical effectiveness. The multi-task framework requires further optimization and independent validation.

## Limitations and Future Directions

The study has several important limitations:

- **Limited sample size:** The matched dataset contained 205 image–clinical records from 115 patients, limiting generalizability and increasing the risk of overfitting.
- **PNG-derived imaging:** Conversion from DICOM to PNG may result in the loss of quantitative PET information, including standardized uptake values (SUVs), and limits direct volumetric analysis.
- **Weakly supervised segmentation:** Threshold-derived masks may introduce systematic errors, particularly for small or low-contrast lesions.
- **Imbalanced treatment categories:** Differences in category representation may affect classification performance and weighted multiclass AUC.
- **Limited predictive performance:** The UNETR model achieved modest treatment discrimination and a survival concordance index close to 0.5.
- **Unvalidated clinical utility:** The potential interpretability benefits of segmentation-based outputs require formal evaluation with oncologists and radiologists.

Future work should investigate larger multicentre DICOM datasets, expert-annotated segmentation references, external validation, improved task balancing, and prospective evaluation of clinical utility.

## Reproducibility and Code Availability

The source code associated with this research is available in this repository:

**[https://github.com/heinminntun/Multi-Task-Deep-Learning-via-UNETR-](https://github.com/heinminntun/Multi-Task-Deep-Learning-via-UNETR-)**

The published study used a controlled Python environment with PyTorch, MONAI, scikit-learn, scikit-survival, and OpenCV.

Please consult the repository's actual source files for the available training scripts, configuration files, preprocessing implementation, and execution instructions. A complete reproduction of the experiments also requires appropriately authorized input data and the corresponding clinical labels.

## Publication

If you use this code or build upon this research, please cite the original publication.

**Tun, H. M., Naing, L., Malik, O. A., Abdullah, M. S., Ta, T., & Abdul Rahman, H. (2026).** *Multi-task Deep Learning via UNETR for PSMA PET/CT Image for Prostate Cancer: Simultaneous Assessment of Prostate Cancer Disease Burden, Treatment Selection and Survival Outcomes.* Journal of Imaging Informatics in Medicine.

https://doi.org/10.1007/s10278-026-02224-3

### BibTeX

```bibtex
@article{tun2026multitaskunetr,
  title   = {Multi-task Deep Learning via UNETR for PSMA PET/CT
             Image for Prostate Cancer: Simultaneous Assessment
             of Prostate Cancer Disease Burden, Treatment
             Selection and Survival Outcomes},
  author  = {Tun, Hein Minn and Naing, Lin and Malik, Owais Ahmed
             and Abdullah, Muhammad Syafiq and Ta, Thu
             and Abdul Rahman, Hanif},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2026},
  doi     = {10.1007/s10278-026-02224-3}
}
```

## Ethical Use and Disclaimer

This repository supports academic research into deep learning for prostate cancer imaging. The models are experimental research tools and are **not clinically validated medical devices**. Do not use their outputs as standalone recommendations for treatment selection, prognosis, or patient management.

All applications involving patient data must comply with applicable ethical approvals, data protection requirements, institutional policies, and relevant clinical governance procedures.

## Acknowledgements

Universiti Brunei Darussalam supported this research through research grant **UBD/RSCH/1.6/FICBF/2025/008**. We acknowledge the Brunei Cancer Centre, Jerudong Park Medical Centre, the Ministry of Health, Brunei Darussalam, and the Medical and Health Research and Ethics Committee for their support.

---

*For questions about the methodology or research collaboration, please refer to the corresponding author information in the published article.*

