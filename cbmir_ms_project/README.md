#Deployed link
https://saksham937-major-project-cbmir-ms-projectapp-mcgt9e.streamlit.app

# Multiple Sclerosis Identification via CBMIR 🧠

This project implements a complete end-to-end **Content-Based Medical Image Retrieval (CBMIR)** system designed to assist in identifying Multiple Sclerosis (MS) lesions. Given a query MRI brain scan, the system retrieves visually and pathologically similar cases from a database, aiding clinicians in diagnosis and analysis.

---

## 🚀 Features

- **NIfTI Processing**: Robust handling, intensity normalization, and axial slice extraction of 3D NIfTI neuroimages.
- **Deep Feature Extraction**: Utilizes transfer learning with ResNet-50 (or EfficientNet-B0) to generate rich, dense feature embeddings (2048-dimensional vectors) from MRI slices.
- **Lightning Fast Retrieval**: Employs **FAISS** (Facebook AI Similarity Search) to index large datasets and perform ultra-fast cosine similarity lookups.
- **Modern Interactive Dashboard**: Built with **Streamlit**, featuring a smooth, dark-mode inspired UI for uploading scans, exploring slices on the Z-axis, and viewing top-K retrieved cases instantly.
- **Lesion Awareness**: Uses ground-truth masks from the ISBI dataset to track and verify the presence of MS lesions in retrieved images.

---

## 🏗️ Project Architecture

```
cbmir_ms_project/
│
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/                       # Directory for raw and processed datasets
│   ├── dummy_isbi/             # Auto-generated dummy ISBI dataset for demo
│   └── processed/              # Extracted 2D JPEGs and FAISS index files
│
├── features/                   # Indexing logic
│   └── indexer.py              # FAISS wrapper for vector indexing and search
│
├── models/                     # Deep learning models
│   └── feature_extractor.py    # PyTorch ResNet-50 feature extractor
│
├── notebooks/                  # Pipeline notebooks and scripts
│   └── build_index.py          # Script to process data and build the FAISS index
│
└── utils/                      # Helper utilities
    ├── data_processor.py       # NIfTI loading, normalization, and slicing
    ├── evaluation.py           # Metrics (Precision@K, Recall@K, mAP)
    └── generate_dummy_data.py  # Generates a pseudo MRI dataset for testing
```

---

## 🛠️ Setup and Installation

### 1. Requirements

Ensure you have Python 3.9+ installed.

```bash
git clone <repository_url>
cd cbmir_ms_project
pip install -r requirements.txt
```

### 2. Dataset Preparation

If you do not have the original **ISBI MS Lesion Segmentation** dataset, you can generate a high-quality dummy dataset to test the pipeline right away.

```bash
python utils/generate_dummy_data.py
```
*This will create synthetic NIfTI volumes with simulated lesions in `data/dummy_isbi/`.*

If you have the real dataset, place the patient folders (e.g., `training01`, `training02`) inside the `data/isbi_dataset/` directory and update the path in `notebooks/build_index.py`.

### 3. Build the FAISS Vector Index

Before running the web app, you must process the 3D scans, extract 2D axial slices, generate deep feature embeddings, and build the FAISS index. Run:

```bash
python notebooks/build_index.py
```
*This script uses ResNet-50 to extract features and saves the `.faiss` index inside `data/processed/index/`.*

### 4. Run the Streamlit Dashboard

Launch the interactive CBMIR application:

```bash
streamlit run app.py
```

Upload a NIfTI file (you can use one of the files from `data/dummy_isbi/training01/flair.nii.gz`), select a slice, and click **"Find Similar Cases"**.

---

## 📊 Evaluation Metrics

The system includes utilities to evaluate retrieval performance in `utils/evaluation.py`. If ground truth category labels (e.g., Has Lesion vs. No Lesion) are known, the system can compute:

* **Precision@K**: Fraction of retrieved images in the top-K that are relevant.
* **Recall@K**: Fraction of all relevant images successfully retrieved in the top-K.
* **mAP (mean Average Precision)**: The mean of average precisions over multiple queries, providing a comprehensive metric of ranking quality.

---

## 💻 Technology Stack

* **Deep Learning Framework:** PyTorch & Torchvision
* **Image Processing:** OpenCV, Pillow, SciPy
* **Medical Imaging Format:** NiBabel
* **Vector Database:** FAISS (CPU)
* **Web Frontend:** Streamlit
* **Data Science:** NumPy, scikit-learn, Matplotlib

---

## 🎓 Academic Viva/Demo Notes

- **"Why FAISS?"** It allows similarity search to scale to millions of feature embeddings efficiently. 
- **"Why ResNet for Feature Extraction?"** A pretrained ResNet on ImageNet contains robust edge and texture filters that translate remarkably well to identifying morphological traits in MRIs. Fine-tuning the network (not included in the base script) would further improve task-specific retrieval.
- **"What does Cosine Similarity signify here?"** It measures the cosine of the angle between two 2048-dimensional feature vectors, capturing structural and contextual similarity independent of magnitude differences in the feature activations.

---

*Prepared for Final Year Project Submission.*

---

## 🚀 Deployment (GitHub & Streamlit)

### Push to GitHub

1. Create a new repository on [GitHub](https://github.com/new).
2. Link your local project to the remote repository and push:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

### Deploy to Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy!** 

*Note: The project's `requirements.txt` has been optimized (`opencv-python-headless`) for cloud deployment, and `.gitignore` has been configured to keep the repository clean while preserving the FAISS index required for the app to run.*
