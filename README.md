# RoleSense - AI Resume Screening System

An intelligent resume screening and matching application built with Streamlit, scikit-learn, and natural language processing techniques. This system helps recruiters and hiring managers efficiently process resumes, match them to job descriptions, and identify the most suitable candidates.

![AI Resume Screening System](https://img.shields.io/badge/AI-Resume%20Screening-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)

## Features

- **Resume Classification**: Automatically categorize resumes into 40+ job categories using machine learning
- **Job Description Matching**: Match resumes to job descriptions using TF-IDF and cosine similarity
- **Hybrid Scoring**: Combine ML classification and similarity scoring for comprehensive candidate evaluation
- **Batch Processing**: Upload and process multiple resumes simultaneously
- **Interactive Visualizations**: View results through intuitive charts and graphs
- **Detailed Analysis**: Get in-depth insights for each resume with pass/fail status and confidence scores
- **Multiple File Formats**: Support for PDF, DOCX, and TXT resume formats

## Project Structure

This repo ships **code only** — the dataset, trained models, and vectorizers are not committed (see `.gitignore`). They're generated locally by running the two notebooks in order. This keeps the repo small and avoids Git LFS entirely.

```
ResumeScreening/
├── app.py                         # Main Streamlit application
├── requirements.txt
├── requirements-dev.txt           # Adds Jupyter/IPython for running the notebooks
├── .env.example                   # Template for NGROK_AUTH_TOKEN
├── cleaning_dataset.ipynb         # Step 1: clean the raw dataset
├── feature_engineering_and_model_training.ipynb         # Step 2: feature engineering + model training
├── run_app_with_ngrok.ipynb         # Optional: launch the app from a notebook (with ngrok tunnel)
├── sample_resumes/                # Sample resume files for manual testing
│
│  # Everything below is generated locally, not committed:
├── data/
│   ├── processed/                 # cleaned + feature-engineered datasets
│   └── arrays/                    # cached train/test vectors
├── models/                        # tfidf_vectorizer.pkl, count_vectorizer.pkl,
│                                   # label_encoder.pkl, best_model_random_forest.pkl
├── results/model_comparison_results.csv
└── logs/streamlit.log
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/durgeshgowdac/ResumeScreening.git
   cd ResumeScreening
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you'll be running the notebooks (steps 5 below) rather than just the app, install the dev extras instead — it pulls in `requirements.txt` plus IPython/Jupyter kernel support:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Get the raw dataset. Place it at `data/raw/train.csv` — the source used originally is [`ahmedheakl/resume-atlas`](https://huggingface.co/datasets/ahmedheakl/resume-atlas) on Hugging Face.

5. Generate the models and processed data by running the notebooks **in this order**:
   1. `cleaning_dataset.ipynb` — cleans `data/raw/train.csv`, writes `data/processed/resume_data_cleaned.{csv,parquet}`
   2. `feature_engineering_and_model_training.ipynb` — vectorizes, trains and compares models, writes everything under `models/`, `data/processed/`, `data/arrays/`, and `results/`

   Both notebooks create their own output folders automatically (`os.makedirs(..., exist_ok=True)`), so no manual folder setup is needed.

6. (Optional) If you plan to use the ngrok tunnel cell in `run_app_with_ngrok.ipynb`, copy `.env.example` to `.env` and fill in your own token:
   ```bash
   cp .env.example .env
   ```

## Usage

Run the app directly:
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

1. Choose your data source:
   - Upload resume files (PDF, DOCX, TXT)
   - Upload a CSV file with resume data
   - Use the pre-loaded dataset generated in Setup step 5
2. Enter a job description in the text area
3. Configure matching parameters:
   - Choose Semantic Similarity or Hybrid Matching
   - Adjust the pass/fail threshold
   - Set weights (α/β) for hybrid scoring, if available
4. View results, drill into per-resume breakdowns, and export as CSV

## Machine Learning Models

The system compares several classifiers during training (Naive Bayes, Logistic Regression, SVM, Random Forest, KNN) and keeps the best performer:

- **Random Forest Classifier**: best performing model in local testing, ~82% accuracy
- **TF-IDF Vectorization**: for text feature extraction and similarity matching
- **Cosine Similarity**: for matching resumes to job descriptions

Exact numbers depend on your run of the dataset — check `results/model_comparison_results.csv` after training.

## Data Processing Pipeline

1. **Cleaning** (`cleaning_dataset.ipynb`): strip emails/phones/URLs, normalize category names, drop short/duplicate entries
2. **Feature Engineering** (`feature_engineering_and_model_training.ipynb`): TF-IDF + Count vectorization, handcrafted NLP features (education/experience/skill keyword counts, etc.)
3. **Model Training**: train/compare classifiers, save the best one
4. **Inference** (`app.py`): extract text from uploaded resumes → vectorize → similarity and/or classifier scoring → pass/fail + visualizations

## Requirements

See `requirements.txt` (app + notebooks) and `requirements-dev.txt` (adds Jupyter/IPython kernel support for running the notebooks interactively). Notable dependencies:
- Streamlit, pandas, scikit-learn, numpy
- pypdf, python-docx (resume text extraction)
- plotly, matplotlib, seaborn, wordcloud (visualization/EDA)
- pyngrok, python-dotenv, watchdog (optional notebook-based launch with a public tunnel)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

- [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) dataset used for training
- Streamlit for the interactive web application framework
- scikit-learn for machine learning algorithms and tools
