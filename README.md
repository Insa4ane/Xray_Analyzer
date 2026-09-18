# 🫁 X-Ray Analyzer

An application for automatically analyzing chest X-ray images for signs of **pneumonia**, using a convolutional neural network (CNN) as a feature extractor combined with an **XGBoost** classifier for the final prediction. The project exposes a REST API (FastAPI) and a simple web interface (Streamlit) for uploading images and viewing the results.

## Table of contents

- [How it works](#how-it-works)
- [Project architecture](#project-architecture)
- [Requirements](#requirements)
- [Running with Docker Compose (recommended)](#running-with-docker-compose-recommended)
- [Running locally (without Docker)](#running-locally-without-docker)
- [Configuration (.env)](#configuration-env)
- [Training the model](#training-the-model)
- [Using the API](#using-the-api)
- [Tests](#tests)
- [Repository structure](#repository-structure)

## How it works

1. **Data** — X-ray images are automatically downloaded from the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset using `kagglehub`.
2. **Preprocessing** — each image is converted to grayscale and resized to `224x224` (`DataLoader/Loader.py`).
3. **CNN model** (`agent/XrayAgent.py`) — a convolutional neural network (PyTorch) acts as both a feature extractor and a base classifier, trained while accounting for class imbalance (`pos_weight` in `BCEWithLogitsLoss`).
4. **XGBoost model** (`agent/XGBoostAgent.py`) — features extracted by the CNN are fed into an `XGBClassifier`, which produces the final prediction along with a confidence score.
5. **API** (`server/api.py`, `server/server.py`) — a FastAPI endpoint accepts images, runs them through both models, and returns the result (`is_healthy`, `confidence`).
6. **Frontend** (`frontend/`) — a Streamlit app lets you upload images, send them to the API, and view the analysis results.

## Project architecture

```
Streamlit (frontend) ──HTTP──▶ FastAPI (server/api.py) ──▶ XrayAgent (CNN) ──▶ XGBoostAgent ──▶ result
                                                                    ▲
                                                                    │
                                                    train/train.py (model training)
```

## Requirements

- Docker + Docker Compose **or** Python 3.12
- A Kaggle account (to download the dataset via `kagglehub`)
- A GPU is optional — the code automatically detects CUDA availability and falls back to CPU if unavailable

## Running with Docker Compose (recommended)

1. Create a `.env` file in the project's root directory (see the [configuration section](#configuration-env)).
2. Build and start all services:

   ```bash
   docker compose up --build
   ```

   Compose will start three services in sequence:

   | Service    | Description                                                                                     | Port   |
   |------------|--------------------------------------------------------------------------------------------------|--------|
   | `train`    | Trains the CNN and XGBoost models if they don't already exist in the `Model_CNN`/`Model_XGB` volumes | —      |
   | `api`      | Starts the FastAPI server with the prediction endpoint                                            | `5000` |
   | `frontend` | Starts the Streamlit interface                                                                    | `8501` |

3. Once training finishes (the `train` service completes successfully), `api` will start, followed by `frontend`.
4. Open your browser at `http://localhost:8501` to upload an X-ray image and see the result.

> Models and training history are stored in named Docker volumes (`models_cnn`, `models_xgb`, `history`, `kaggle_cache`), so running `docker compose up` again won't retrain the models from scratch.

## Running locally (without Docker)

1. Create and activate a virtual environment, then install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set up the `.env` file (see below).
3. Train the models (if you don't already have them):

   ```bash
   python -m train.train
   ```

4. Start the API:

   ```bash
   uvicorn server.api:app --host 0.0.0.0 --port 5000
   ```

5. In a separate terminal, start the frontend:

   ```bash
   streamlit run frontend/main_front.py
   ```

## Configuration (.env)

The project reads its configuration from environment variables via `python-dotenv` (`config/config.py`). Example `.env` file:

```env
# Kaggle API key required to download the dataset via kagglehub
API_KEY=your_kaggle_key

# Backend address the Streamlit frontend uses to reach the API
# (automatically set to http://api:5000/ in Docker Compose)
BACKEND_URL=http://localhost:5000/
```

Other model parameters (image size, number of epochs, model save paths, etc.) are defined in `config/config.py` and can be adjusted directly in the code.

## Training the model

The `train/train.py` script:

1. Downloads the `chest-xray-pneumonia` dataset from Kaggle.
2. Splits the data into training and test sets (`DataLoader.split_data`).
3. Trains the CNN (`XrayAgent`) with loss-based early stopping.
4. Saves the model weights (`Model_CNN/xray_cnn_model.pth`) and training metric plots (`History/plots/`).
5. Trains an XGBoost classifier on features extracted by the CNN (`Model_XGB/xgb_model.joblib`).
6. Saves the training history and evaluation results as JSON files in the `History/` directory.

To run it:

```bash
python -m train.train
```

## Using the API

### `POST /make_predict`

Accepts one or more image files (`multipart/form-data`, field `files`) and returns a list of predictions.

**Example (curl):**

```bash
curl -X POST "http://localhost:5000/make_predict" \
  -F "files=@path/to/image1.jpg" \
  -F "files=@path/to/image2.jpg"
```

**Example response:**

```json
[
  {
    "is_healthy": true,
    "confidence": 0.94
  },
  {
    "is_healthy": false,
    "confidence": 0.88
  }
]
```

If the models haven't been trained/loaded yet, the API returns a `503 Model is not available` error.

## Tests

The project includes a set of unit tests in the `test/` directory (covering the API, `DataLoader`, the CNN/XGBoost agents, and the training process, among others). Run them with:

```bash
pytest
```

## Repository structure

```
Xray_Analyzer-main/
├── agent/              # CNN model (XrayAgent) and XGBoost classifier (XGBoostAgent)
├── config/              # Application configuration (environment variables, hyperparameters)
├── DataLoader/          # Downloading and preprocessing data from the chest-xray-pneumonia dataset
├── frontend/            # Streamlit-based user interface
├── server/              # FastAPI-based API
├── train/               # Model training script
├── utils/               # Helper utilities (e.g. training metric plots)
├── test/                # Unit tests (pytest)
├── docker-compose.yml   # Service orchestration: train / api / frontend
├── Dockerfile           # Base image for all services
└── requirements.txt     # Python dependencies
```

## Notes

- In this project, the CNN model plays a dual role: it is trained on its own as a binary classifier, while its `feature_extractor` layer is also used as a feature extractor for XGBoost.
- Class imbalance in the dataset is compensated for using the `pos_weight` parameter in the CNN's loss function.
- Images uploaded via the API are temporarily saved to disk (`tempfile.TemporaryDirectory`) and processed the same way as training data, to keep preprocessing consistent.
