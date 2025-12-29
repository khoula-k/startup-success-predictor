Here is the clean, professional README file without emojis.

---

# Startup Success Predictor

A full-stack machine learning application that predicts whether a startup will succeed or fail based on early-stage metrics (funding, location, relationships, etc.).

This project demonstrates the use of **FastAPI** for the backend, **Streamlit** for the frontend, and **Docker** for containerization, managed by the **UV** package manager.

---

## Project Architecture

* **Backend (`/backend`)**: Built with **FastAPI**. It loads a pre-trained Random Forest model (`.joblib`) and serves predictions via a REST API. It uses **UV** for ultra-fast dependency management.
* **Frontend (`/frontend`)**: Built with **Streamlit**. A user-friendly interface that collects startup data and communicates with the backend.
* **Model**: A Random Forest Classifier trained on Kaggle startup datasets.

## Tech Stack

* **Python 3.11**
* **FastAPI** (API)
* **Streamlit** (UI)
* **Scikit-Learn** (ML Model)
* **UV** (Package Manager)
* **Docker & Docker Compose**

---

## Docker Optimization & Comparison

This project explores different Docker strategies to optimize image size and security. We compared a **Simple Build** (single stage) vs. a **Multi-Stage Build** (builder pattern).

### Results

The Multi-Stage build strategy successfully reduced the image size by discarding build tools, cache, and unnecessary files from the final production image.

| Image | Simple Build Size | Multi-Stage Build Size | Reduction |
| --- | --- | --- | --- |
| **Backend** | 2.54 GB | **2.33 GB** | ~210 MB |
| **Frontend** | 1.00 GB | **783 MB** | ~217 MB |

> **Key Takeaway:** Multi-stage builds result in lighter, more secure containers by separating the *build environment* (compilers, heavy tools) from the *runtime environment*.

---

## How to Run

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Steps

1. **Clone the repository** (if you haven't already):
```bash
git clone https://github.com/khoula-k/startup-success-predictor.git
cd startup-success-predictor

```


2. **Build and Start the Containers**:
Run the following command in the project root:
```bash
docker compose up --build

```


3. **Access the Application**:
* **Frontend (UI):** Open [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501) in your browser.
* **Backend (API Docs):** Open [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs) to see the automatic Swagger documentation.


4. **Stop the App**:
Press `Ctrl + C` in your terminal or run:
```bash
docker compose down

```
