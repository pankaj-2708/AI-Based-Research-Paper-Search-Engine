# AI-Based-Research-Paper-Search-Engine

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project implements an AI-based research paper search engine. It uses **SPECTER2 (by AllenAI)** to generate dense vector representations of research papers. Users can describe their query in natural language, and the system retrieves the most relevant papers from the corpus based on semantic similarity.

> **Note:**  
> The corpus consists of **540,000 research papers from arXiv**. The model may fail to retrieve some relevant papers if they are **not present in this corpus**.

---

## Project Workflow

1. **Paper Encoding**  
   All research papers from the corpus (540K papers) are encoded into **768-dimensional embeddings** using **SPECTER2 (AllenAI)**.

2. **Index Construction**  
   The generated embeddings are used to build a **FAISS index** using the **IndexIVFFlat** method for efficient similarity search.

3. **Query Encoding**  
   The user’s natural-language query is encoded into a vector using the **same SPECTER2 model**.

4. **Retrieval**  
   The system retrieves the **Top-K closest embeddings** from the FAISS index and returns the corresponding **paper titles and arXiv links** to the user.

## Tech Stack

1. **Hugging Face**  
   Used for loading and running the pretrained **SPECTER2** model.

2. **FAISS**  
   Enables efficient similarity search over the research paper corpus.

3. **AWS S3**  
   Used to store the FAISS index and the corresponding metadata/dataframes.

4. **Streamlit**  
   Provides the frontend interface for user interaction.

5. **FastAPI**  
   Serves as the backend API to handle query processing and retrieval.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

