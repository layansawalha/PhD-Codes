{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Hybrid Multimodal and Quantum Machine Learning Architectures\
\
This repository contains the official code, models and experimental setups for my PhD research. The methodology investigates multimodal and hybrid fusion architectures and their effect on classification and prediction performance across medical, scientific and construction datasets[cite: 11].\
\
## Overview of Studies\
\
The research is hypothesis-driven and divided into four core studies[cite: 11]:\
\
*   **Study 1: Quantum & Classical SVM Fusion (Medical Tabular Data)** \
    It is hypothesised that the fusion of a Quantum Support Vector Machine with a classical Support Vector Machine through ensemble methods will improve breast cancer classification accuracy beyond that of either paradigm in isolation[cite: 11]. Evaluated on the WBCD and BCCD datasets[cite: 11].\
*   **Study 2: Mid-Level Multimodal Fusion (Medical Imaging & Text)**\
    It is hypothesised that the mid-level fusion of BERT, GPT-2 and ResNet18 representations will outperform single-modality baselines on combined clinical text and ultrasound image data[cite: 11]. Evaluated on the Breast Lesions Ultrasound (USG) Collection[cite: 11].\
*   **Study 3: Multimodal Scientific PDF Classification**\
    It is hypothesised that the multimodal fusion of BERT, GPT-2 and a ResNet-18 vision encoder over document-extracted images will outperform classical, ensemble and single-transformer baselines on full-text scientific PDF classification[cite: 11]. Evaluated on the NUS Keyphrase Extraction Corpus[cite: 11].\
*   **Study 4: Advanced Regression Fusion Strategies (Construction Costs)**\
    It is hypothesised that three distinct fusion strategies (quantum-enhanced regression, transformer-boosted regression and neural-meta-learner ensemble fusion) will outperform traditional regressors on UK construction cost prediction[cite: 11]. Evaluated on the BCIS Construction Cost Dataset[cite: 11].\
\
## Experimental Design & Validation\
\
This codebase adopts a post-positivist quantitative experimental design[cite: 11]. To ensure rigorous validation and prevent data leakage, the following protocols are enforced across the repository:\
*   **Strict Partitioning:** Preprocessing transformations (e.g. Z-score standardisation, TF-IDF vectorisation and PCA) are fitted exclusively on training partitions[cite: 11].\
*   **Evaluation Metrics:** Models are evaluated using accuracy, precision, recall and F1-score for classification tasks[cite: 11]. Regression tasks are evaluated using MAE, RMSE and R\'b2[cite: 11].\
*   **Reproducibility:** A fundamental random seed of `42` is fixed across all cross-validation splits, model initialisations and training loops to ensure exact reproducibility[cite: 11].\
\
## Installation and Requirements\
\
This project requires **Python 3.10** and relies on specific library versions to ensure deterministic behaviour across PyTorch, TensorFlow and Qiskit environments[cite: 11]. \
\
To replicate the environment, run:\
```bash\
pip install -r requirements.txt}