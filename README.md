
# ML Alzheimer's Project

[![GitHub license](https://img.shields.io/badge/license-SJSU-blue.svg)](LICENSE)

> **Machine Learning for Alzheimer's Disease Analysis**

---

## Table of Contents
- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Project Structure](#project-structure)
- [Contact](#contact)
- [How to Run](#how-to-run)
---

## Overview

This repository hosts a project that combines machine learning with bioinformatics to investigate potential biomarkers for Alzheimer's disease. By integrating robust software engineering practices with advanced bioinformatic techniques, we have built a reproducible pipeline to analyze biomedical datasets, preprocess the data, train models, and validate findings using tools such as BLAST.

---

## Project Motivation

Alzheimer's disease continues to challenge the medical community in diagnosis and treatment. Our project is designed to:

- **Identify Biomarkers:** Utilize classification models to distinguish between Alzheimer's-related genetic sequences and non-related sequences.
- **Validation Integration:** Run BLAST searches to align unknown sequences with known Alzheimer's genes.
- **Showcase Engineering Excellence:** Implement best coding practices including modularity, documentation, testing, and scalability.

This work bridges cutting-edge machine learning methods with practical software engineering to provide insights into one of the most complex neurodegenerative disorders.

---

## Project Structure

```plaintext
123a/
├── data/
│   ├── GDS4758.soft
│   ├── oasis_cross-sectional.csv 
│   └── .DS_Store
├── docs/
│   ├── Propsoal
│   ├── Presnetation
├── src/
│   ├── phenotype/
│   │   └── phenotype_pipeline.py
│   └── 123a gene sequence anyalsis.py
├── README.md
├── requirements.txt
└── .gitignore


```

## How to Run

- **Phenotype:**  
  1. Download (or move) `oasis_cross-sectional.csv` into the project root.  
  2. Install the Python requirements:  
     ```bash
     pip install -r requirements.txt
     ```  
  3. Execute the pipeline:  
     ```bash
     python src/phenotype/phenotype_pipeline.py
     ```  
  4. After it finishes you’ll find three output plots in the root folder:  
     - `oasis_dementia_donut.png`  
     - `age_vs_mmse.png`  
     - `decision_tree.png`  

- **Gene Sequencing:**
  	There is two important files to run one is the data file GDS4758.soft which stores all the data for gene sequencing. By saving this file 	in the same folder as the 123a gene sequence anyalsis.py and running the application should work and give you all the anyalsis done for 	the gene sequencing.

---

## Contact
•	Nick Tran: nick.tran@sjsu.com
•	George Wilfert: george.wilfert@sjsu.com

