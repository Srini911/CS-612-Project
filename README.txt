
Rational Design of Non-Harmful Hemoglobin Mutations for Enhanced Hydroxyurea Binding in Sickle Cell Anemia
===========================================================================================================

Project Structure
-----------------
This project contains two main components:

1. Simulations Pipeline: Performs molecular docking simulations using AutoDock Vina.
2. ML Pipeline: Trains and evaluates machine learning models to predict mutation-based drug suitability.

Ensure you have Python installed (preferably version 3.9+) and all necessary packages before proceeding.

Directory Structure
-------------------
docking_project/
│
├── ML Coding/
│   ├── CS612 Dataset.csv
│   ├── ml_pipeline.py
│   ├── plots/                ← Contains output plots
│   └── pipeline_results/     ← Results folder for evaluation outputs
│
├── Simulations/
│   ├── Ligand/
│   │   └── hydroxyurea.pdbqt
│   ├── Receptors/
│   │   └── *.pdbqt           ← All 50 mutated hemoglobin receptor files
│   ├── vina_output/          ← Stores docking output files
│   └── simulations_pipeline.py
│
└── README.txt                ← This File
│
└── Project Report              


====================================================
INSTRUCTIONS: 1. RUNNING THE MACHINE LEARNING PIPELINE
====================================================

Step 1: Setup Environment
-------------------------
Install required packages using pip:

pip install numpy pandas matplotlib scikit-learn scipy matplotlib-venn

Step 2: Run the ML Code
-----------------------
Navigate to the ML Coding directory:

cd "docking_project/ML Coding"

Run the pipeline:

python ml_pipeline.py

This script will:
- Load `CS612 Dataset.csv`
- Split into training and testing sets
- Train Logistic Regression and Random Forest models
- Plot accuracy, learning curves, ROC curves, confusion matrices
- Save all plots and results to `pipeline_results/` and `plots/` folders

==================================================
INSTRUCTIONS: 2. RUNNING THE SIMULATIONS PIPELINE
==================================================

Step 1: Requirements
--------------------
You must have **AutoDock Vina** installed and added to your system PATH.

Download here: https://vina.scripps.edu/downloads/

Also install Python dependencies:

pip install biopython

Step 2: Set Up Ligand and Receptor Files
----------------------------------------
Ensure all `.pdbqt` files for:
- 50 mutant receptors are placed in `Simulations/Receptors/`
- The hydroxyurea ligand is in `Simulations/Ligand/hydroxyurea.pdbqt`

Step 3: Run the Docking Pipeline
--------------------------------
Navigate to the Simulations directory:

cd "docking_project/Simulations"

Run the pipeline:

python simulations_pipeline.py

This script will:
- Automatically loop through all receptors in `Receptors/`
- Perform docking with `hydroxyurea.pdbqt`
- Save all Vina outputs in `vina_output/`

Ensure that Vina is callable via `vina` command in terminal. If not, specify the full path to Vina binary inside `simulations_pipeline.py`.

========================================================
TROUBLESHOOTING
========================================================
- If AutoDock Vina fails to run, check that the binary is correctly installed and accessible.
- For ML errors, ensure dataset file is clean and paths are correctly set in the script.
- If a folder like `pipeline_results/` or `vina_output/` does not exist, the script will create it automatically.

========================================================
CONTACT
========================================================
This project was developed as part of CS612 Bioinformatics Course Project at UMass Boston. For technical issues, consult the instructor or your project supervisor.
