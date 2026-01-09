# ROOT Object Performance Analysis (`root-obj-perf`)

## Overview
This repository contains a Python-based framework for evaluating the performance of reconstructed physics objects—specifically jets—within the context of Di-Higgs to 4b ($HH \to 4b$) analyses. It utilizes the "Scikit-HEP" ecosystem (`uproot`, `awkward`, `vector`) to process ROOT files efficiently without relying on the ROOT C++ framework.

The primary goal of this analysis is to compare the performance of different jet reconstruction algorithms (Offline vs. Level-1 Trigger jets) by evaluating:
* **Reconstruction Efficiency & Purity**
* **B-Tagging Performance** (ROC curves, operating points)
* **Jet Energy Scale & Resolution** (Response curves)
* **Di-Higgs Reconstruction** (Pairing efficiency and mass reconstruction)

## Repository Structure

### Core Analysis
* **`read-data.ipynb`**: The main driver of the analysis. It is a Jupyter Notebook that executes the full workflow: loading data, applying selections, calculating metrics, and generating performance plots.

### Configuration
* **`hh-bbbb-obj-config.json`**: A JSON configuration file that centralizes all analysis parameters. It defines:
    * Input file paths and tree names.
    * Jet collection names (e.g., Offline, L1NG, L1Ext).
    * Tagging algorithms and cut values (working points).
    * Kinematic cuts ($p_T$, $\eta$) and matching criteria ($\Delta R$).

### Helper Modules
* **`data_loading_helpers.py`**: Handles I/O operations.
    * Loads ROOT trees using `uproot`.
    * Restructures flat TTree branches into jagged `awkward` arrays.
    * Constructs 4-vector objects (using the `vector` library) for easy kinematic manipulation.
    * Applies Jet Energy Corrections (JEC) and regressions (e.g., PNet, UParT).
* **`analysis_helpers.py`**: Contains the core physics logic.
    * **Matching**: Algorithms to match Generated particles to Reconstructed jets (DeltaR matching, Hungarian algorithm for 1-to-1 uniqueness).
    * **Metrics**: Functions to calculate efficiency, purity, and ROC curve points (TPR/FPR).
* **`plotting_helpers.py`**: A suite of `matplotlib`-based visualization tools.
    * 1D Histograms (Kinematic distributions).
    * ROC Curves (B-tagging performance).
    * 2D Heatmaps (Response maps, efficiency maps).
    * Resolution Plots (Mean and width of response distributions).

## Dependencies
The framework relies on the modern Python HEP stack:
* `python >= 3.8`
* `uproot`: For reading ROOT files.
* `awkward`: For manipulating jagged arrays.
* `vector`: For 4-vector physics math.
* `numpy` & `scipy`: For numerical operations and fitting.
* `matplotlib` & `seaborn`: For plotting.
* `scikit-learn`: For ROC curve calculation (`roc_curve`, `auc`).

## Setup & Usage

1.  **Configure the Analysis**:
    Open `hh-bbbb-obj-config.json` and ensure the `file_pattern` points to your local ROOT data files. Adjust cuts and collection names as necessary.

    ```json
    {
        "file_pattern": "/path/to/data/hh4b/data_*.root",
        "offline": { "tagger_name": "btagPNetB", ... },
        ...
    }
    ```

2.  **Run the Analysis**:
    Open `read-data.ipynb` in a Jupyter environment. The notebook is structured sequentially:
    * **Imports & Config**: Loads libraries and the JSON config.
    * **Data Loading**: Reads events and applies corrections (e.g., PNet/UParT regression).
    * **Gen Selection**: Filters for Gen-level b-quarks originating from Higgs decays.
    * **Matching**: Matches Gen b-quarks to Offline and L1 jets.
    * **Plotting**: Generates efficiency plots, ROC curves, and resolution histograms.

## Analysis Details

The notebook performs a deep dive into object performance through several distinct stages:

### 1. Object Matching & Selection
* **Gen-Matching**: B-quarks from Higgs decays are identified by tracing PDG IDs (`pdgId == 5` from `pdgId == 25`).
* **Jet Matching**: Reconstructed jets are matched to Gen b-quarks using a $\Delta R < 0.4$ cone. The code supports both simple "closest-neighbor" matching and "cross-matching" (mutual nearest neighbors) to ensure purity.

### 2. B-Tagging Performance
* **ROC Curves**: Compares the discrimination power of various taggers (e.g., PNet vs. UParT vs. L1 algorithms).
* **2D Performance**: Analyzes tagging performance in bins of $p_T$ and $\eta$ to identify regions of inefficiency.
* **Working Point Optimization**: Contains logic to scan thresholds and find cuts that yield specific signal efficiencies (e.g., 70%).

### 3. Jet Energy Scale & Resolution (JES/JER)
* **Response**: Calculates $p_{T}^{Reco} / p_{T}^{Gen}$.
* **Calibration**: Applies regression corrections (like `PNetRegPtRawCorr`) to raw jet momenta.
* **Resolution Plots**: Fits the response distribution with Gaussian models to extract the energy scale bias (mean) and resolution (width/$\sigma$) as a function of $p_T$ and $\eta$.

### 4. Di-Higgs ($HH \to 4b$) Reconstruction
* **Pairing Algorithms**: Implements logic to group the 4 leading jets into two Higgs candidates.
* **Metric**: Minimizes the distance parameter $D_{HH}$ to find the best jet pairings:
    $$D_{HH} = \frac{|m_{h1} - \frac{125}{120}m_{h2}|}{\sqrt{1 + (125/120)^2}}$$
* **Mass Reconstruction**: Plots the reconstructed mass of the leading vs. subleading Higgs candidates and visualizes the signal region ellipse.

## Key Files

| File | Description |
| :--- | :--- |
| `read-data.ipynb` | Main analysis notebook. |
| `hh-bbbb-obj-config.json` | Global configuration file. |
| `analysis_helpers.py` | Physics algorithms (matching, purity/efficiency). |
| `plotting_helpers.py` | Visualization library. |
| `data_loading_helpers.py` | Data ingestion and 4-vector construction. |
