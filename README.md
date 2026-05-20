# NiTi-AM CrystalFunction Map

**NiTi-AM CrystalFunction Map** is a Streamlit-based research platform for **laser powder bed fusion (LPBF / PBF-LB/M) of NiTi shape-memory alloys**.

The purpose of the project is not only to calculate **volumetric energy density**. The goal is to connect the full chain between processing, metallurgical state, crystallography, and functional performance:

```text
LPBF parameters
→ melt-pool stability
→ defect / cracking / keyhole risk
→ Ni evaporation and composition shift
→ B2 ↔ B19′ transformation window
→ superelastic or shape-memory suitability
→ literature benchmark comparison
→ ML-assisted prediction when enough real data exist
→ SEM / XRD / EBSD result analysis

The central question is:

Can we print NiTi while preserving the crystallographic and functional conditions required for useful superelastic or shape-memory behavior?

Why this project exists

NiTi is not a normal printable alloy where density alone is enough. In LPBF NiTi, a part can look dense but still fail functionally if:

Ni evaporation shifts the composition,
the transformation temperatures move outside the service window,
the material becomes martensitic at room temperature,
keyhole defects or cracks reduce recoverable strain,
oxygen and secondary phases alter the transformation behavior,
the B2/B19′ balance is no longer suitable for the target function.

For that reason, the app treats LPBF NiTi as a process–crystallography–function problem.

What the app does
1. Process–Function Screening

The user can enter:

laser power
scan speed
hatch spacing
layer thickness
powder Ni at.%
measured Ni at.% if available
oxygen level
remelting / rescanning passes
heat treatment temperature and time
service temperature
target function: superelastic or thermal actuation

The app then calculates:

volumetric energy density
linear energy density
lack-of-fusion risk
cracking risk
keyhole risk
Ni-loss risk
estimated effective Ni content
estimated Mf / Ms / As / Af
functional interpretation for the chosen service temperature

The output is not presented as a final material qualification. It is a screening result that tells the user what should be checked experimentally.

2. Metallurgical Scenario Engine

The app classifies the selected process condition into readable metallurgical scenarios.

Examples:

Low-energy discontinuity / cracking risk

At too low energy density, the melt pool may not provide enough bonding. In this region, lack of fusion, interlaminar cracks, or discontinuous tracks can dominate before shape-memory behavior is even meaningful.

Functional low-energy superelastic window

A moderate energy window can provide enough bonding while limiting excessive Ni evaporation. This is the most promising region for room-temperature superelasticity in Ni-rich LPBF NiTi.

Transition zone: keyhole / partial superelasticity

At higher energy, melting may improve, but keyhole instability and local Ni loss can begin to reduce the stability of the functional response.

High-energy Ni-loss / martensite-at-service risk

At high energy input, Ni evaporation can become strong. Since NiTi transformation temperatures are highly sensitive to Ni/Ti ratio, Ni loss can raise transformation temperatures and make the material martensitic at service temperature.

3. Transformation Window

The app visualizes the transformation interval using:

Mf → Ms    martensite formation on cooling
As → Af    austenite recovery on heating

It also marks the selected service temperature.

This helps the user understand whether the material is likely to be:

austenitic at service temperature,
martensitic at service temperature,
or inside a mixed / unstable transformation range.

For superelastic design, the material generally needs to be austenitic at the service temperature. For actuation design, the activation temperature should be close to the intended operating window.

4. Literature Benchmark Data

The project separates data into different layers.

data/exact_literature_process_rows.csv

Contains exact row-level process/function data extracted from the literature. The first benchmark set includes Ni-rich Ni51.3Ti48.7 LPBF conditions, including laser power, scan speed, hatch spacing, layer thickness, energy density, microstructural observations, and functional class.

data/paper_level_numeric_facts.csv

Contains exact paper-level numerical facts, such as transformation-temperature ranges, lattice parameters, reported property limits, ML dataset size, and benchmark values.

data/demo_training_seed.csv

Contains synthetic demonstration rows used only to test the ML interface. These rows are not evidence and should not be used for scientific claims.

data/scenario_rules.csv

Contains readable metallurgical scenario rules used by the app.

This separation is important because a small exact dataset is more defensible than a large dataset with unclear or invented values.

5. Machine-Learning Workbench

The app includes a Machine-Learning Workbench, but it does not force ML when the data are not sufficient.

The ML page works in two modes:

Functional classification

When exact row-level data are limited, the app can classify process conditions into functional classes such as:

excellent
partial
poor / not tested

This is useful when the literature gives reliable process labels but does not provide enough numerical DSC, density, or mechanical data for regression.

Regression

When enough real rows are available, the app can train regression models for targets such as:

relative density
porosity
Ms / Mf / As / Af
hysteresis
recoverable strain
residual strain
UTS
elongation

The implemented model options include:

Gaussian Process Regression
Random Forest
Extra Trees
Gradient Boosting

Gaussian Process is useful for small datasets because it can provide uncertainty. Tree-based models are useful for nonlinear process–property relationships once the dataset becomes larger.

The app also prevents fake ML: if a target does not have enough real non-null rows, the regression model is not trained. The user is told exactly what data are missing.

6. Characterization Upload Lab

The app includes a first-pass analysis section for experimental results.

SEM / optical microstructure

The user can upload a microstructure image. The app performs a simple segmentation of dark features and reports:

feature area fraction
feature count
average feature size
image texture descriptors

This is intended for quick screening of pores, cracks, pull-outs, or dark features. It is not a replacement for calibrated metallography.

XRD

The user can upload a two-column XRD file with:

two_theta_deg
intensity

The app detects peaks, estimates d-spacing, and gives first-pass phase hints.

This is useful for screening possible B2/B19′ regions or secondary-phase indicators, but final phase identification still requires proper calibration and refinement.

EBSD / TKD-style CSV

The user can upload an orientation map exported as CSV. The app can inspect:

x / y coordinates
Euler angles
phase ID
confidence index
image quality
grain ID if available

The current version is a starter layer for orientation-map inspection. A future version can extend this toward:

B2 parent reconstruction
B19′ variant indexing
misorientation histograms
variant-pair statistics
build-direction texture analysis
Crystallographic basis

The project uses reference B2/B19′ crystallographic values for NiTi:

B2 austenite:
a = 3.015 Å

B19′ martensite:
a = 2.89 Å
b = 4.12 Å
c = 4.62 Å
β = 96.8°

These values are used to structure the crystallographic part of the app and to prepare later extensions toward B2 parent reconstruction, B19′ variant analysis, and compatibility descriptors.

The crystallographic direction of the project is based on the idea that printed NiTi should not be judged only by density. The B2/B19′ transformation, orientation relationships, habit-plane logic, martensite variants, and texture can all influence the final functional behavior.

Papers and data used

The current version is built around several literature directions.

Xiang et al., 2024 — LPBF Ni-rich NiTi process window

This paper provides the main row-level benchmark currently used in the app. It studies LPBF Ni-rich Ni51.3Ti48.7 and shows that process parameters strongly affect microstructure, transformation behavior, and superelasticity.

The app uses this paper to define several process scenarios:

low-energy cracking / poor continuity,
useful low-energy superelastic window,
transition region with partial superelasticity,
high-energy Ni-loss / martensite-at-service risk.

DOI: 10.3390/met14090961

Xue et al., 2022 — defect-free LPBF NiTi

This work is used as a benchmark for the idea that high-quality LPBF NiTi requires more than density. It links superior tensile superelasticity to elimination of porosity and cracks, controlled Ni evaporation, and oxygen control.

DOI: 10.1016/j.actamat.2022.117781

Cayron, 2020 — B2→B19′ crystallography from EBSD/TKD

This work supports the crystallographic layer of the project, especially the B2/B19′ lattice parameters and the idea that EBSD/TKD can be used to understand martensitic transformation, orientation relationships, reconstructed B2 parent grains, and variant-related features.

DOI: 10.3390/cryst10070562

Li et al., 2024 — ML process optimisation for LPBF NiTi

This paper supports the machine-learning direction of the project. It reports a dataset of 195 entries from 23 publications and predicts density, ultimate tensile strength, elongation, and thermal hysteresis for LPBF NiTi.

DOI: 10.1080/17452759.2024.2364221

ML work on LPBF-NiTi phase transition temperature

The project also follows the idea that phase-transition temperature can be predicted using interpretable process/composition features and ML methods such as GRNN. This supports the future direction of predicting Ms/Mf/As/Af once enough row-level DSC data are added.

Repository structure
NiTi-AM-CrystalFunction-Map/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── exact_literature_process_rows.csv
│   ├── paper_level_numeric_facts.csv
│   ├── scenario_rules.csv
│   └── demo_training_seed.csv
│
├── assets/
│   └── demo_microstructure.png
│
├── docs/
│   ├── DATA_EXTRACTION_PROTOCOL.md
│   └── LITERATURE_DATA_AUDIT.md
│
├── pages/
│   ├── 01_Process_Function_Map.py
│   ├── 02_Literature_Benchmark_Data.py
│   ├── 03_ML_Workbench.py
│   ├── 04_Prediction_Studio.py
│   ├── 05_Characterization_Upload_Lab.py
│   ├── 06_Crystallography_Compatibility.py
│   ├── 07_Data_Extraction_Protocol.py
│   └── 08_Model_Card_and_Limits.py
│
└── src/
    ├── constants.py
    ├── data_utils.py
    ├── process_models.py
    ├── ml_models.py
    ├── crystallography.py
    ├── image_analysis.py
    ├── xrd_tools.py
    └── ebsd_tools.py
How to run

Install the required packages:

pip install -r requirements.txt

Run the app:

streamlit run app.py
Requirements
streamlit
numpy
pandas
plotly
scikit-learn
scipy
pillow
matplotlib
joblib
Current limits

This project is still a research screening platform. It does not replace experimental validation.

The app does not certify:

medical-device performance,
fatigue life,
final transformation temperatures,
final phase fractions,
defect-free quality,
EBSD/TKD parent reconstruction,
or industrial process qualification.

The outputs should be checked with:

DSC,
XRD,
SEM / optical metallography,
EBSD / TKD,
density or CT measurement,
cyclic compression/tension,
and composition measurement by EDS/ICP where possible.
Next development steps

The next useful upgrades are:

Add more exact row-level LPBF NiTi data from papers.
Digitize DSC curves and stress–strain curves where tables are not available.
Expand the ML dataset with real values for density, Af, hysteresis, UTS, elongation, recovery strain, and residual strain.
Add proper model cards for each ML target.
Add EBSD/TKD-based B2 parent reconstruction.
Add B19′ variant indexing and misorientation histograms.
Add a more rigorous crystallographic compatibility module.
Add uncertainty bands around transformation-temperature predictions.
Main warning

The app is a research screening and benchmark tool.

It is not a substitute for:

DSC
XRD
SEM
EBSD/TKD
density/porosity measurement
cyclic compression/tension
composition analysis
medical or industrial qualification

The safest use is to treat the app as a structured way to connect process parameters, literature evidence, and experimental planning.


The benchmark rows and scenario logic in this README are supported mainly by Xiang et al. for Ni-rich LPBF NiTi process/superelasticity behavior, Xue et al. for defect-free LPBF NiTi and 6% tensile superelasticity, Cayron’s EBSD/TKD paper for B2/B19′ crystallographic constants and reconstruction logic, Li et al. for the 195-entry ML optimisation direction, and the GRNN transition-temperature study for ML prediction of LPBF-NiTi transformation behavior. :contentReference[oaicite:0]{index=0}
::contentReference[oaicite:1]{index=1}
