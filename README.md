# NiTi-AM CrystalFunction Map

**NiTi-AM CrystalFunction Map** is a Streamlit research platform for **laser powder bed fusion of NiTi shape-memory alloys**. It is built around one practical question:

> Can a printed NiTi part keep the crystallographic and functional conditions needed for useful superelastic or shape-memory behavior?

The project does not treat LPBF NiTi as a simple density problem. A dense part can still fail functionally if the process changes Ni content, shifts transformation temperatures, produces keyhole defects, creates cracks, or moves the material into the wrong B2/B19′ state.

The app connects:

```text
LPBF / PBF-LB/M parameters
→ melt-pool stability
→ lack-of-fusion, cracking, and keyhole risk
→ Ni evaporation and composition shift
→ B2 ↔ B19′ transformation window
→ superelastic or shape-memory suitability
→ comparison with literature benchmark cases
→ ML-based prediction when enough real data are available
→ SEM, XRD, and EBSD-style result analysis
```

---

## Project aim

The aim is to build a transparent benchmark tool for LPBF NiTi. The app is meant to help researchers compare process conditions, understand functional risks, organize literature data, and plan the next experimental checks.

The central idea is that NiTi should be judged by the full chain:

```text
process → chemistry → microstructure → crystallography → transformation → function
```

not by energy density or density alone.

---

## Why NiTi needs this type of tool

NiTi is very sensitive to processing because the shape-memory and superelastic response depend strongly on the Ni/Ti ratio, phase state, transformation temperatures, and defect population.

In LPBF NiTi, a process window may look acceptable from a printing point of view but still be poor for functional performance. Typical problems include:

- Ni evaporation during high-energy processing;
- shift of Ms, Mf, As, and Af outside the service window;
- martensite present at the wrong temperature;
- lack-of-fusion defects at low energy input;
- keyhole pores and local composition changes at high energy input;
- cracking or interlayer discontinuity;
- secondary phases and oxide-related effects;
- unstable cyclic recovery or high residual strain.

For this reason, the app is structured as a **process–crystallography–function map**.

---

## Current app modules

### 1. Process–Function Screening

The main page lets the user enter a new LPBF NiTi process condition:

- laser power;
- scan speed;
- hatch spacing;
- layer thickness;
- powder Ni content;
- measured Ni content, if available;
- oxygen level;
- remelting or rescanning passes;
- heat treatment temperature and time;
- service temperature;
- target function: superelasticity or thermal actuation.

The app calculates:

- volumetric energy density;
- linear energy density;
- lack-of-fusion risk;
- cracking risk;
- keyhole risk;
- Ni-loss risk;
- estimated effective Ni content;
- estimated Mf, Ms, As, and Af;
- a functional interpretation for the selected service temperature.

The output is a screening result. It does not replace experimental validation.

---

### 2. Metallurgical scenario engine

The app assigns the selected process condition to a readable metallurgical scenario. These scenarios are based on the logic observed in LPBF NiTi literature.

The current scenarios are:

#### Low-energy discontinuity / cracking risk

At low energy input, the melt pool may not provide enough bonding. Lack of fusion, interlaminar cracking, or discontinuous tracks can dominate before functional NiTi behavior becomes meaningful.

#### Functional low-energy superelastic window

A moderate energy window can give enough bonding while limiting excessive Ni evaporation. This is the most promising region for room-temperature superelasticity in Ni-rich LPBF NiTi.

#### Transition zone: keyhole / partial superelasticity

At higher energy input, melting may improve, but keyhole instability, local Ni loss, and secondary-phase effects can begin to reduce functional recovery.

#### High-energy Ni-loss / martensite-at-service risk

At high energy input, Ni evaporation becomes a major risk. Since NiTi transformation temperatures are highly sensitive to Ni content, Ni loss can raise Ms and Af and make the material martensitic at service temperature.

---

### 3. Transformation window

The app plots the transformation interval using:

```text
Mf → Ms    martensite formation on cooling
As → Af    austenite recovery on heating
```

The selected service temperature is plotted on the same chart.

This helps the user see whether the material is likely to be:

- austenitic at service temperature;
- martensitic at service temperature;
- inside a mixed or unstable transformation interval.

For superelastic design, the material usually needs to be austenitic at the service temperature. For thermal actuation, Af should be close to the intended activation temperature.

---

### 4. Benchmark data

The project separates the data into clear layers.

```text
data/exact_literature_process_rows.csv
```

Exact row-level process data extracted from literature. The current benchmark set includes Ni-rich Ni51.3Ti48.7 LPBF conditions with laser power, scan speed, hatch spacing, layer thickness, energy density, microstructural observation, and functional class.

```text
data/paper_level_numeric_facts.csv
```

Paper-level numerical facts such as reported transformation-temperature ranges, lattice parameters, property limits, ML dataset size, and benchmark values.

```text
data/scenario_rules.csv
```

Readable metallurgical rules used by the scenario engine.

```text
data/demo_training_seed.csv
```

Synthetic demonstration rows used only for checking the ML interface. These rows are not literature evidence and should not be used for scientific claims.

This separation is important. A small exact dataset is more useful than a larger unclear dataset.

---

## Machine-learning workbench

The app includes a machine-learning workbench, but it does not force regression when the data are insufficient.

### Functional classification

When the exact dataset contains process labels but not enough numerical targets, the app can classify a process into functional classes such as:

```text
excellent
partial
poor / not tested
```

This is useful when papers provide reliable qualitative or semi-quantitative functional outcomes but not enough row-level DSC, density, tensile, or hysteresis values.

### Regression

When enough real rows are available, the app can train regression models for targets such as:

- relative density;
- porosity;
- Ms, Mf, As, and Af;
- hysteresis;
- recoverable strain;
- residual strain;
- ultimate tensile strength;
- elongation.

The current model options are:

- Gaussian Process Regression;
- Random Forest;
- Extra Trees;
- Gradient Boosting.

Gaussian Process Regression is included because small experimental datasets need uncertainty-aware models. Tree-based models are included because LPBF process–property relationships are often nonlinear.

If a target does not have enough real non-null rows, the app does not train a regression model. It reports the missing data instead.

---

## Characterization upload lab

The app includes a first-pass analysis area for experimental results.

### SEM / optical microstructure

The user can upload a microstructure image. The app performs simple segmentation of dark features and reports:

- feature area fraction;
- feature count;
- mean feature size;
- image texture descriptors.

This is intended for quick screening of pores, cracks, pull-outs, or dark features. It should be checked against calibrated metallography before any final conclusion.

### XRD

The user can upload a two-column XRD file:

```text
two_theta_deg
intensity
```

The app detects peaks, estimates d-spacing, and gives first-pass phase hints. This can help screen possible B2/B19′ regions or secondary-phase indicators. Final phase identification still requires proper calibration, background treatment, and refinement.

### EBSD / TKD-style CSV

The user can upload an orientation-map CSV with fields such as:

- x and y coordinates;
- Euler angles;
- phase ID;
- confidence index;
- image quality;
- grain ID.

The current version provides map inspection and basic statistics. The long-term target is to extend this section toward:

- B2 parent reconstruction;
- B19′ variant indexing;
- misorientation histograms;
- variant-pair statistics;
- texture analysis relative to build direction;
- stress-assisted variant selection.

---

## Crystallographic basis

The crystallographic part of the app uses reference NiTi B2/B19′ parameters:

```text
B2 austenite:
a = 3.015 Å

B19′ martensite:
a = 2.89 Å
b = 4.12 Å
c = 4.62 Å
β = 96.8°
```

These values are used as a starting point for the transformation-window and compatibility logic. The current version is not a full crystallographic reconstruction code. It is a framework prepared for future additions such as B2 parent reconstruction, B19′ variant indexing, correspondence operators, and habit-plane or compatibility analysis.

---

## Literature basis

The current version is built around several literature directions.

### Xiang et al., 2024 — LPBF Ni-rich NiTi process window

This work provides the first row-level benchmark used in the app. It studies LPBF Ni-rich Ni51.3Ti48.7 and relates process parameters to microstructure, transformation behavior, and superelasticity.

It supports the current process scenarios:

- low-energy cracking or poor continuity;
- useful low-energy superelastic window;
- transition region with partial superelasticity;
- high-energy Ni-loss and poor superelasticity.

DOI: `10.3390/met14090961`

### Xue et al., 2022 — defect-free LPBF NiTi

This work supports the idea that LPBF NiTi performance depends on the simultaneous control of defects, Ni evaporation, oxygen, and transformation behavior. It is used as a reference for the defect-free NiTi route and high-quality tensile superelasticity.

DOI: `10.1016/j.actamat.2022.117781`

### Cayron, 2020 — B2→B19′ crystallography from EBSD/TKD

This work supports the crystallographic direction of the project, especially the use of B2/B19′ lattice data, EBSD/TKD analysis, orientation relationships, reconstructed B2 parent grains, and martensitic transformation geometry.

DOI: `10.3390/cryst10070562`

### Li et al., 2024 — ML process optimisation for LPBF NiTi

This work supports the machine-learning direction of the project. It reports a multi-publication dataset for LPBF NiTi and predicts density, ultimate tensile strength, elongation, and thermal hysteresis.

DOI: `10.1080/17452759.2024.2364221`

### ML prediction of LPBF-NiTi phase transition temperature

The project also follows recent work on using process and composition features with ML models such as GRNN to predict phase-transition temperatures. This direction is relevant for future prediction of Ms, Mf, As, and Af once enough row-level DSC data are added.

---

## Repository structure

```text
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
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MohammadAminNouri/NiTi-AM-CrystalFunction-Map.git
cd NiTi-AM-CrystalFunction-Map
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## Requirements

The main dependencies are:

```text
streamlit
numpy
pandas
plotly
scikit-learn
scipy
pillow
matplotlib
joblib
```

---

## Current limitations

The app is a research screening tool. It does not certify material performance or replace experimental characterization.

The outputs should be checked with:

- DSC for transformation temperatures;
- XRD for phase identification;
- SEM or optical metallography for defect analysis;
- EBSD/TKD for orientation and variant analysis;
- density or CT measurement for porosity;
- EDS/ICP for composition;
- cyclic compression or tension for functional response.

The app does not yet provide full B2 parent reconstruction, complete B19′ variant indexing, or industrial/medical qualification.

---

## Development roadmap

The next steps are:

1. Add more exact row-level LPBF NiTi data from papers.
2. Digitize DSC and stress–strain curves where tables are not available.
3. Expand real targets for density, Af, hysteresis, UTS, elongation, recovery strain, and residual strain.
4. Add proper model cards for each ML target.
5. Add EBSD/TKD-based B2 parent reconstruction.
6. Add B19′ variant indexing and misorientation histograms.
7. Add a more rigorous crystallographic compatibility module.
8. Add uncertainty bands around transformation-temperature predictions.
9. Separate models for density, transformation temperatures, mechanical properties, and functional class.

---

## Use of results

The safest use of this app is as a structured research assistant for:

- comparing process conditions;
- organizing literature data;
- selecting experimental windows;
- identifying missing measurements;
- linking process parameters to functional risks;
- planning DSC, XRD, SEM, EBSD/TKD, and mechanical tests.

The app should not be used as a final process qualification tool.

---

## Citation

If this repository is used or extended, cite the repository and the papers used for the relevant data or model assumptions.
