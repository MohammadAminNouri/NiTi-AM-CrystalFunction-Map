# NiTi-AM CrystalFunction Map

A benchmark-oriented Streamlit platform for **laser powder bed fusion (LPBF / PBF-LB/M) of NiTi shape-memory alloys**.

The app connects:

```text
LPBF parameters
→ melt-pool / defect / Ni-loss risk
→ B2 ↔ B19′ transformation window
→ crystallographic compatibility and variant-screening descriptors
→ superelastic or shape-memory functional suitability
→ literature benchmark comparison
→ ML-assisted prediction when enough real data exist
→ SEM/XRD/EBSD user-result analysis
```

## What was fixed in this version

- The transformation-window plot now displays actual Mf–Ms and As–Af intervals instead of an empty axis.
- The ML page no longer fails silently when exact data are sparse. It explains why regression cannot be trained, offers functional classification on exact rows, and allows demo rows only for interface testing.
- The app now has a scenario engine with metallurgical explanations for low-energy failure, useful superelastic window, keyhole/partial response, high-energy Ni-loss risk, defect-free optimization and ML benchmarking.
- Characterization upload pages now explain what SEM/XRD/EBSD results mean for functional NiTi, not just show images.

## Data layers

```text
data/exact_literature_process_rows.csv   # exact A1-D4 row-level process/function data
data/paper_level_numeric_facts.csv       # exact numerical constants/ranges/facts
data/scenario_rules.csv                  # readable metallurgical scenario logic
data/demo_training_seed.csv              # synthetic demo rows; never evidence
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main warning

The app is a research screening and benchmark tool. It is not a substitute for DSC, XRD, EBSD/TKD, density/porosity measurement, cyclic compression/tension, or medical/device qualification.
