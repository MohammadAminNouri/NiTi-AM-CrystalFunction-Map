# NiTi-AM CrystalFunction Map

A benchmark-oriented Streamlit platform for laser powder bed fusion (LPBF / PBF-LB/M) of NiTi shape-memory alloys.

The project is built around a full metallurgical chain:

```text
LPBF parameters
→ melt/defect window and Ni-loss risk
→ B2 parent phase and B2→B19′ crystallography
→ martensite variants, metric distortion and compatibility descriptors
→ transformation temperatures, hysteresis and recoverable function
→ ML-assisted process-function prediction
→ SEM/XRD/EBSD/TKD result analysis
```

The app is not a generic energy-density calculator. Its central question is:

> Can a printed NiTi part remain functionally useful as a shape-memory or superelastic material after processing?

## Core features

- **Interactive process-function map** for power, speed, hatch, layer thickness, Ni content, remelting, heat treatment and service temperature.
- **Exact literature benchmark database** separated from demo/synthetic rows.
- **Machine-learning workbench** with strict validation, uncertainty-aware models, multi-target mode, error metrics and feature importance.
- **Prediction studio** for users who want to input their own process and estimate density, Af, hysteresis, UTS or elongation.
- **Characterization upload lab** for SEM/optical micrographs, XRD CSV patterns and EBSD/TKD-style orientation CSV maps.
- **Crystallography and compatibility module** using editable B2/B19′ lattice metrics, transformation descriptors, variant-pair tables and stress-assisted variant selection screening.
- **Data extraction protocol** so the database can grow into a real benchmark dataset.

## What is exact and what is not

The repository contains three data layers:

1. `data/exact_literature_process_rows.csv`  
   Exact process rows and sample-level observations extracted from openly accessible text/tables.

2. `data/paper_level_numeric_facts.csv`  
   Exact paper-level numerical facts such as reported ranges, maxima, dataset sizes and crystallographic constants.

3. `data/demo_training_seed.csv`  
   Synthetic demonstration rows only for testing the ML interface. These rows are clearly labeled and must not be used as literature evidence.

The app warns the user when demo rows are included in training.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Minimal CSV schema for real ML training

A real paper-extracted training row should ideally include:

```text
source_id, doi, sample_id, alloy_system,
laser_power_W, scan_speed_mm_s, hatch_spacing_mm, layer_thickness_mm,
VED_J_mm3, powder_Ni_at_pct, measured_Ni_at_pct, oxygen_ppm,
build_plate_C, scan_strategy, remelt_passes,
heat_treatment_C, heat_treatment_min,
relative_density_pct, porosity_pct,
Ms_C, Mf_C, As_C, Af_C, hysteresis_K,
recoverable_strain_pct, residual_strain_pct,
UTS_MPa, elongation_pct,
phase_label, precipitates, notes
```

## Publication-grade upgrade path

1. Extract 100-300 real rows from LPBF-NiTi papers.
2. Keep extracted process parameters, DSC data, density, mechanical results and heat-treatment conditions in row-level form.
3. Store figure-digitized data separately from table-extracted data.
4. Add EBSD/TKD orientation maps and actual variant labels when available.
5. Train and report external validation, not only cross-validation.
6. Publish a model card and data audit.

## Academic positioning

The project treats LPBF-NiTi as a coupled **processing-crystallography-function** problem. Density alone is not enough. A dense NiTi part can still be a poor functional material if composition shift, transformation temperature, martensite compatibility, hysteresis or texture is wrong.

## License

MIT for code. Literature data must be stored as extracted numerical facts with source metadata.
