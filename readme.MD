# Multi-Resource Renewable Energy Potential Mapping
## ML-Based Decision Support System | Odisha, India

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-production-success.svg)](https://github.com/assassindiv/sustainable-resource_mapping)

---

## 📋 Overview

A comprehensive **6-phase machine learning pipeline** for optimal renewable energy site selection across Odisha, India. This system integrates spatial analysis, constraint modeling, and advanced ML to generate actionable suitability maps for **Solar**, **Wind**, and **Biomass** installations at block-level resolution (314 administrative units).

**Key Innovation:** Hybrid ML framework combining K-Means clustering, Random Forest classification, XGBoost validation, and Analytical Hierarchy Process (AHP) for expert knowledge integration.

---

## 🎯 Project Objectives

- **Spatial Optimization:** Identify highest-potential zones for each renewable energy technology
- **Resource Allocation:** Generate integrated decision maps showing optimal technology per location
- **Policy Support:** Provide granular, block-level recommendations for energy infrastructure planning
- **Scalability:** Reproducible methodology applicable to other regions and resource types

---

## 📊 The 6-Phase Methodology

### **Phase 1: Data Acquisition & Preprocessing** ✅
**Objective:** Assemble comprehensive geospatial datasets for Odisha

**Datasets Acquired:**
- **Infrastructure:** Roads, railways, transmission lines, substations (OSM + Power Grid data)
- **Environmental:** Protected areas (WDPA), water bodies, forest cover
- **Geophysical:** DEM terrain (slope/aspect), solar irradiance (GHI/DNI), wind speed (100m height)
- **Socioeconomic:** Population density, settlement data (Census 2021)

**Processing:** All datasets clipped to Odisha administrative boundary and standardized to common CRS

---

### **Phase 2: Spatial Constraint Modeling** ✅
**Objective:** Define "No-Go Zones" where energy infrastructure is legally or physically prohibited

**Methodology:**
- **Exclusion Buffers:** 1km around protected areas, 500m around water bodies
- **Terrain Filtering:** Exclude slopes >10° (Solar) and >15° (Wind/Biomass)
- **Land Use Restrictions:** Mask dense urban areas and restricted zones

**Output:** `constraints_mask.tif` (Binary: 0=Restricted, 1=Available)

---

### **Phase 3: Feature Engineering** ✅
**Objective:** Transform raw spatial data into normalized ML-ready features

**Features Generated (7 variables):**
1. `solar_mean` — Average solar irradiance (kWh/m²/day)
2. `wind_mean` — Average wind speed (m/s at 100m)
3. `pop_mean` — Population density (proxy for biomass availability)
4. `dist_roads_mean` — Distance to major roads (km)
5. `dist_trans_mean` — Distance to transmission lines (km)
6. `dist_sub_mean` — Distance to substations (km)
7. `constraint_pct` — Percentage of block under constraints

**Processing:** All features normalized to 0.0–1.0 range for model stability

**Output:** `block_features.csv` (314 blocks × 7 features)

---

### **Phase 4: Multi-Criteria Zone Classification** ✅
**Objective:** Create authoritative suitability zones using GIS-based overlay analysis

**Methodology:**
- **Weighted Overlay:** Combine normalized features using domain-expert weights (AHP)
- **Classification:** Threshold-based categorization into High/Medium/Low suitability
- **Zone Extraction:** Identify top 30% spatial clusters for each technology

**Outputs:**
- `top_solar_zones.gpkg` — High-potential solar regions
- `top_wind_zones.gpkg` — High-potential wind corridors
- `top_biomass_zones.gpkg` — High-potential biomass areas

---

### **Phase 5: Machine Learning Pipeline** ✅ **(Current Focus)**
**Objective:** Train ML models for automated, data-driven site prediction

**Workflow:**
1. **Unsupervised Clustering** (K-Means, k=4) → Group similar blocks
2. **Rule-Based Labeling** → Assign initial energy types (SOLAR/WIND/BIOMASS/HYBRID)
3. **Hyperparameter Tuning** (GridSearchCV) → Optimize Random Forest parameters
4. **Model Training** → Supervised learning with 5-fold cross-validation
5. **Validation** → Compare with XGBoost, compute feature importance
6. **Confidence Scoring** → Override low-confidence predictions to HYBRID

**Academic Enhancements:**
- ✓ **Correlation Analysis:** Multicollinearity detection (no pairs >0.85)
- ✓ **k-Optimization:** Elbow & Silhouette methods validate k=4
- ✓ **Ablation Study:** Quantify individual feature contribution
- ✓ **MAUP Analysis:** Block vs district scale comparison
- ✓ **GridSearchCV:** Systematic hyperparameter optimization (18 combinations)

**Performance:**
- Random Forest Accuracy: **97.8% ± 1.3%** (5-fold CV)
- XGBoost Accuracy: **97.5% ± 1.4%** (validation)
- High Confidence Predictions: **72%** of blocks (>80% confidence)

**Outputs:**
- `final_predictions_simple.csv` — Block-level predictions with confidence scores
- `model_simple.pkl` — Trained Random Forest model
- 7 visualization PNGs (correlation, ablation, confusion matrices, etc.)

---

### **Phase 6: Decision Support System & Visualization** ✅
**Objective:** Deliver interactive maps for policy and investment decisions

**Core Deliverables:**
1. **Solar Suitability Heatmap** — Probability map (0-1 scale) showing solar potential
2. **Wind Suitability Heatmap** — Probability map for wind energy corridors
3. **Biomass Suitability Heatmap** — Vegetation/residue energy potential zones
4. **Optimal Resource Allocation Map** 🏆 
   - **Logic:** For each pixel, `argmax(Solar, Wind, Biomass)` determines best technology
   - **Output:** Single-layer decision map: "Region A → Solar, Region B → Wind"
   - **Use Case:** Direct infrastructure investment planning

**Integration:** Maps exportable to QGIS/ArcGIS for further spatial analysis and reporting

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/assassindiv/sustainable-resource_mapping.git
cd sustainable-resource_mapping/miniproject

# Install dependencies
pip install -r requirements.txt

# Verify installation
python check_installation.py
```

### Run Phase 5 Pipeline
```bash
python newphase5.py
```

**Expected Runtime:** 30-60 seconds (314 blocks, GridSearchCV included)

---

## 📂 Project Structure

```
miniproject/
├── newphase5.py                    # Main ML pipeline (269 lines, optimized)
├── requirements.txt                # Python dependencies
├── phase4_block/
│   ├── block_features.csv          # Input: 314 blocks × 7 features
│   ├── top_solar_zones.gpkg        # Phase 4 GIS outputs
│   ├── top_wind_zones.gpkg
│   └── top_biomass_zones.gpkg
├── new5op/                         # Phase 5 outputs (auto-generated)
│   ├── final_predictions_simple.csv
│   ├── model_simple.pkl
│   ├── correlation_heatmap.png
│   ├── optimal_k_selection.png
│   ├── ablation_study.png
│   └── ... (7 total visualizations)
└── readme.MD                       # This file
```

---

## 📊 Key Results

### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RF CV Accuracy** | 97.8% ± 1.3% | Excellent predictive power |
| **XGBoost Accuracy** | 97.5% ± 1.4% | Strong model agreement |
| **Silhouette Score** | 0.234 | Fair cluster separation (k=4) |
| **High Confidence Blocks** | 72% | Majority predictions are robust |

### Prediction Distribution
- **SOLAR:** 250 blocks (80%) — Dominated by high irradiance state-wide
- **WIND:** 2 blocks (0.6%) — Limited high-altitude wind corridors
- **HYBRID:** 62 blocks (20%) — Mixed-resource transition zones
- **BIOMASS:** 0 blocks — Low population-normalized potential

### Critical Findings
1. **Feature Ablation:** `wind_mean` most critical (6.7% accuracy drop when removed)
2. **MAUP Effects:** 7/30 districts show scale-dependent predictions
3. **Expert-ML Alignment:** AHP weights moderately correlate with RF feature importance (r=0.68)

---

## 🎓 Academic Contributions

1. **Methodological Rigor:** 5 enhancements (correlation, k-opt, tuning, ablation, MAUP) exceed standard ML practices
2. **Spatial Scale Analysis:** First study to quantify MAUP effects in renewable energy site selection for Odisha
3. **Reproducibility:** Fully documented parameters, open-source code, citation-ready methodology
4. **Applied Impact:** Directly applicable to state energy policy and investment planning

---

## 📖 Citation

```bibtex
@software{odisha_renewable_ml_2026,
  author = {Divyansu Puri and Aryan Singh and Ashutosh Singh and Madhusudan and Keshav and Astitva Tripathi},
  title = {Multi-Resource Renewable Energy Potential Mapping: ML-Based Decision Support System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/assassindiv/sustainable-resource_mapping}}
}
```

---

## 🛠️ System Requirements

- **Python:** 3.8+ (tested on 3.9, 3.10, 3.11)
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** ~500 MB (including dependencies and outputs)
- **OS:** Windows, macOS, Linux

---

## 📧 Contact & Support

**Project Lead:** Asutosh Singh  
**Institution:** KIIT University  
**Repository:** [github.com/assassindiv/sustainable-resource_mapping](https://github.com/assassindiv/sustainable-resource_mapping)  
**Issues:** Use GitHub Issues for bug reports and feature requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Data Sources:**
- Solar/Wind: NASA POWER, Global Wind Atlas
- Infrastructure: OpenStreetMap, National Power Grid of India
- Administrative: Census of India 2021, Survey of India

**Methodological References:**
- Hengl et al. (2018) - Random Forest for spatial prediction
- Openshaw (1984) - The Modifiable Areal Unit Problem
- Saaty (1980) - Analytical Hierarchy Process

---

**Last Updated:** March 9, 2026 | **Version:** 2.0 | **Status:** ✅ Production Ready
