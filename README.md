# FedCORA Dataset — Indoor Air Quality Measurements from Three Schools

Real-world indoor environmental monitoring data collected from three schools with
distinct occupancy and ventilation characteristics. The dataset accompanies the
**FedCORA** framework for context-aware federated learning of indoor CO₂ prediction
in heterogeneous smart-building environments.

## Author

**Montaser N. A. Ramadan**

- Email: montaser.ramadan@gmail.com
- Google Scholar: [Profile](https://scholar.google.com/citations?user=ixG_9iUAAAAJ&hl=en)

---

## Overview

Each school is treated as an independent client whose data is **non-IID** (non-independent
and identically distributed) relative to the others — they differ in student density,
ventilation quality, and resulting air-quality dynamics. This heterogeneity makes the
dataset well suited for studying federated and context-aware learning methods.

Measurements are recorded at a fixed **10-minute sampling interval** throughout 2025.

| School | Samples | Period (2025) | CO₂ min / mean / max (ppm) | CO₂ std | Characteristics |
|--------|--------:|---------------|----------------------------|--------:|-----------------|
| A | 13,129 | Jan 02 – Apr 03 | 418 / 1,016 / 1,841 | 202 | Low density, good ventilation |
| B | 14,586 | Jan 01 – Apr 12 | 418 / 1,280 / 2,275 | 265 | Medium density, moderate ventilation |
| C |  9,703 | Jan 04 – Mar 13 | 418 / 1,239 / 2,290 | 451 | High density, poor ventilation |
| **Total** | **37,418** | | | | **Heterogeneous / non-IID** |

---

## Data Description

The raw datasets are provided as Excel (`.xlsx`) files, one per school.
Every record (row) contains the following columns:

| Column | Unit | Description |
|--------|------|-------------|
| `time of read` | `YYYY-MM-DD HH:MM` | Timestamp of the reading (10-minute interval) |
| `CO2 (ppm)` | ppm | Carbon dioxide concentration |
| `CH2O (mg/m3)` | mg/m³ | Formaldehyde concentration |
| `VOC (GRADE)` | grade | Volatile organic compound level (index) |
| `PM2.5 (μg/m3)` | μg/m³ | Fine particulate matter (≤ 2.5 μm) |
| `PM10 (μg/m3)` | μg/m³ | Coarse particulate matter (≤ 10 μm) |
| `Temperature (°)` | °C | Indoor air temperature |
| `Humidity (%)` | % RH | Relative humidity |

---

## Repository Structure

```
FedCORA/
|
|-- README.md
|
|-- school-A.xlsx
|-- school-B.xlsx
+-- school-C.xlsx
```

### Loading the data (Python)

```python
import pandas as pd

df = pd.read_excel("/school-A.xlsx")
print(df.head())
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{ramadan2026fedcora,
  title  = {FedCORA: daptive Federated Learning through Knowledge-Guided Context-Aware Aggregation in Heterogeneous Non-IID Environments},
  author = {Ramadan, Montaser N. A.},
  year   = {2026},
  url    = {https://github.com/montaserramadan/FedCora}
}
```

---

## License & Usage Permission

This dataset is released under the **Creative Commons Attribution 4.0 International
(CC BY 4.0)** license — see the [LICENSE](LICENSE) file for details.

You are **free to**:

- **Share** — copy and redistribute the data in any medium or format.
- **Adapt** — remix, transform, and build upon the data for any purpose, including
  commercial use.

**Under the following condition:**

- **Attribution** — you must give appropriate credit by citing the work above, provide
  a link to this repository, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/

---

## Contact

For questions, data details, or collaboration inquiries:

**Montaser N. A. Ramadan** — montaser.ramadan@gmail.com
