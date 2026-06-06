# Mitchell Training Bbox — Multi-Year Quality Audit

**Total year-rows:** 207  | ✓ pass: 25  | ⚠ flag: 62  | ✗ fail: 118  | — no data: 2

## Summary by cover type

| Cover type | ✓ pass | ⚠ flag | ✗ fail | — no data |
|------------|--------|--------|--------|-----------|
| presence   |      2 |     33 |     45 |         1 |
| mangrove   |      1 |      5 |     30 |         0 |
| bare       |      1 |      7 |     18 |         1 |
| riparian   |      1 |     13 |     13 |         0 |
| water      |     20 |      4 |     12 |         0 |

---

## Presence regions

### mitchell_presence_1  (presence · train)
Bbox: [141.363789, -16.33677, 141.365969, -16.335197]  Tile: 54KWG

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.86 | 0.13 | 0.74 | -4.52 | 0.25 | ✗ fail |
| 2018 | 0.96 | 0.16 | 0.80 | -4.07 | 0.16 | ✗ fail |
| 2019 | 0.86 | 0.21 | 0.65 | -4.68 | 0.29 | ⚠ flag |
| 2020 | 0.92 | 0.26 | 0.66 | -4.32 | 0.12 | ✗ fail |
| 2021 | 0.92 | 0.26 | 0.65 | -4.60 | 0.39 | ✗ fail |
| 2022 | 0.86 | 0.11 | 0.75 | -4.39 | 0.17 | ✗ fail |
| 2023 | 0.93 | 0.25 | 0.67 | -4.45 | 0.10 | ✗ fail |
| 2024 | 0.90 | 0.27 | 0.62 | -4.62 | 0.19 | ⚠ flag |
| 2025 | 0.87 | 0.22 | 0.66 | -4.52 | 0.09 | ⚠ flag |
| **All-year** | 0.86–0.96 | 0.11–0.27 | 0.62–0.80 | -4.68–-4.07 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.25 > 0.12 — possible spatial mixing
- 2018: IQR max 0.16 > 0.12 — possible spatial mixing
- 2019: IQR max 0.29 > 0.12 — possible spatial mixing
- 2021: IQR max 0.39 > 0.12 — possible spatial mixing
- 2022: IQR max 0.17 > 0.12 — possible spatial mixing
- 2024: IQR max 0.19 > 0.12 — possible spatial mixing

### mitchell_presence_2  (presence · train)
Bbox: [141.400465, -16.324751, 141.402291, -16.323115]  Tile: 54KWG

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | — | 0.24 | — | -4.84 | 0.17 | ⚠ flag |
| 2018 | 0.90 | 0.14 | 0.75 | -4.45 | 0.22 | ✗ fail |
| 2019 | 0.87 | 0.28 | 0.60 | -5.17 | 0.35 | ⚠ flag |
| 2020 | 0.85 | 0.28 | 0.57 | -4.82 | 0.38 | ⚠ flag |
| 2021 | 0.90 | 0.08 | 0.82 | -5.12 | 0.17 | ✗ fail |
| 2022 | 0.89 | 0.14 | 0.75 | -4.85 | 0.14 | ✗ fail |
| 2023 | — | 0.22 | — | -5.38 | 0.03 | ✓ pass |
| 2024 | 0.84 | 0.24 | 0.60 | -5.35 | 0.09 | ⚠ flag |
| 2025 | 0.90 | 0.20 | 0.70 | -5.36 | 0.09 | ✗ fail |
| **All-year** | 0.84–0.90 | 0.08–0.28 | 0.57–0.82 | -5.36–-4.45 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.17 > 0.12 — possible spatial mixing
- 2018: IQR max 0.22 > 0.12 — possible spatial mixing
- 2019: IQR max 0.35 > 0.12 — possible spatial mixing
- 2020: IQR max 0.38 > 0.12 — possible spatial mixing
- 2021: IQR max 0.17 > 0.12 — possible spatial mixing
- 2022: IQR max 0.14 > 0.12 — possible spatial mixing

### mitchell_presence_3  (presence · train)
Bbox: [141.379285, -16.345988, 141.380974, -16.34196]  Tile: 54KWG

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.80 | 0.21 | 0.59 | -4.27 | 0.28 | ⚠ flag |
| 2018 | 0.89 | 0.11 | 0.78 | -3.92 | 0.19 | ✗ fail |
| 2019 | 0.86 | 0.22 | 0.64 | -4.69 | 0.30 | ⚠ flag |
| 2020 | 0.85 | 0.32 | 0.53 | -4.15 | 0.12 | ⚠ flag |
| 2021 | 0.84 | 0.11 | 0.74 | -4.48 | 0.39 | ✗ fail |
| 2022 | 0.90 | 0.25 | 0.64 | -4.30 | 0.19 | ⚠ flag |
| 2023 | 0.88 | 0.25 | 0.62 | -4.66 | 0.15 | ⚠ flag |
| 2024 | 0.86 | 0.27 | 0.60 | -4.81 | 0.13 | ⚠ flag |
| 2025 | 0.88 | 0.17 | 0.71 | -4.67 | 0.14 | ✗ fail |
| **All-year** | 0.80–0.90 | 0.11–0.32 | 0.53–0.78 | -4.81–-3.92 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.28 > 0.12 — possible spatial mixing
- 2018: IQR max 0.19 > 0.12 — possible spatial mixing
- 2019: IQR max 0.30 > 0.12 — possible spatial mixing
- 2021: IQR max 0.39 > 0.12 — possible spatial mixing
- 2022: IQR max 0.19 > 0.12 — possible spatial mixing
- 2023: IQR max 0.15 > 0.12 — possible spatial mixing
- 2024: IQR max 0.13 > 0.12 — possible spatial mixing
- 2025: IQR max 0.14 > 0.12 — possible spatial mixing

### mitchell_presence_4  (presence · train)
Bbox: [141.389167, -16.199947, 141.3905, -16.197966]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.71 | 0.11 | 0.60 | -4.54 | 0.16 | ✗ fail |
| 2018 | 0.84 | 0.05 | 0.79 | -4.15 | 0.27 | ✗ fail |
| 2019 | 0.85 | 0.25 | 0.60 | -4.75 | 0.15 | ⚠ flag |
| 2020 | 0.83 | 0.15 | 0.68 | -4.46 | 0.14 | ✗ fail |
| 2021 | 0.79 | 0.17 | 0.62 | -4.34 | 0.28 | ✗ fail |
| 2022 | 0.82 | 0.22 | 0.60 | -4.47 | 0.13 | ⚠ flag |
| 2023 | 0.85 | 0.21 | 0.63 | -4.40 | 0.12 | ⚠ flag |
| 2024 | — | — | — | — | — | — no data |
| 2025 | 0.83 | 0.22 | 0.61 | -4.63 | 0.16 | ⚠ flag |
| **All-year** | 0.71–0.85 | 0.05–0.25 | 0.60–0.79 | -4.75–-4.15 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.16 > 0.12 — possible spatial mixing
- 2018: IQR max 0.27 > 0.12 — possible spatial mixing
- 2019: IQR max 0.15 > 0.12 — possible spatial mixing
- 2020: IQR max 0.14 > 0.12 — possible spatial mixing
- 2021: IQR max 0.28 > 0.12 — possible spatial mixing
- 2022: IQR max 0.13 > 0.12 — possible spatial mixing
- 2025: IQR max 0.16 > 0.12 — possible spatial mixing

### mitchell_presence_5  (presence · train)
Bbox: [141.463662, -15.990899, 141.465735, -15.987729]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.84 | 0.14 | 0.69 | -4.31 | 0.10 | ✗ fail |
| 2018 | 0.92 | 0.18 | 0.74 | -4.09 | 0.22 | ✗ fail |
| 2019 | 0.90 | 0.29 | 0.61 | -4.27 | 0.27 | ⚠ flag |
| 2020 | 0.90 | 0.08 | 0.81 | -4.32 | 0.14 | ✗ fail |
| 2021 | 0.88 | 0.30 | 0.57 | -4.09 | 0.34 | ⚠ flag |
| 2022 | 0.84 | 0.24 | 0.60 | -4.28 | 0.25 | ⚠ flag |
| 2023 | 0.97 | 0.27 | 0.70 | -4.43 | 0.13 | ✗ fail |
| 2024 | 0.84 | 0.16 | 0.67 | -4.16 | 0.11 | ✗ fail |
| 2025 | 0.81 | 0.16 | 0.65 | -4.21 | 0.09 | ✗ fail |
| **All-year** | 0.81–0.97 | 0.08–0.30 | 0.57–0.81 | -4.43–-4.09 | — | **✗ fail** |

**Notes:**
- 2018: IQR max 0.22 > 0.12 — possible spatial mixing
- 2019: IQR max 0.27 > 0.12 — possible spatial mixing
- 2020: IQR max 0.14 > 0.12 — possible spatial mixing
- 2021: IQR max 0.34 > 0.12 — possible spatial mixing
- 2022: IQR max 0.25 > 0.12 — possible spatial mixing
- 2023: IQR max 0.13 > 0.12 — possible spatial mixing

### mitchell_val_presence_1  (presence · val)
Bbox: [141.364065, -16.33944, 141.364817, -16.336892]  Tile: 54KWG

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.85 | 0.12 | 0.73 | -4.58 | 0.51 | ✗ fail |
| 2018 | 0.94 | 0.22 | 0.72 | -4.30 | 0.37 | ✗ fail |
| 2019 | 0.86 | 0.24 | 0.62 | -4.42 | 0.24 | ⚠ flag |
| 2020 | 0.91 | 0.27 | 0.64 | -4.42 | 0.13 | ✗ fail |
| 2021 | 0.91 | 0.26 | 0.65 | -4.56 | 0.28 | ✗ fail |
| 2022 | 0.84 | 0.19 | 0.64 | -4.19 | 0.24 | ✗ fail |
| 2023 | 0.93 | 0.26 | 0.67 | -4.24 | 0.49 | ✗ fail |
| 2024 | 0.93 | 0.28 | 0.65 | -4.35 | 0.08 | ✗ fail |
| 2025 | 0.89 | 0.26 | 0.63 | -4.39 | 0.16 | ⚠ flag |
| **All-year** | 0.84–0.94 | 0.12–0.28 | 0.62–0.73 | -4.58–-4.19 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.51 > 0.12 — possible spatial mixing
- 2018: IQR max 0.37 > 0.12 — possible spatial mixing
- 2019: IQR max 0.24 > 0.12 — possible spatial mixing
- 2020: IQR max 0.13 > 0.12 — possible spatial mixing
- 2021: IQR max 0.28 > 0.12 — possible spatial mixing
- 2022: IQR max 0.24 > 0.12 — possible spatial mixing
- 2023: IQR max 0.49 > 0.12 — possible spatial mixing
- 2025: IQR max 0.16 > 0.12 — possible spatial mixing

### mitchell_val_presence_2  (presence · val)
Bbox: [141.394728, -16.320307, 141.398059, -16.319024]  Tile: 54KWG

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | — | 0.34 | — | -5.10 | 0.36 | ⚠ flag |
| 2018 | 0.88 | 0.04 | 0.83 | -4.73 | 0.29 | ✗ fail |
| 2019 | 0.88 | 0.25 | 0.64 | -5.12 | 0.46 | ⚠ flag |
| 2020 | 0.89 | 0.28 | 0.62 | -4.98 | 0.19 | ⚠ flag |
| 2021 | 0.89 | 0.15 | 0.74 | -4.99 | 0.18 | ✗ fail |
| 2022 | 0.88 | 0.13 | 0.74 | -4.70 | 0.18 | ✗ fail |
| 2023 | — | 0.22 | — | -5.42 | 0.04 | ✓ pass |
| 2024 | 0.87 | 0.24 | 0.62 | -5.21 | 0.13 | ⚠ flag |
| 2025 | 0.87 | 0.22 | 0.65 | -5.04 | 0.12 | ⚠ flag |
| **All-year** | 0.87–0.89 | 0.04–0.28 | 0.62–0.83 | -5.21–-4.70 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.36 > 0.12 — possible spatial mixing
- 2018: IQR max 0.29 > 0.12 — possible spatial mixing
- 2019: IQR max 0.46 > 0.12 — possible spatial mixing
- 2020: IQR max 0.19 > 0.12 — possible spatial mixing
- 2021: IQR max 0.18 > 0.12 — possible spatial mixing
- 2022: IQR max 0.18 > 0.12 — possible spatial mixing
- 2024: IQR max 0.13 > 0.12 — possible spatial mixing
- 2025: IQR max 0.12 > 0.12 — possible spatial mixing

### mitchell_val_presence_3  (presence · val)
Bbox: [141.417057, -16.209358, 141.420611, -16.206871]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.84 | 0.15 | 0.70 | -4.71 | 0.14 | ✗ fail |
| 2018 | 0.84 | 0.22 | 0.62 | -4.36 | 0.08 | ⚠ flag |
| 2019 | 0.72 | 0.23 | 0.49 | -4.50 | 0.36 | ✗ fail |
| 2020 | 0.83 | 0.16 | 0.67 | -4.44 | 0.17 | ✗ fail |
| 2021 | 0.82 | 0.11 | 0.71 | -4.45 | 0.23 | ✗ fail |
| 2022 | 0.72 | 0.14 | 0.58 | -4.65 | 0.24 | ✗ fail |
| 2023 | 0.87 | 0.12 | 0.75 | -4.69 | 0.10 | ✗ fail |
| 2024 | 0.81 | 0.16 | 0.66 | -5.01 | 0.18 | ✗ fail |
| 2025 | 0.82 | 0.22 | 0.60 | -4.87 | 0.07 | ⚠ flag |
| **All-year** | 0.72–0.87 | 0.11–0.23 | 0.49–0.75 | -5.01–-4.36 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.14 > 0.12 — possible spatial mixing
- 2019: IQR max 0.36 > 0.12 — possible spatial mixing
- 2020: IQR max 0.17 > 0.12 — possible spatial mixing
- 2021: IQR max 0.23 > 0.12 — possible spatial mixing
- 2022: IQR max 0.24 > 0.12 — possible spatial mixing
- 2024: IQR max 0.18 > 0.12 — possible spatial mixing

### mitchell_val_presence_4  (presence · val)
Bbox: [141.474997, -15.959035, 141.477229, -15.956661]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.78 | 0.15 | 0.64 | -4.58 | 0.40 | ✗ fail |
| 2018 | 0.86 | 0.09 | 0.77 | -4.34 | 0.09 | ✗ fail |
| 2019 | 0.82 | 0.13 | 0.69 | -4.75 | 0.44 | ✗ fail |
| 2020 | 0.83 | 0.08 | 0.75 | -4.50 | 0.22 | ✗ fail |
| 2021 | 0.85 | 0.10 | 0.75 | -4.49 | 0.18 | ✗ fail |
| 2022 | 0.77 | 0.25 | 0.52 | -4.64 | 0.14 | ⚠ flag |
| 2023 | 0.84 | 0.26 | 0.58 | -4.81 | 0.05 | ⚠ flag |
| 2024 | 0.80 | 0.26 | 0.54 | -4.67 | 0.05 | ⚠ flag |
| 2025 | 0.79 | 0.22 | 0.57 | -5.02 | 0.05 | ⚠ flag |
| **All-year** | 0.77–0.86 | 0.08–0.26 | 0.52–0.77 | -5.02–-4.34 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.40 > 0.12 — possible spatial mixing
- 2019: IQR max 0.44 > 0.12 — possible spatial mixing
- 2020: IQR max 0.22 > 0.12 — possible spatial mixing
- 2021: IQR max 0.18 > 0.12 — possible spatial mixing
- 2022: IQR max 0.14 > 0.12 — possible spatial mixing

## Mangrove regions

### mitchell_absence_mangrove_1  (absence · train)
Bbox: [141.432703, -15.82378, 141.434759, -15.81969]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.85 | 0.67 | 0.18 | -5.29 | 0.22 | ✗ fail |
| 2018 | 0.94 | 0.56 | 0.38 | -5.14 | 0.24 | ✗ fail |
| 2019 | 0.87 | 0.43 | 0.44 | -5.43 | 0.15 | ✗ fail |
| 2020 | 0.86 | 0.61 | 0.26 | -5.56 | 0.21 | ⚠ flag |
| 2021 | 0.92 | 0.32 | 0.60 | -5.54 | 0.52 | ✗ fail |
| 2022 | 0.87 | 0.53 | 0.34 | -5.55 | 0.22 | ✗ fail |
| 2023 | 0.92 | 0.63 | 0.28 | -5.55 | 0.26 | ⚠ flag |
| 2024 | 0.92 | 0.69 | 0.23 | -5.50 | 0.12 | ✓ pass |
| 2025 | 0.93 | 0.56 | 0.37 | -5.77 | 0.38 | ✗ fail |
| **All-year** | 0.85–0.94 | 0.32–0.69 | 0.18–0.60 | -5.77–-5.14 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.22 > 0.12 — possible spatial mixing
- 2018: IQR max 0.24 > 0.12 — possible spatial mixing
- 2019: IQR max 0.15 > 0.12 — possible spatial mixing
- 2020: IQR max 0.21 > 0.12 — possible spatial mixing
- 2021: IQR max 0.52 > 0.12 — possible spatial mixing
- 2022: IQR max 0.22 > 0.12 — possible spatial mixing
- 2023: IQR max 0.26 > 0.12 — possible spatial mixing
- 2025: IQR max 0.38 > 0.12 — possible spatial mixing

### mitchell_absence_mangrove_2  (absence · train)
Bbox: [141.501087, -15.849494, 141.502921, -15.844463]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.89 | 0.59 | 0.30 | -5.22 | 0.19 | ✗ fail |
| 2018 | 0.94 | 0.19 | 0.76 | -5.09 | 0.43 | ✗ fail |
| 2019 | 0.91 | 0.32 | 0.58 | -5.39 | 0.16 | ✗ fail |
| 2020 | 0.91 | 0.38 | 0.53 | -5.56 | 0.43 | ✗ fail |
| 2021 | 0.96 | 0.60 | 0.36 | -5.33 | 0.17 | ✗ fail |
| 2022 | 0.93 | 0.17 | 0.76 | -5.51 | 0.26 | ✗ fail |
| 2023 | 0.96 | 0.52 | 0.43 | -5.62 | 0.23 | ✗ fail |
| 2024 | 0.92 | 0.70 | 0.22 | -5.59 | 0.14 | ⚠ flag |
| 2025 | 0.95 | 0.72 | 0.23 | -5.54 | 0.16 | ⚠ flag |
| **All-year** | 0.89–0.96 | 0.17–0.72 | 0.22–0.76 | -5.62–-5.09 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.19 > 0.12 — possible spatial mixing
- 2018: IQR max 0.43 > 0.12 — possible spatial mixing
- 2019: IQR max 0.16 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-02-09 (0.05) — likely cloud/shadow
- 2020: IQR max 0.43 > 0.12 — possible spatial mixing
- 2021: IQR max 0.17 > 0.12 — possible spatial mixing
- 2022: IQR max 0.26 > 0.12 — possible spatial mixing
- 2023: IQR max 0.23 > 0.12 — possible spatial mixing
- 2024: IQR max 0.14 > 0.12 — possible spatial mixing
- 2025: IQR max 0.16 > 0.12 — possible spatial mixing

### mitchell_absence_mangrove_3  (absence · train)
Bbox: [141.425301, -16.05773, 141.427833, -16.054378]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.55 | 0.08 | 0.46 | -6.18 | 0.24 | ✗ fail |
| 2018 | 0.60 | 0.06 | 0.54 | -5.49 | 0.34 | ✗ fail |
| 2019 | 0.47 | 0.17 | 0.30 | -6.18 | 0.41 | ✗ fail |
| 2020 | 0.49 | 0.10 | 0.39 | -6.32 | 0.71 | ✗ fail |
| 2021 | 0.66 | 0.13 | 0.53 | -6.77 | 0.92 | ✗ fail |
| 2022 | 0.58 | 0.37 | 0.21 | -6.29 | 0.70 | ✗ fail |
| 2023 | 0.77 | 0.37 | 0.40 | -6.20 | 0.28 | ✗ fail |
| 2024 | 0.80 | 0.50 | 0.30 | -5.86 | 0.19 | ✗ fail |
| 2025 | 0.84 | 0.40 | 0.44 | -5.96 | 0.20 | ✗ fail |
| **All-year** | 0.47–0.84 | 0.06–0.50 | 0.21–0.54 | -6.77–-5.49 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-01-20 (0.05) — likely cloud/shadow
- 2017: IQR max 0.24 > 0.12 — possible spatial mixing
- 2018: IQR max 0.34 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-01-10 (-0.12) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-16 (-0.30) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-15 (-0.07) — likely cloud/shadow
- 2019: IQR max 0.41 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-04-14 (0.02) — likely cloud/shadow
- 2020: IQR max 0.71 > 0.12 — possible spatial mixing
- 2021: near-zero NDVI 2021-03-15 (-0.02) — likely cloud/shadow
- 2021: IQR max 0.92 > 0.12 — possible spatial mixing
- 2022: IQR max 0.70 > 0.12 — possible spatial mixing
- 2023: IQR max 0.28 > 0.12 — possible spatial mixing
- 2024: near-zero NDVI 2024-03-24 (-0.03) — likely cloud/shadow
- 2024: IQR max 0.19 > 0.12 — possible spatial mixing
- 2025: IQR max 0.20 > 0.12 — possible spatial mixing

### mitchell_val_absence_mangrove_1  (absence · val)
Bbox: [141.416778, -15.894227, 141.419044, -15.891997]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.85 | 0.59 | 0.26 | -4.75 | 0.49 | ✗ fail |
| 2018 | 0.92 | 0.63 | 0.29 | -4.55 | 0.15 | ⚠ flag |
| 2019 | 0.83 | 0.61 | 0.22 | -4.96 | 0.28 | ✗ fail |
| 2020 | 0.83 | 0.65 | 0.18 | -5.24 | 0.12 | ✗ fail |
| 2021 | 0.89 | 0.54 | 0.35 | -4.93 | 0.33 | ✗ fail |
| 2022 | 0.83 | 0.35 | 0.49 | -5.16 | 0.50 | ✗ fail |
| 2023 | 0.88 | 0.13 | 0.75 | -4.99 | 0.08 | ✗ fail |
| 2024 | 0.85 | 0.67 | 0.18 | -4.96 | 1.31 | ✗ fail |
| 2025 | 0.84 | 0.49 | 0.35 | -5.06 | 0.49 | ✗ fail |
| **All-year** | 0.83–0.92 | 0.13–0.67 | 0.18–0.75 | -5.24–-4.55 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.49 > 0.12 — possible spatial mixing
- 2018: IQR max 0.15 > 0.12 — possible spatial mixing
- 2019: IQR max 0.28 > 0.12 — possible spatial mixing
- 2020: IQR max 0.12 > 0.12 — possible spatial mixing
- 2021: IQR max 0.33 > 0.12 — possible spatial mixing
- 2022: IQR max 0.50 > 0.12 — possible spatial mixing
- 2024: IQR max 1.31 > 0.12 — possible spatial mixing
- 2025: IQR max 0.49 > 0.12 — possible spatial mixing

## Bare regions

### mitchell_absence_bare_1  (absence · train)
Bbox: [141.490979, -16.057881, 141.494836, -16.055694]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.48 | 0.05 | 0.43 | -7.09 | 0.11 | ✗ fail |
| 2018 | 0.49 | 0.11 | 0.38 | -7.14 | 0.11 | ✗ fail |
| 2019 | — | 0.15 | — | -7.31 | 0.09 | ✓ pass |
| 2020 | 0.49 | 0.04 | 0.45 | -7.65 | 0.11 | ✗ fail |
| 2021 | 0.62 | 0.17 | 0.45 | -7.08 | 0.14 | ⚠ flag |
| 2022 | 0.51 | 0.15 | 0.36 | -7.35 | 0.10 | ✗ fail |
| 2023 | 0.51 | 0.18 | 0.33 | -7.84 | 0.08 | ✗ fail |
| 2024 | — | — | — | — | — | — no data |
| 2025 | 0.49 | 0.11 | 0.38 | -7.78 | 0.09 | ✗ fail |
| **All-year** | 0.48–0.62 | 0.04–0.18 | 0.33–0.45 | -7.84–-7.08 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-01-20 (0.04) — likely cloud/shadow
- 2021: IQR max 0.14 > 0.12 — possible spatial mixing

### mitchell_absence_bare_2  (absence · train)
Bbox: [141.637069, -15.392288, 141.640681, -15.389818]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.67 | 0.13 | 0.54 | -6.91 | 0.09 | ⚠ flag |
| 2018 | 0.77 | 0.12 | 0.65 | -6.16 | 0.20 | ✗ fail |
| 2019 | 0.62 | 0.15 | 0.47 | -6.02 | 0.11 | ✗ fail |
| 2020 | 0.61 | -0.04 | 0.65 | -7.05 | 0.15 | ✗ fail |
| 2021 | 0.79 | 0.08 | 0.71 | -6.80 | 0.12 | ⚠ flag |
| 2022 | 0.76 | 0.14 | 0.62 | -6.95 | 0.12 | ⚠ flag |
| 2023 | 0.79 | 0.14 | 0.65 | -6.82 | 0.05 | ⚠ flag |
| 2024 | 0.78 | 0.19 | 0.59 | -6.59 | 0.33 | ⚠ flag |
| 2025 | 0.68 | 0.09 | 0.59 | -7.03 | 0.08 | ⚠ flag |
| **All-year** | 0.61–0.79 | -0.04–0.19 | 0.47–0.71 | -7.05–-6.02 | — | **✗ fail** |

**Notes:**
- 2018: IQR max 0.20 > 0.12 — possible spatial mixing
- 2020: IQR max 0.15 > 0.12 — possible spatial mixing
- 2021: IQR max 0.12 > 0.12 — possible spatial mixing
- 2022: IQR max 0.12 > 0.12 — possible spatial mixing
- 2024: IQR max 0.33 > 0.12 — possible spatial mixing

### mitchell_val_absence_bare_1  (absence · val)
Bbox: [141.546159, -15.953764, 141.548024, -15.951436]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.53 | -0.02 | 0.56 | -4.92 | 0.11 | ✗ fail |
| 2018 | 0.55 | 0.17 | 0.39 | -4.97 | 0.16 | ✗ fail |
| 2019 | 0.56 | -0.05 | 0.61 | -5.22 | 0.23 | ✗ fail |
| 2020 | 0.53 | 0.00 | 0.53 | -4.85 | 0.17 | ✗ fail |
| 2021 | 0.66 | 0.12 | 0.54 | -5.35 | 0.17 | ✗ fail |
| 2022 | 0.59 | 0.09 | 0.49 | -5.45 | 0.13 | ✗ fail |
| 2023 | 0.59 | 0.07 | 0.52 | -6.90 | 0.36 | ✗ fail |
| 2024 | 0.67 | 0.12 | 0.56 | -6.43 | 0.05 | ✗ fail |
| 2025 | 0.55 | 0.15 | 0.40 | -6.26 | 0.04 | ✗ fail |
| **All-year** | 0.53–0.67 | -0.05–0.17 | 0.39–0.61 | -6.90–-4.85 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-01-20 (-0.00) — likely cloud/shadow
- 2018: IQR max 0.16 > 0.12 — possible spatial mixing
- 2019: IQR max 0.23 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-04-19 (-0.08) — likely cloud/shadow
- 2020: IQR max 0.17 > 0.12 — possible spatial mixing
- 2021: IQR max 0.17 > 0.12 — possible spatial mixing
- 2022: IQR max 0.13 > 0.12 — possible spatial mixing
- 2023: IQR max 0.36 > 0.12 — possible spatial mixing

## Riparian regions

### mitchell_absence_riparian_1  (absence · train)
Bbox: [141.757433, -15.904606, 141.761327, -15.902438]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.79 | 0.39 | 0.40 | -4.37 | 0.24 | ⚠ flag |
| 2018 | 0.82 | 0.14 | 0.68 | -4.30 | 0.22 | ✗ fail |
| 2019 | 0.78 | 0.19 | 0.59 | -4.57 | 0.24 | ✗ fail |
| 2020 | 0.77 | 0.12 | 0.64 | -4.64 | 0.25 | ✗ fail |
| 2021 | 0.81 | 0.44 | 0.38 | -4.78 | 0.27 | ⚠ flag |
| 2022 | 0.81 | 0.24 | 0.56 | -4.97 | 0.29 | ✗ fail |
| 2023 | 0.83 | 0.18 | 0.65 | -4.84 | 0.35 | ✗ fail |
| 2024 | 0.80 | 0.45 | 0.36 | -4.70 | 0.24 | ⚠ flag |
| 2025 | 0.82 | 0.47 | 0.36 | -4.80 | 0.39 | ⚠ flag |
| **All-year** | 0.77–0.83 | 0.12–0.47 | 0.36–0.68 | -4.97–-4.30 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.24 > 0.12 — possible spatial mixing
- 2018: IQR max 0.22 > 0.12 — possible spatial mixing
- 2019: IQR max 0.24 > 0.12 — possible spatial mixing
- 2020: IQR max 0.25 > 0.12 — possible spatial mixing
- 2021: IQR max 0.27 > 0.12 — possible spatial mixing
- 2022: IQR max 0.29 > 0.12 — possible spatial mixing
- 2023: IQR max 0.35 > 0.12 — possible spatial mixing
- 2024: IQR max 0.24 > 0.12 — possible spatial mixing
- 2025: IQR max 0.39 > 0.12 — possible spatial mixing

### mitchell_absence_riparian_2  (absence · train)
Bbox: [141.989778, -15.915257, 141.993131, -15.912756]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.69 | 0.00 | 0.69 | -4.94 | 0.09 | ✗ fail |
| 2018 | 0.68 | 0.21 | 0.47 | -4.49 | 0.20 | ✗ fail |
| 2019 | 0.73 | 0.18 | 0.55 | -4.59 | 0.36 | ✗ fail |
| 2020 | 0.67 | 0.21 | 0.46 | -4.64 | 0.43 | ✗ fail |
| 2021 | 0.73 | 0.37 | 0.36 | -4.64 | 0.20 | ⚠ flag |
| 2022 | 0.73 | 0.22 | 0.51 | -4.57 | 0.11 | ✗ fail |
| 2023 | 0.69 | 0.35 | 0.34 | -4.44 | 0.08 | ✓ pass |
| 2024 | 0.70 | 0.46 | 0.24 | -4.74 | 0.12 | ✗ fail |
| 2025 | 0.70 | 0.36 | 0.34 | -4.64 | 0.24 | ⚠ flag |
| **All-year** | 0.67–0.73 | 0.00–0.46 | 0.24–0.69 | -4.94–-4.44 | — | **✗ fail** |

**Notes:**
- 2018: IQR max 0.20 > 0.12 — possible spatial mixing
- 2019: IQR max 0.36 > 0.12 — possible spatial mixing
- 2020: IQR max 0.43 > 0.12 — possible spatial mixing
- 2021: IQR max 0.20 > 0.12 — possible spatial mixing
- 2024: IQR max 0.12 > 0.12 — possible spatial mixing
- 2025: IQR max 0.24 > 0.12 — possible spatial mixing

### mitchell_val_absence_riparian_1  (absence · val)
Bbox: [141.626992, -15.86482, 141.628852, -15.863669]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.85 | 0.32 | 0.52 | -4.08 | 0.16 | ⚠ flag |
| 2018 | 0.86 | 0.36 | 0.49 | -4.42 | 0.19 | ⚠ flag |
| 2019 | 0.83 | 0.26 | 0.57 | -4.79 | 0.17 | ✗ fail |
| 2020 | 0.82 | 0.35 | 0.47 | -4.50 | 0.15 | ⚠ flag |
| 2021 | 0.88 | 0.08 | 0.80 | -4.64 | 0.27 | ✗ fail |
| 2022 | 0.84 | 0.42 | 0.42 | -4.54 | 0.17 | ⚠ flag |
| 2023 | 0.87 | 0.36 | 0.51 | -4.53 | 0.14 | ⚠ flag |
| 2024 | 0.87 | 0.45 | 0.42 | -4.88 | 0.20 | ⚠ flag |
| 2025 | 0.89 | 0.48 | 0.41 | -4.68 | 0.12 | ⚠ flag |
| **All-year** | 0.82–0.89 | 0.08–0.48 | 0.41–0.80 | -4.88–-4.08 | — | **✗ fail** |

**Notes:**
- 2017: IQR max 0.16 > 0.12 — possible spatial mixing
- 2018: IQR max 0.19 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-02-19 (0.02) — likely cloud/shadow
- 2019: IQR max 0.17 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-02-09 (0.05) — likely cloud/shadow
- 2020: IQR max 0.15 > 0.12 — possible spatial mixing
- 2021: IQR max 0.27 > 0.12 — possible spatial mixing
- 2022: IQR max 0.17 > 0.12 — possible spatial mixing
- 2023: IQR max 0.14 > 0.12 — possible spatial mixing
- 2024: IQR max 0.20 > 0.12 — possible spatial mixing

## Water regions

### mitchell_absence_water_1  (absence · train)
Bbox: [141.563041, -15.858784, 141.563571, -15.857719]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | -0.35 | -0.98 | 0.62 | -1.03 | 0.13 | ✓ pass |
| 2018 | -0.11 | -0.96 | 0.85 | -1.58 | 0.50 | ✓ pass |
| 2019 | -0.13 | -0.85 | 0.73 | -1.38 | 0.45 | ✓ pass |
| 2020 | 0.01 | -0.88 | 0.89 | -1.43 | 0.49 | ⚠ flag |
| 2021 | -0.16 | -0.98 | 0.82 | -1.74 | 0.45 | ✓ pass |
| 2022 | 0.11 | -0.76 | 0.87 | -2.75 | 0.16 | ✗ fail |
| 2023 | -0.45 | -0.74 | 0.29 | -2.24 | 0.18 | ✓ pass |
| 2024 | -0.27 | -0.97 | 0.71 | -2.04 | 0.18 | ✓ pass |
| 2025 | -0.46 | -0.82 | 0.36 | -1.79 | 0.12 | ✓ pass |
| **All-year** | -0.46–0.11 | -0.98–-0.74 | 0.29–0.89 | -2.75–-1.03 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-03-01 (-0.53) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-10 (-0.35) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-30 (-0.69) — likely cloud/shadow
- 2017: IQR max 0.13 > 0.12 — possible spatial mixing
- 2018: near-zero NDVI 2018-01-10 (-0.11) — likely cloud/shadow
- 2018: near-zero NDVI 2018-02-09 (-0.49) — likely cloud/shadow
- 2018: near-zero NDVI 2018-02-19 (-0.41) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-16 (-0.46) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-31 (-0.45) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-05 (-0.87) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-10 (-0.49) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-15 (-0.59) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-20 (-0.96) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-25 (-0.75) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-15 (-0.89) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-25 (-0.96) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-30 (-0.73) — likely cloud/shadow
- 2018: IQR max 0.50 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-01-05 (-0.21) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-10 (-0.31) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-15 (-0.30) — likely cloud/shadow
- 2019: near-zero NDVI 2019-02-14 (-0.39) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-01 (-0.58) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-06 (-0.44) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-26 (-0.13) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-10 (-0.53) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-15 (-0.55) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-20 (-0.65) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-25 (-0.71) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-05 (-0.89) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-10 (-0.55) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-25 (-0.62) — likely cloud/shadow
- 2019: IQR max 0.45 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-01-05 (-0.43) — likely cloud/shadow
- 2020: near-zero NDVI 2020-01-15 (0.01) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-04 (-0.36) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-09 (-0.26) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-14 (-0.23) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-20 (-0.50) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-14 (-0.92) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-19 (-0.34) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-24 (-0.97) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-29 (-0.78) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-09 (-0.43) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-29 (-0.54) — likely cloud/shadow
- 2020: IQR max 0.49 > 0.12 — possible spatial mixing
- 2021: near-zero NDVI 2021-01-29 (-0.16) — likely cloud/shadow
- 2021: near-zero NDVI 2021-02-13 (-0.19) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-05 (-0.38) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-10 (-0.64) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-25 (-0.41) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-14 (-0.26) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-19 (-0.51) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-04 (-0.33) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-09 (-0.37) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-14 (-0.90) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-19 (-0.67) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-24 (-0.90) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-29 (-0.70) — likely cloud/shadow
- 2021: IQR max 0.45 > 0.12 — possible spatial mixing
- 2022: near-zero NDVI 2022-01-09 (-0.37) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-08 (-0.10) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-13 (-0.22) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-28 (-0.38) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-10 (-0.44) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-20 (-0.29) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-30 (-0.13) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-04 (-0.93) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-09 (-0.65) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-14 (-0.75) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-29 (-0.68) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-04 (-0.69) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-14 (-0.77) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-19 (-0.67) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-24 (-0.36) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-29 (-0.33) — likely cloud/shadow
- 2022: IQR max 0.16 > 0.12 — possible spatial mixing
- 2023: near-zero NDVI 2023-01-09 (-0.46) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-10 (-0.54) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-25 (-0.52) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-30 (-0.52) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-04 (-0.45) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-24 (-0.63) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-29 (-0.46) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-04 (-0.58) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-09 (-0.77) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-14 (-0.77) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-19 (-0.95) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-24 (-0.92) — likely cloud/shadow
- 2023: IQR max 0.18 > 0.12 — possible spatial mixing
- 2024: near-zero NDVI 2024-01-24 (-0.39) — likely cloud/shadow
- 2024: near-zero NDVI 2024-03-04 (-0.27) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-03 (-0.49) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-08 (-0.50) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-18 (-0.35) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-28 (-0.49) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-08 (-0.56) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-13 (-0.80) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-23 (-0.89) — likely cloud/shadow
- 2024: IQR max 0.18 > 0.12 — possible spatial mixing
- 2025: near-zero NDVI 2025-02-22 (-0.59) — likely cloud/shadow
- 2025: near-zero NDVI 2025-02-27 (-0.51) — likely cloud/shadow
- 2025: near-zero NDVI 2025-03-09 (-0.46) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-28 (-0.82) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-30 (-0.84) — likely cloud/shadow
- 2025: near-zero NDVI 2025-05-18 (-0.78) — likely cloud/shadow
- 2025: IQR max 0.12 > 0.12 — possible spatial mixing

### mitchell_absence_water_2  (absence · train)
Bbox: [141.561771, -15.858072, 141.562854, -15.857077]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | -0.11 | -0.99 | 0.88 | -1.63 | 0.46 | ✓ pass |
| 2018 | -0.11 | -0.92 | 0.81 | -1.38 | 0.49 | ✓ pass |
| 2019 | -0.13 | -0.86 | 0.73 | -1.38 | 0.44 | ✓ pass |
| 2020 | 0.01 | -0.88 | 0.90 | -1.66 | 0.57 | ⚠ flag |
| 2021 | -0.19 | -0.99 | 0.80 | -2.28 | 0.39 | ✓ pass |
| 2022 | -0.11 | -0.72 | 0.61 | -3.31 | 0.14 | ✓ pass |
| 2023 | -0.45 | -0.72 | 0.27 | -2.93 | 0.16 | ✓ pass |
| 2024 | -0.02 | -0.84 | 0.82 | -2.04 | 0.27 | ✓ pass |
| 2025 | -0.47 | -0.77 | 0.30 | -2.06 | 0.12 | ✓ pass |
| **All-year** | -0.47–0.01 | -0.99–-0.72 | 0.27–0.90 | -3.31–-1.38 | — | **⚠ flag** |

**Notes:**
- 2017: near-zero NDVI 2017-03-01 (-0.53) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-10 (-0.11) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-30 (-0.68) — likely cloud/shadow
- 2017: IQR max 0.46 > 0.12 — possible spatial mixing
- 2018: near-zero NDVI 2018-01-10 (-0.11) — likely cloud/shadow
- 2018: near-zero NDVI 2018-02-09 (-0.48) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-16 (-0.45) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-31 (-0.41) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-05 (-0.87) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-10 (-0.49) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-15 (-0.59) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-20 (-0.95) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-25 (-0.77) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-15 (-0.83) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-25 (-0.93) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-30 (-0.70) — likely cloud/shadow
- 2018: IQR max 0.49 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-01-05 (-0.21) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-10 (-0.30) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-15 (-0.30) — likely cloud/shadow
- 2019: near-zero NDVI 2019-02-14 (-0.38) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-01 (-0.57) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-26 (-0.13) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-10 (-0.52) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-15 (-0.55) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-20 (-0.66) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-25 (-0.71) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-05 (-0.85) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-10 (-0.53) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-25 (-0.62) — likely cloud/shadow
- 2019: IQR max 0.44 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-01-05 (-0.43) — likely cloud/shadow
- 2020: near-zero NDVI 2020-01-10 (-0.06) — likely cloud/shadow
- 2020: near-zero NDVI 2020-01-15 (0.00) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-04 (-0.35) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-09 (-0.26) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-14 (0.01) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-15 (-0.15) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-20 (-0.50) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-14 (-0.92) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-19 (-0.31) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-24 (-0.95) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-29 (-0.75) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-09 (-0.42) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-29 (-0.54) — likely cloud/shadow
- 2020: IQR max 0.57 > 0.12 — possible spatial mixing
- 2021: near-zero NDVI 2021-01-29 (-0.21) — likely cloud/shadow
- 2021: near-zero NDVI 2021-02-13 (-0.19) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-05 (-0.30) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-10 (-0.60) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-15 (-0.43) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-25 (-0.41) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-14 (-0.26) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-19 (-0.52) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-04 (-0.34) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-09 (-0.37) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-14 (-0.90) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-19 (-0.68) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-24 (-0.90) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-29 (-0.67) — likely cloud/shadow
- 2021: IQR max 0.39 > 0.12 — possible spatial mixing
- 2022: near-zero NDVI 2022-01-09 (-0.36) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-08 (-0.11) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-13 (-0.23) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-28 (-0.36) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-10 (-0.44) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-20 (-0.27) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-30 (-0.16) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-04 (-0.88) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-09 (-0.65) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-14 (-0.75) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-29 (-0.68) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-04 (-0.70) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-14 (-0.76) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-19 (-0.72) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-24 (-0.44) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-29 (-0.34) — likely cloud/shadow
- 2022: IQR max 0.14 > 0.12 — possible spatial mixing
- 2023: near-zero NDVI 2023-01-09 (-0.47) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-10 (-0.55) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-25 (-0.49) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-30 (-0.53) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-04 (-0.45) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-24 (-0.63) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-29 (-0.46) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-04 (-0.59) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-09 (-0.77) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-14 (-0.79) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-19 (-0.94) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-24 (-0.92) — likely cloud/shadow
- 2023: IQR max 0.16 > 0.12 — possible spatial mixing
- 2024: near-zero NDVI 2024-01-24 (-0.27) — likely cloud/shadow
- 2024: near-zero NDVI 2024-02-18 (-0.02) — likely cloud/shadow
- 2024: near-zero NDVI 2024-03-04 (-0.27) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-08 (-0.48) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-18 (-0.34) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-28 (-0.49) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-08 (-0.56) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-13 (-0.79) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-23 (-0.82) — likely cloud/shadow
- 2024: IQR max 0.27 > 0.12 — possible spatial mixing
- 2025: near-zero NDVI 2025-02-22 (-0.61) — likely cloud/shadow
- 2025: near-zero NDVI 2025-02-27 (-0.51) — likely cloud/shadow
- 2025: near-zero NDVI 2025-03-09 (-0.47) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-28 (-0.83) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-30 (-0.85) — likely cloud/shadow
- 2025: near-zero NDVI 2025-05-18 (-0.76) — likely cloud/shadow
- 2025: IQR max 0.12 > 0.12 — possible spatial mixing

### mitchell_absence_water_3  (absence · train)
Bbox: [141.56677, -15.86149, 141.567707, -15.860561]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.02 | -0.99 | 1.01 | -1.15 | 0.40 | ⚠ flag |
| 2018 | 0.19 | -0.91 | 1.10 | -1.20 | 0.64 | ✗ fail |
| 2019 | 0.13 | -0.82 | 0.96 | -1.29 | 0.55 | ✗ fail |
| 2020 | -0.03 | -0.87 | 0.84 | -1.65 | 0.56 | ✓ pass |
| 2021 | -0.10 | -0.99 | 0.89 | -2.02 | 0.46 | ✓ pass |
| 2022 | 0.05 | -0.61 | 0.66 | -2.84 | 0.44 | ⚠ flag |
| 2023 | -0.40 | -0.66 | 0.25 | -2.51 | 0.47 | ✓ pass |
| 2024 | 0.26 | -0.74 | 1.00 | -1.64 | 0.27 | ✗ fail |
| 2025 | -0.43 | -0.71 | 0.28 | -1.38 | 0.11 | ✓ pass |
| **All-year** | -0.43–0.26 | -0.99–-0.61 | 0.25–1.10 | -2.84–-1.15 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-03-01 (-0.53) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-10 (-0.34) — likely cloud/shadow
- 2017: near-zero NDVI 2017-04-30 (0.02) — likely cloud/shadow
- 2017: IQR max 0.40 > 0.12 — possible spatial mixing
- 2018: near-zero NDVI 2018-01-10 (-0.10) — likely cloud/shadow
- 2018: near-zero NDVI 2018-02-09 (-0.48) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-16 (-0.42) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-31 (-0.44) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-05 (-0.76) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-10 (-0.44) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-15 (-0.53) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-20 (-0.82) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-25 (-0.68) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-15 (-0.83) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-25 (-0.84) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-30 (-0.61) — likely cloud/shadow
- 2018: IQR max 0.64 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-01-05 (-0.20) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-10 (-0.30) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-15 (-0.29) — likely cloud/shadow
- 2019: near-zero NDVI 2019-02-14 (-0.40) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-01 (-0.56) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-06 (-0.46) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-26 (-0.12) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-10 (-0.51) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-15 (-0.52) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-20 (-0.63) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-25 (-0.68) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-05 (-0.73) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-10 (-0.12) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-25 (-0.60) — likely cloud/shadow
- 2019: IQR max 0.55 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-01-15 (-0.03) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-09 (-0.25) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-15 (-0.17) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-20 (-0.48) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-25 (-0.11) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-14 (-0.86) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-19 (-0.38) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-24 (-0.94) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-29 (-0.75) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-09 (-0.11) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-29 (-0.53) — likely cloud/shadow
- 2020: IQR max 0.56 > 0.12 — possible spatial mixing
- 2021: near-zero NDVI 2021-01-29 (-0.29) — likely cloud/shadow
- 2021: near-zero NDVI 2021-02-13 (-0.20) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-05 (-0.50) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-10 (-0.54) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-25 (-0.36) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-09 (-0.42) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-14 (-0.23) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-19 (-0.48) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-29 (-0.10) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-04 (-0.32) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-09 (-0.36) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-14 (-0.75) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-19 (-0.63) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-24 (-0.85) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-29 (-0.62) — likely cloud/shadow
- 2021: IQR max 0.46 > 0.12 — possible spatial mixing
- 2022: near-zero NDVI 2022-01-09 (-0.37) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-13 (-0.22) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-28 (-0.36) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-10 (-0.43) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-20 (-0.28) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-30 (-0.11) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-04 (-0.59) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-09 (-0.60) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-14 (-0.70) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-29 (-0.58) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-04 (-0.62) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-14 (-0.65) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-24 (-0.10) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-29 (-0.30) — likely cloud/shadow
- 2022: IQR max 0.44 > 0.12 — possible spatial mixing
- 2023: near-zero NDVI 2023-01-09 (-0.47) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-10 (-0.57) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-25 (-0.65) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-30 (-0.51) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-04 (-0.44) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-24 (-0.60) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-29 (-0.40) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-04 (-0.54) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-09 (-0.71) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-14 (-0.72) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-19 (-0.84) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-24 (-0.84) — likely cloud/shadow
- 2023: IQR max 0.47 > 0.12 — possible spatial mixing
- 2024: near-zero NDVI 2024-01-24 (-0.39) — likely cloud/shadow
- 2024: near-zero NDVI 2024-02-18 (-0.35) — likely cloud/shadow
- 2024: near-zero NDVI 2024-03-04 (-0.33) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-03 (-0.55) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-08 (-0.44) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-18 (-0.33) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-28 (-0.46) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-08 (-0.49) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-13 (-0.74) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-18 (-0.31) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-23 (-0.68) — likely cloud/shadow
- 2024: IQR max 0.27 > 0.12 — possible spatial mixing
- 2025: near-zero NDVI 2025-02-22 (-0.59) — likely cloud/shadow
- 2025: near-zero NDVI 2025-02-27 (-0.47) — likely cloud/shadow
- 2025: near-zero NDVI 2025-03-09 (-0.43) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-28 (-0.78) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-30 (-0.80) — likely cloud/shadow
- 2025: near-zero NDVI 2025-05-18 (-0.68) — likely cloud/shadow

### mitchell_val_absence_water_1  (absence · val)
Bbox: [141.563797, -15.85329, 141.565461, -15.852743]  Tile: 54LWH

| Year | Wet NDVI (peak) | Dry NDVI (floor) | Wet–dry Δ | VH/VV dry (dB) | IQR max | Verdict |
|------|----------------|-----------------|-----------|---------------|---------|---------|
| 2017 | 0.41 | -0.64 | 1.05 | -2.03 | 1.07 | ✗ fail |
| 2018 | 0.68 | -0.45 | 1.13 | -2.22 | 1.34 | ✗ fail |
| 2019 | 0.66 | -0.49 | 1.15 | -2.30 | 1.10 | ✗ fail |
| 2020 | 0.16 | -0.59 | 0.75 | -2.66 | 0.62 | ✗ fail |
| 2021 | 0.78 | -0.69 | 1.47 | -2.67 | 1.23 | ✗ fail |
| 2022 | 0.59 | -0.36 | 0.94 | -3.00 | 0.48 | ✗ fail |
| 2023 | -0.04 | -0.34 | 0.30 | -2.81 | 0.26 | ✓ pass |
| 2024 | 0.18 | -0.39 | 0.57 | -2.60 | 0.96 | ✗ fail |
| 2025 | 0.30 | -0.37 | 0.67 | -2.34 | 0.41 | ✗ fail |
| **All-year** | -0.04–0.78 | -0.69–-0.34 | 0.30–1.47 | -3.00–-2.03 | — | **✗ fail** |

**Notes:**
- 2017: near-zero NDVI 2017-03-01 (-0.28) — likely cloud/shadow
- 2017: IQR max 1.07 > 0.12 — possible spatial mixing
- 2018: near-zero NDVI 2018-01-10 (-0.01) — likely cloud/shadow
- 2018: near-zero NDVI 2018-02-09 (-0.17) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-16 (-0.23) — likely cloud/shadow
- 2018: near-zero NDVI 2018-03-31 (-0.02) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-05 (0.00) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-10 (-0.15) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-20 (-0.25) — likely cloud/shadow
- 2018: near-zero NDVI 2018-04-25 (-0.27) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-15 (-0.33) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-25 (-0.43) — likely cloud/shadow
- 2018: near-zero NDVI 2018-05-30 (-0.34) — likely cloud/shadow
- 2018: IQR max 1.34 > 0.12 — possible spatial mixing
- 2019: near-zero NDVI 2019-01-05 (-0.06) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-10 (-0.19) — likely cloud/shadow
- 2019: near-zero NDVI 2019-01-15 (-0.18) — likely cloud/shadow
- 2019: near-zero NDVI 2019-02-14 (-0.33) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-06 (-0.11) — likely cloud/shadow
- 2019: near-zero NDVI 2019-03-26 (-0.05) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-10 (-0.31) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-15 (-0.34) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-20 (-0.40) — likely cloud/shadow
- 2019: near-zero NDVI 2019-04-25 (-0.45) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-05 (-0.39) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-10 (-0.21) — likely cloud/shadow
- 2019: near-zero NDVI 2019-05-25 (-0.41) — likely cloud/shadow
- 2019: IQR max 1.10 > 0.12 — possible spatial mixing
- 2020: near-zero NDVI 2020-01-05 (-0.17) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-04 (-0.23) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-09 (-0.14) — likely cloud/shadow
- 2020: near-zero NDVI 2020-02-14 (-0.08) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-15 (-0.15) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-20 (-0.28) — likely cloud/shadow
- 2020: near-zero NDVI 2020-03-25 (-0.07) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-14 (-0.36) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-19 (-0.19) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-24 (-0.54) — likely cloud/shadow
- 2020: near-zero NDVI 2020-04-29 (-0.37) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-09 (-0.08) — likely cloud/shadow
- 2020: near-zero NDVI 2020-05-29 (-0.18) — likely cloud/shadow
- 2020: IQR max 0.62 > 0.12 — possible spatial mixing
- 2021: near-zero NDVI 2021-01-29 (-0.11) — likely cloud/shadow
- 2021: near-zero NDVI 2021-03-05 (-0.27) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-14 (-0.10) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-19 (-0.37) — likely cloud/shadow
- 2021: near-zero NDVI 2021-04-29 (-0.12) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-04 (-0.20) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-09 (-0.23) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-14 (-0.40) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-19 (-0.39) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-24 (-0.53) — likely cloud/shadow
- 2021: near-zero NDVI 2021-05-29 (-0.41) — likely cloud/shadow
- 2021: IQR max 1.23 > 0.12 — possible spatial mixing
- 2022: near-zero NDVI 2022-02-08 (-0.03) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-13 (-0.13) — likely cloud/shadow
- 2022: near-zero NDVI 2022-02-23 (-0.06) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-10 (-0.28) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-15 (-0.04) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-20 (-0.13) — likely cloud/shadow
- 2022: near-zero NDVI 2022-03-30 (-0.08) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-04 (-0.31) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-09 (-0.35) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-14 (-0.41) — likely cloud/shadow
- 2022: near-zero NDVI 2022-04-29 (-0.27) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-04 (-0.34) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-09 (-0.11) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-14 (-0.36) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-24 (-0.21) — likely cloud/shadow
- 2022: near-zero NDVI 2022-05-29 (-0.06) — likely cloud/shadow
- 2022: IQR max 0.48 > 0.12 — possible spatial mixing
- 2023: near-zero NDVI 2023-01-09 (-0.40) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-10 (-0.41) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-15 (-0.04) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-20 (-0.36) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-25 (-0.21) — likely cloud/shadow
- 2023: near-zero NDVI 2023-03-30 (-0.33) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-04 (-0.23) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-19 (-0.11) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-24 (-0.37) — likely cloud/shadow
- 2023: near-zero NDVI 2023-04-29 (-0.08) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-04 (-0.28) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-09 (-0.47) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-14 (-0.43) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-19 (-0.51) — likely cloud/shadow
- 2023: near-zero NDVI 2023-05-24 (-0.47) — likely cloud/shadow
- 2023: IQR max 0.26 > 0.12 — possible spatial mixing
- 2024: near-zero NDVI 2024-01-24 (-0.31) — likely cloud/shadow
- 2024: near-zero NDVI 2024-02-18 (-0.21) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-08 (-0.20) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-18 (-0.16) — likely cloud/shadow
- 2024: near-zero NDVI 2024-04-28 (-0.29) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-08 (-0.33) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-13 (-0.45) — likely cloud/shadow
- 2024: near-zero NDVI 2024-05-23 (-0.42) — likely cloud/shadow
- 2024: IQR max 0.96 > 0.12 — possible spatial mixing
- 2025: near-zero NDVI 2025-02-22 (-0.38) — likely cloud/shadow
- 2025: near-zero NDVI 2025-02-27 (-0.29) — likely cloud/shadow
- 2025: near-zero NDVI 2025-03-09 (-0.18) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-28 (-0.45) — likely cloud/shadow
- 2025: near-zero NDVI 2025-04-30 (-0.51) — likely cloud/shadow
- 2025: near-zero NDVI 2025-05-18 (-0.41) — likely cloud/shadow
- 2025: IQR max 0.41 > 0.12 — possible spatial mixing

