# Dataset Sources

## Training Dataset — 15,898 Images (After Deduplication)

| Class | Source | Images | Institution | License |
|---|---|---|---|---|
| COVID-19 | COVID-19 Radiography Database | 3,949 | Qatar University | CC BY 4.0 |
| Pneumonia | Chest X-Ray Images (Paul Mooney) | 3,976 | Guangzhou Women & Children's MC | CC BY 4.0 |
| TB | TB Chest X-ray Dataset (Rahman) | ~700 | Multiple | CC BY 4.0 |
| TB | TBX11K (Liu et al. CVPR 2020) | ~799 | Multiple | Research |
| TB | Mendeley TB (Kiran 2024) | ~2,494 | Multiple | CC BY 4.0 |
| Normal | Multiple sources | 4,000 | Multiple | Various |

## External Validation Datasets

| Dataset | Institution | Country | Images | License |
|---|---|---|---|---|
| NIH ChestX-ray14 | NIH Clinical Center | USA | 176 (test) | CC0 |
| Epic Chittagong | Epic Chittagong Hospital | Bangladesh | 513 (test) | CC BY 4.0 |

## Data Integrity
- Total before deduplication: 16,000
- Duplicates removed (MD5): 102
- Cross-split leakage after fix: 0
- Final unique images: 15,898
