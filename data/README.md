# Data Directory

## Quick Start

```bash
# 1. Generate synthetic patients (requires Java 11+)
cd scripts
chmod +x download_synthea.sh
./download_synthea.sh 10000  # 10K patients

# 2. Process into MedMem0 format
python process_synthea.py

# 3. Seed into Mem0 (after backend is set up)
python seed_patients.py
```

## Directory Structure

```
data/
├── raw/
│   └── synthea/
│       ├── csv/           # Synthea CSV output
│       │   ├── patients.csv
│       │   ├── conditions.csv
│       │   ├── medications.csv
│       │   ├── encounters.csv
│       │   └── observations.csv
│       └── fhir/          # FHIR JSON (optional)
├── processed/
│   ├── patients.jsonl     # Full patient records
│   └── mem0_records.jsonl # Ready for Mem0 ingestion
└── README.md
```

## Data Schema

### patients.jsonl
```json
{
  "id": "uuid",
  "name": "John Doe",
  "birth_date": "1960-01-15",
  "gender": "M",
  "conditions": [...],
  "medications": [...],
  "encounters": [...],
  "observations": [...]
}
```

### mem0_records.jsonl
```json
{
  "user_id": "patient-uuid",
  "content": "Visit on 2024-01-15. Type: outpatient. Reason: Diabetes follow-up...",
  "metadata": {
    "type": "visit",
    "date": "2024-01-15",
    "encounter_type": "outpatient"
  }
}
```

## Generating Gold Eval Set

After processing, manually curate 100 patients for eval:

```bash
python scripts/create_gold_set.py --n 100
```

This selects patients with diverse conditions for eval coverage.
