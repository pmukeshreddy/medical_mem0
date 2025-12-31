#!/bin/bash

# Download and run Synthea synthetic patient generator
# Usage: ./download_synthea.sh [num_patients]

NUM_PATIENTS=${1:-10000}
DATA_DIR="../data/raw/synthea"

echo "=== Synthea Synthetic Patient Generator ==="
echo "Generating $NUM_PATIENTS patients..."

# Create directories
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit 1

# Download Synthea if not exists
if [ ! -f "synthea-with-dependencies.jar" ]; then
    echo "Downloading Synthea..."
    curl -L -O https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
fi

# Check Java
if ! command -v java &> /dev/null; then
    echo "ERROR: Java not found. Install Java 11+ first."
    echo "  Ubuntu: sudo apt install openjdk-11-jre"
    echo "  Mac: brew install openjdk@11"
    exit 1
fi

# Run Synthea
echo "Generating patients (this may take a while)..."
java -jar synthea-with-dependencies.jar \
    -p "$NUM_PATIENTS" \
    --exporter.years_of_history=10 \
    --exporter.fhir.export=true \
    --exporter.csv.export=true \
    --exporter.baseDirectory=./output

# Move outputs
mv output/fhir ./fhir 2>/dev/null
mv output/csv ./csv 2>/dev/null
rm -rf output

echo ""
echo "=== Done ==="
echo "FHIR JSON: $DATA_DIR/fhir/"
echo "CSV files: $DATA_DIR/csv/"
echo ""
echo "Next: python process_synthea.py"
