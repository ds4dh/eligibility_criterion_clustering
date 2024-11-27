#!/bin/bash

# Bash script variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)
PYTHON_SCRIPT_PATH="./src/generate_data.py"
RESULT_DIR="./experiments/experiment_3_results"
NUM_SAMPLES=100
MAX_TIME_PER_TRIAL=$((30 * 60))  # 30 minutes

# List of filter level combinations ("COND_FILTER_LVL ITRV_FILTER_LVL")
FILTER_COMBINATIONS=(
  "4 3"
)

# List of filters for selecting clinical trials for a given experiment
EVALUATED_CT_COND_TYPE_FILTERS=(
  "C01"
  "C04"
  "C14"
  "C20"
)

# Function to get the current number of rows in the CSV file using a Python script
get_csv_row_count() {
  python - <<END
import csv
try:
    with open("$1", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    print(row_count)
except FileNotFoundError:
    print(0)
END
}

# Loop through each filter combination
for FILTER_COMBO in "${FILTER_COMBINATIONS[@]}"; do
  for EVALUATED_CT_COND_TYPE_FILTER in "${EVALUATED_CT_COND_TYPE_FILTERS[@]}"; do
    # Variables passed as arguments to the python script
    IFS=' ' read -r COND_FILTER_LVL ITRV_FILTER_LVL <<< "$FILTER_COMBO"
    EMBED_MODEL_ID=pubmed-bert-sentence
    RESULT_PATH="${RESULT_DIR}/model-${EMBED_MODEL_ID}_type${EVALUATED_CT_COND_TYPE_FILTER}_cond-${COND_FILTER_LVL}_itrv-${ITRV_FILTER_LVL}.csv"
    DESIRED_LINE_COUNT=$((NUM_SAMPLES + 1))

    # Run the script until the result csv file has the desired number of lines
    while true; do
      # Compute line count for exit condition
      CURRENT_LINE_COUNT=$(get_csv_row_count "$RESULT_PATH")  
      echo "Current line count $CURRENT_LINE_COUNT for COND_FILTER_LVL=${COND_FILTER_LVL} ITRV_FILTER_LVL=${ITRV_FILTER_LVL}"

      # Run data generation script
      echo "Running script src/generate_data.py with EVALUATED_CT_COND_TYPE_FILTER=${EVALUATED_CT_COND_TYPE_FILTER}, COND_FILTER_LVL=${COND_FILTER_LVL} and ITRV_FILTER_LVL=${ITRV_FILTER_LVL}"
      timeout $MAX_TIME_PER_TRIAL python "$PYTHON_SCRIPT_PATH" \
        --embed_model_id $EMBED_MODEL_ID \
        --evaluation_cond_type_filter $EVALUATED_CT_COND_TYPE_FILTER \
        --cond_filter_lvl $COND_FILTER_LVL \
        --itrv_filter_lvl $ITRV_FILTER_LVL \
        --result_dir $RESULT_DIR \
        --num_samples $NUM_SAMPLES
      
      # Check for CPP error
      if [[ $? -ne 0 ]]; then
        echo "Script met an unforseen CPP or time-out error during pipeline. Continuing from before the error with a new clinical trial..."
      fi
      sleep 5

      # Check for exit condition
      if [[ $CURRENT_LINE_COUNT -ge $DESIRED_LINE_COUNT ]]; then
        echo "Desired line count reached for COND_FILTER_LVL=${COND_FILTER_LVL} ITRV_FILTER_LVL=${ITRV_FILTER_LVL}"
        break
      fi

    done
  done
done