#!/bin/bash

# This script runs a PySpark script multiple times with different memory values.

SPARK_HOME="/usr/local/spark"

PYSPARK_SCRIPT="./ProjectDat535/sparkreadertest.py"

NUM_RUNS=10

MEMORY_VALUES=("1g" "2g" "3g" "4g" "5g" "6g" "7g" "8g" "9g" "10g")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Output file for recording execution times
OUTPUT_FILE="execution_times_${TIMESTAMP}.txt"
# Run the PySpark script multiple times with different memory values
for ((i=1; i<=$NUM_RUNS; i++)); do
  # Choose the memory value for this run
  MEMORY="${MEMORY_VALUES[i % ${#MEMORY_VALUES[@]}]}" # Cycling through the array

  echo "Run $i with $MEMORY of memory:"

  # Record start time
  START_TIME=$(date +%s.%N)

  # Run PySpark with the specified memory
  $SPARK_HOME/bin/spark-submit \
    --master local[*] \
    --executor-memory $MEMORY \
    $PYSPARK_SCRIPT

  # Record end time
  END_TIME=$(date +%s.%N)

  # Calculate and print execution time
  ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)
  echo "Execution time: $ELAPSED_TIME seconds"

  # Write execution time to the output file
  echo "Run $i with $MEMORY of memory: $ELAPSED_TIME seconds" >> $OUTPUT_FILE

  # Add a separator line between runs
  echo "------------------------------------------"

done

echo "All runs completed. Execution times are recorded in $OUTPUT_FILE"
