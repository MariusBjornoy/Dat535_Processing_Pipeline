#!/usr/bin/env bash

# Set the path to your Python script
python_script="/home/ubuntu./ProjectDat535/sparkreadertest.py"

# Output file to save the execution times
output_file="~/ProjectDat535/execution_times.txt"

#pip install delta-spark
#pip install Delta
#pip install findspark

# Loop to run the script ten times
for ((i=1; i<=10; i++))
do
    # Record the start time
    start = $SECONDS
    #start_time=$(date +%s.%N)

    echo "Staring spark"
    #spark-submit ./ProjectDat535/sparkreadertest.py

    # Run the Python script using python3
    /bin/python3 ./ProjectDat535/sparkreadertest.py

    # Record the end time
    #end_time=$(date +%s.%N)

    # Calculate the execution time (in seconds)
    #execution_time=$(echo "$end_time - $start_time")

    execution_time=$(( SECONDS - start ))
    # Print the execution time for each run
    echo "Run $i: $execution_time seconds" >> "$output_file"
    #echo "Run $i: $execution_time seconds"
    
done

# Deactivate the virtual environment
deactivate
