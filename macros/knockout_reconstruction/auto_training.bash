#!/bin/bash

# Array of input files
input_files=("model_file_01.txt" "model_file_02.txt" "model_file_03.txt" "model_file_04.txt" "model_file_05.txt" "model_file_06.txt" "model_file_07.txt" "model_file_08.txt" "model_file_09.txt" "model_file_10.txt" "model_file_11.txt" "model_file_12.txt" "model_file_13.txt" "model_file_14.txt" "model_file_15.txt" "model_file_16.txt")

# Number of processes
num_processes=${#input_files[@]}

# Create a new screen session
screen -S "auto_training" -d -m

# Run processes in parallel
for ((i=0; i<num_processes; i++)); do
    screen -S "auto_training" -X screen -t "Process_$((i+1))" bash -c "/home/gabri/CODE/build_sokai/KnockoutBinaries/train_model_three.bash ${input_files[i]} > logs.txt"
done

# Attach to the screen session
screen -r "auto_training"
