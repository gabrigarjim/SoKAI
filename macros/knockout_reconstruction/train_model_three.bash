# training a five-layer model in an automated way
input_file=$1

declare -A model_array

while IFS= read -r line
do
  ./KnockoutReconstructionThree $line
  sleep 1
done < $input_file
