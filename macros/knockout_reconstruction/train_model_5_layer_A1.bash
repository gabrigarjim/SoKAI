# training a five-layer model in an automated way
declare -A model_array

while IFS= read -r line
do
  ./KnockoutReconstruction $line
  sleep 1
done < five_layer_models_A.txt
