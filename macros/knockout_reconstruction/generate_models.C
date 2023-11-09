void generate_models(){

 // Files with model structure
 ofstream *five_layer_models_Q = new ofstream("five_layer_models_Q.txt");
 ofstream *four_layer_models_Q = new ofstream("four_layer_models_Q.txt");
 ofstream *three_layer_models_Q = new ofstream("three_layer_models_Q.txt");

 ofstream *five_layer_models_A = new ofstream("five_layer_models_A.txt");
 ofstream *four_layer_models_A = new ofstream("four_layer_models_A.txt");
 ofstream *three_layer_models_A = new ofstream("three_layer_models_A.txt");


 vector<string> vNeurons_H1 = {"10","20"};
 vector<string> vNeurons_H2 = {"10","20"};
 vector<string> vNeurons_H3 = {"10","20"};

 vector<string> vActivationFunctions_H1 = {"LeakyReLU","Sigmoid"};
 vector<string> vActivationFunctions_H2 = {"LeakyReLU","Sigmoid"};
 vector<string> vActivationFunctions_H3 = {"LeakyReLU","Sigmoid"};

 int counter_5 = 500;

 for(int i = 0 ; i < vNeurons_H1.size(); i++){
  for(int j = 0 ; j < vNeurons_H2.size(); j++){
   for(int k = 0 ; k < vNeurons_H3.size(); k++){

  for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){
   for(int m = 0 ; m < vActivationFunctions_H2.size(); m++){
    for(int n = 0 ; n < vActivationFunctions_H3.size(); n++){

       counter_5++;
       *five_layer_models_A<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
       <<vNeurons_H3.at(k)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<vActivationFunctions_H3.at(n)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_5)<<endl;

       counter_5++;
       *five_layer_models_Q<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
       <<vNeurons_H3.at(k)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<vActivationFunctions_H3.at(n)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_5)<<endl;



     }}}}}}


  int counter_4 = 400;

  for(int i = 0 ; i < vNeurons_H1.size(); i++){
   for(int j = 0 ; j < vNeurons_H2.size(); j++){

   for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){
    for(int m = 0 ; m < vActivationFunctions_H2.size(); m++){

        counter_4++;
        *four_layer_models_A<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
        <<vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H2.at(m)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_4)<<endl;

        counter_4++;
        *four_layer_models_Q<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
        <<vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H2.at(m)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_4)<<endl;



      }}}}


  int counter_3 = 300;

  for(int i = 0 ; i < vNeurons_H1.size(); i++){
    for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){

        counter_3++;
        *three_layer_models_A<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<"LeakyReLU"<<" "<<
        vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_3)<<endl;

        counter_3++;
        *three_layer_models_Q<<"100"<<" "<<"10"<<" "<<"10"<<" "<<"16"<<" "<<vNeurons_H1.at(i)<<" "<<"LeakyReLU"<<" "
        <<vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_3)<<endl;



      }}





}
