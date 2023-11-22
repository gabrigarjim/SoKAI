void generate_models(){

 // Files with model structure
 ofstream *three_layer_models = new ofstream("three_layer_models.txt");
 ofstream *five_layer_models = new ofstream("five_layer_models.txt");
 ofstream *four_layer_models = new ofstream("four_layer_models.txt");


 vector<string> vNeurons_H1 = {"8","16"};
 vector<string> vNeurons_H2 = {"8","16"};
 vector<string> vNeurons_H3 = {"8","16"};


 vector<string> vActivationFunctions_H1 = {"LeakyReLU","Sigmoid"};
 vector<string> vActivationFunctions_H2 = {"LeakyReLU","Sigmoid"};
 vector<string> vActivationFunctions_H3 = {"LeakyReLU","Sigmoid"};
 vector<string> vBatchSizes = {"8","16"};

 int counter_5 = 5000;

 for(int i = 0 ; i < vNeurons_H1.size(); i++){
  for(int j = 0 ; j < vNeurons_H2.size(); j++){
   for(int k = 0 ; k < vNeurons_H3.size(); k++){

  for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){
   for(int m = 0 ; m < vActivationFunctions_H2.size(); m++){
    for(int n = 0 ; n < vActivationFunctions_H3.size(); n++){
      for(int q = 0 ; q < vBatchSizes.size(); q++){

       counter_5++;
       *five_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(q)<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
       <<vNeurons_H3.at(k)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<vActivationFunctions_H3.at(n)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_5)<<endl;

       counter_5++;
       *five_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(q)<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
       <<vNeurons_H3.at(k)<<" "<<"LeakyReLU"<<" "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<vActivationFunctions_H3.at(n)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_5)<<endl;


     }}}}}}}


  int counter_4 = 4000;

  for(int i = 0 ; i < vNeurons_H1.size(); i++){
   for(int j = 0 ; j < vNeurons_H2.size(); j++){

   for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){
    for(int m = 0 ; m < vActivationFunctions_H2.size(); m++){
        for(int q = 0 ; q < vBatchSizes.size(); q++){


        counter_4++;
        *four_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(q)<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
        <<"LeakyReLU "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_4)<<endl;

        counter_4++;
        *four_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(q)<<" "<<vNeurons_H1.at(i)<<" "<<vNeurons_H2.at(j)<<" "
        <<"LeakyReLU "<<vActivationFunctions_H1.at(l)<<" "<<vActivationFunctions_H2.at(m)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_4)<<endl;



      }}}}}


  int counter_3 = 3000;

  for(int i = 0 ; i < vNeurons_H1.size(); i++){
    for(int l = 0 ; l < vActivationFunctions_H1.size(); l++){
      for(int k = 0 ; k < vBatchSizes.size(); k++){


        counter_3++;
        *three_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(k)<<" "<<vNeurons_H1.at(i)<<" "<<"LeakyReLU"<<" "<<
        vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<"Absolute"<<" "<<to_string(counter_3)<<endl;

        counter_3++;
        *three_layer_models<<"800"<<" "<<"30000"<<" "<<"10"<<" "<<vBatchSizes.at(k)<<" "<<vNeurons_H1.at(i)<<" "<<"LeakyReLU"<<" "<<
        vActivationFunctions_H1.at(l)<<" "<<"LeakyReLU"<<" "<<"Quadratic"<<" "<<to_string(counter_3)<<endl;



      }}}





}
