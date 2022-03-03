#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKBackProp.h"

int main () {

  cout<<"##############################################################"<<endl;
  cout<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #"<<endl;
  cout<<"##############################################################"<<endl;

  int seed = 2021;

  std::vector<float> v;
  v.push_back(0.3);
  v.push_back(0.7);

  std::vector<float> targetVector;
  targetVector.push_back(1.0);



  SKWeights *input_layer_weights = new SKWeights(2,1);
  SKLayer   *layer_1 = new SKLayer(2,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(2,1);

  SKLayer   *layer_2 = new SKLayer(1,"Sigmoid");

  SKPropagator * prop_i1 = new SKPropagator();
  SKPropagator * prop_12 = new SKPropagator();

  SKBackProp *backPropagator = new SKBackProp();
  backPropagator->SetLoss("Quadratic");

  input_layer_weights->Init(seed);
  weights_12->Init(seed);


  for(int n = 0 ; n < 10000 ; n++) {
   prop_i1->Feed(&v,layer_1,input_layer_weights);
   prop_12->Propagate(layer_1,layer_2,weights_12);

  cout<<"=========== Loss : " <<backPropagator->Loss(layer_2,&targetVector)<<endl;

  backPropagator->CalculateGradients(weights_12,layer_1,layer_2,&targetVector);

  layer_2->Print();

  layer_1->Clear();
  layer_2->Clear();
}
  return 0;
}
