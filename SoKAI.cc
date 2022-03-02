#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"

int main () {

  cout<<"##############################################################"<<endl;
  cout<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #"<<endl;
  cout<<"##############################################################"<<endl;

  int seed = 2021;

  std::vector<float> v;
  v.push_back(0.1);
  v.push_back(0.1);

  SKWeights *input_layer_weights = new SKWeights(2,1);
  SKLayer   *layer_1 = new SKLayer(2,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(2,1);

  SKLayer   *layer_2 = new SKLayer(1,"Sigmoid");

  SKPropagator * prop_i1 = new SKPropagator();
  SKPropagator * prop_12 = new SKPropagator();


  input_layer_weights->Init(seed);
  weights_12->Init(seed);



   prop_i1->Feed(&v,layer_1,input_layer_weights);
   prop_12->Propagate(layer_1,layer_2,weights_12);


  cout<<"Input Layer Weights :"<<endl;
  input_layer_weights->Print();

  cout<<"Weights Layer 1-2 :"<<endl;
  weights_12->Print();

  cout<<" Model Output : "<<endl;
  layer_2->Print();

  return 0;
}
