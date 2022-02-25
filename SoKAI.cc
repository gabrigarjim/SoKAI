#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"

int main () {

  int seed = 2021;

  std::vector<float> *v;
  v->push_back(6.0);

  SKWeights *input_layer_weights = new SKWeights(1,1);
  SKLayer   *layer_1 = new SKLayer(1,"Linear");
  SKWeights *weights_12 = new SKWeights(1,2);

  SKLayer   *layer_2 = new SKLayer(2,"Linear");
  SKWeights *weights_23 = new SKWeights(2,1);
  SKLayer   *layer_3 = new SKLayer(1,"Linear");

  SKPropagator * prop_i1 = new SKPropagator();
  SKPropagator * prop_12 = new SKPropagator();
  SKPropagator * prop_23 = new SKPropagator();


  input_layer_weights->Init(seed-1);
  weights_12->Init(seed);
  weights_23->Init(seed+1);



   prop_i1->Feed(v,layer_1,input_layer_weights);
   prop_12->Propagate(layer_1,layer_2,weights_12);
   prop_23->Propagate(layer_2,layer_3,weights_23);


  cout<<"Input Layer Weights :"<<endl;
  input_layer_weights->Print();

  cout<<"Weights Layer 1-2 :"<<endl;
  weights_12->Print();

  cout<<"Weights Layer 2-3 :"<<endl;
  weights_23->Print();

  cout<<" Model Output : "<<endl;
  layer_3->Print();

  return 0;
}
