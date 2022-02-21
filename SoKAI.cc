#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"

int main () {

  SKNeuron *neuron = new SKNeuron("Sigmoid");

  neuron->Input(1.787);

  std::cout<<"Neuron Output : "<<neuron->Output()<<std::endl;

  SKLayer   *layer_1 = new SKLayer(3,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(3,2);

  SKLayer   *layer_2 = new SKLayer(2,"Sigmoid");
  SKWeights *weights_23 = new SKWeights(2,1);
  SKLayer   *layer_3 = new SKLayer(1,"Sigmoid");

  weights_12->Init(2021);
  weights_23->Init(2021);

  weights_12->Print();
  weights_23->Print();

  return 0;
}
