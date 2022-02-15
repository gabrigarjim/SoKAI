#include "SKNeuron.h"
#include "SKLayer.h"

int main () {

  SKNeuron *neuron = new SKNeuron("Sigmoid");

  neuron->Input(1.787);

  std::cout<<"Neuron Output : "<<neuron->Output()<<std::endl;

  SKLayer *layer = new SKLayer(23,"Sigmoid");
  


  return 0;
}
