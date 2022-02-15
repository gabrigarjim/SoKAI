#include "SKLayer.h"

/* ----- Standard Constructor ----- */
SKLayer::SKLayer(int size, string activation) {

  fSize = size;

  SKNeuron neuron(activation);

  for ( int i = 0; i < fSize ; i++)
    vNeurons.push_back(neuron);

}

/* ----- Standard Destructor ----- */
SKLayer::~SKLayer(){}
