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


/* ----- Public Method WriteOutput ----- */
void SKLayer::WriteOutput(){

  for ( int i = 0 ; i < fSize ; i++ )
    vLayerOutput.push_back(vNeurons.at(i).Output());

}


/* ----- Public Method Print ----- */
void SKLayer::Print(){

  for ( int i = 0 ; i < vLayerOutput.size() ; i++ )
    cout<<vLayerOutput.at(i)<<" ";

  cout<<endl;
}
