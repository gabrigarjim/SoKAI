#include "SKPropagator.h"

/* ----- Standard Constructor ----- */
SKPropagator::SKPropagator() {}

/* ----- Standard Destructor ----- */
SKPropagator::~SKPropagator(){}


void SKPropagator::Propagate(SKLayer *previousLayer, SKLayer *nextLayer, SKWeights *weights){

  float out;

  // GG - Every prev-m neuron must have n-next outputs
  for( int n = 0 ; n < nextLayer->fSize ; n++)
   for( int m = 0 ; m < previousLayer->fSize ; m++){

      out = weights->mWeightMatrix[m][n] * previousLayer->vLayerOutput.at(m);
      nextLayer->vNeurons.at(n).Input(out);

   }


    nextLayer->WriteOutput();

}


 void SKPropagator::Feed(vector<float> *inputVector , SKLayer *firstLayer, SKWeights *weights) {


   for( int n = 0 ; n < firstLayer->fSize ; n++){

         firstLayer->vNeurons.at(n).Input(inputVector->at(n)*weights->mWeightMatrix[n][0]);


   }

   firstLayer->WriteOutput();


 }
