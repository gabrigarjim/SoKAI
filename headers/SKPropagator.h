#ifndef SKPROPAGATOR_H
#define SKPROPAGATOR_H

#include "SKLibraries.h"
#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"

using namespace std;

class SKPropagator {

 public:

  /* ----- Standard Constructor ----- */

  SKPropagator();

  /* ----- Public Method Propagate ----- */
  void Propagate(SKLayer *previousLayer, SKLayer *nextLayer, SKWeights *weights);

  /* ----- Public Method Feed ----- */
  void Feed(vector<float> *inputVector , SKLayer *firstLayer);


  /* ----- Standard Destructor ----- */
  ~SKPropagator();


 private:


};

#endif
