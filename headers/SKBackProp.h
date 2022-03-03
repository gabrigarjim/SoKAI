#ifndef SKBACKPROP_H
#define SKBACKPROP_H

#include "SKLibraries.h"
#include "SKWeights.h"
#include "SKLayer.h"

using namespace std;

class SKBackProp {

 public:

  /* ----- Standard Constructor ----- */
  SKBackProp();


  /* ----- Standard Destructor ----- */
  ~SKBackProp();

  /* ----- Method Set Loss ----- */
  void SetLoss(string loss){ sLoss = loss;};

  /* ----- Method Get Loss ----- */
  float Loss(SKLayer *outputLayer, vector<float> *targetVector);

  /* ----- Method Calculate Gradients -----*/
  void CalculateGradients(SKWeights *weights,SKLayer *prevLayer,SKLayer *nextLayer, vector<float> *targetVector);

 private:
 string sLoss;
 /* ---- GG - Derivatives as aux functions inside the class ---- */
 float SigmoidDer(float arg);
 float TanhDer(float arg);
 float LinearDer(float arg);
 float ReLUDer(float arg);
 float LeakyReLUDer(float arg);


};

#endif
