#ifndef SKWEIGHTS_H
#define SKWEIGHTS_H

#include "SKLibraries.h"

using namespace std;

class SKWeights {

public:

  /* ----- Standard Constructor ----- */
  SKWeights(int rows, int columns);

 /* ----- Standard Destructor ----- */
 ~SKWeights();

 /* ----- Public Method Init ----- */
 void Init(int seed);

 /* ----- Public Method Init Gradients ----- */
 void InitGradients();

 /* ----- Public Method Zero Gradients ----- */
 void ZeroGradients();


 /* ----- Public Method Print Weights -----*/
 void Print();

 private:

  int fRows;
  int fColumns;
  vector<vector<double>> mWeightMatrix;


 friend class SKPropagator;
 friend class SKBackProp;
 friend class SKModel;

 };

 #endif
