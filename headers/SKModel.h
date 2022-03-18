#ifndef SKMODEL_H
#define SKMODEL_H

#include "SKLibraries.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"

using namespace std;

class SKModel {

public:

  /* ----- Standard Constructor ----- */
  SKModel();

 /* ----- Standard Destructor ----- */
 ~SKModel();

 /* ----- Public Method Add Layer ----- */
  void AddLayer(SKLayer *layer);

 /* ----- Public Method Init ----- */
  void Init();

 /* ----- Public Method Add Weights -----*/
 void AddWeights(SKWeights *weights);


 /* ----- Public Set Input Sample ----- */
 void SetInputSample(vector<vector<float>> *input);

 /* ----- Public Set Input Label ----- */
 void SetInputLabels(vector<float> *labels);


 /* ----- Public Method Propagate ----- */
 void Propagate();


 /* ----- Public Method Clear ----- */
 void Clear();

private:

 vector<SKLayer*> vModelLayers;
 vector<SKWeights*> vModelWeights;

 vector<float> *vInputLabels;
 vector<vector<float>> *mInputSample;
 vector<float> *vInput;

 SKPropagator * propagator;

 int nDataSize;
 int nDataNRows;
 int nDataNColumns;
 int nTotalWeights;
 int nLayers;


};

 #endif
