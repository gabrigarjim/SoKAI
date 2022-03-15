#ifndef SKMODEL_H
#define SKMODEL_H

#include "SKLibraries.h"
#include "SKLayer.h"
#include "SKWeights.h"

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


private:

 vector<SKLayer*> vModelLayers;
 vector<SKWeights*> vModelWeights;

};

 #endif
