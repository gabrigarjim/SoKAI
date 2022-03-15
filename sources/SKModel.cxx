#include "SKModel.h"

/* ----- Standard Constructor ----- */
SKModel::SKModel() {

}

/* ----- Standard Destructor ----- */
SKModel::~SKModel(){}


/* ----- Public Method Add Layer ----- */
void SKModel::AddLayer(SKLayer *layer){

  vModelLayers.push_back(layer);

}

/* ----- Public Method Add Weights ----- */
void SKModel::AddWeights(SKWeights *weights){

  vModelWeights.push_back(weights);

}



/* ----- Public Method Init ----- */
void SKModel::Init(){


  int nLayers = vModelLayers.size();
  int nTotalWeights =0;

  for(int i = 0 ; i < vModelWeights.size() ; i++)
   nTotalWeights = nTotalWeights + (vModelWeights.at(i)->fRows)*(vModelWeights.at(i)->fColumns);

  LOG(INFO)<<"Initializing Model ----------";
  LOG(INFO)<<"Feed forward model with "<<nLayers<<" layers";
  LOG(INFO)<<"Number of trainable parameters : "<<nTotalWeights;




}
