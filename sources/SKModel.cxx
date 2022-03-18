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

/* ----- Public Method Set Input ----- */
void SKModel::SetInputSample(vector<vector<float>> *input){

   mInputSample = input;

}

/* ----- Public Method Set Input Label ----- */
void SKModel::SetInputLabels(vector<float> *labels){

 vInputLabels = labels;

}


/* ----- Public Method Init ----- */
void SKModel::Init(){


  nLayers = vModelLayers.size();
  nTotalWeights = 0;

  nDataSize=mInputSample->size();
  nDataNRows=nDataSize;
  nDataNColumns = mInputSample->at(0).size();

  for(int i = 0 ; i < vModelWeights.size() ; i++)
   nTotalWeights = nTotalWeights + (vModelWeights.at(i)->fRows)*(vModelWeights.at(i)->fColumns);

  LOG(INFO)<<"Initializing Model ----------";
  LOG(INFO)<<"Feed forward model with "<<nLayers<<" layers";
  LOG(INFO)<<"Number of trainable parameters : "<<nTotalWeights;
  LOG(INFO)<<"Data Size : "<<nDataSize<<" Input Samples";
  LOG(INFO)<<"Number of Features : "<<nDataNColumns;

  propagator = new SKPropagator();


}





 void SKModel::Propagate(){


  for (int n = 0 ; n < nDataSize ; n++){

   vInput = &mInputSample->at(n);

   propagator->Feed(vInput,vModelLayers.at(0));

   for(int i = 1 ; i < nLayers ; i++)
     propagator->Propagate(vModelLayers.at(i-1),vModelLayers.at(i),vModelWeights.at(i-1));


     vModelLayers.at(nLayers-1)->Print();

     Clear();

   }



 }



void SKModel::Clear(){

   for (int i = 0 ; i < nLayers ; i++)
     vModelLayers.at(i)->Clear();


 }
