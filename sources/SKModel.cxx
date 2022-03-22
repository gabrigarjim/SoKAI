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
void SKModel::SetInputSample(vector<vector<double>> *input){

   mInputSample = input;

}

/* ----- Public Method Set Input Label ----- */
void SKModel::SetInputLabels(vector<vector<double>> *labels){

   mInputLabels = labels;

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

  LOG(INFO)<<"Initializing Model .......";
  LOG(INFO)<<"Feed forward model with "<<nLayers<<" layers";
  LOG(INFO)<<"Number of trainable parameters : "<<nTotalWeights;
  LOG(INFO)<<"Data Size : "<<nDataSize<<" Input Samples";
  LOG(INFO)<<"Number of Features : "<<nDataNColumns;

  propagator = new SKPropagator();


}





 void SKModel::Propagate(int n){

   vInput = &mInputSample->at(n);
   vLabel = &mInputLabels->at(n);


   propagator->Feed(vInput,vModelLayers.at(0));

   for(int i = 1 ; i < nLayers ; i++)
     propagator->Propagate(vModelLayers.at(i-1),vModelLayers.at(i),vModelWeights.at(i-1));


     QuadraticLoss(&(vModelLayers.at(nLayers-1)->vLayerOutput),vLabel);

     //cout<<"Loss : "<<vLossVector.at(0)<<" "<<vLossVector.at(1)<<" "<<vLossVector.at(2)<<endl;



     nIterations++;



 }



void SKModel::Clear(){

   for (int i = 0 ; i < nLayers ; i++)
     vModelLayers.at(i)->Clear();

     vLossVector.clear();
 }


void SKModel::QuadraticLoss(vector<double> *outputVector, vector<double> *targetVector) {

     for(int i = 0 ; i < outputVector->size() ; i++){

       vLossVector.push_back((1.0/outputVector->size())*pow(outputVector->at(i) - targetVector->at(i),2));

     }

}


void SKModel::Backpropagate(){


 /* Now the mother of the lamb .....*/


 /* Second weight matrix */

 int nWeightsRows,nWeightsColumns;
 int nWeightsRowsFirst,nWeightsColumnsFirst;

 nWeightsRows = vModelWeights.at(1)->fRows;
 nWeightsColumns = vModelWeights.at(1)->fColumns;

 double mWeightsGradients[nWeightsRows][nWeightsColumns]={{0.0}};

 for (int i = 0 ; i < nWeightsRows ; i++){
  for(int j = 0 ; j < nWeightsColumns ; j++){

    mWeightsGradients[i][j] = (1.0/vModelLayers.at(nLayers-1)->fSize)*(vModelLayers.at(nLayers-1)->vLayerOutput.at(j)- vLabel->at(j))
                              *SigmoidDer(vModelLayers.at(nLayers-1)->vNeurons.at(j).fInput)*(vModelLayers.at(nLayers-2)->vLayerOutput.at(i));
  }
}



/* First Weight Matrix*/

nWeightsRowsFirst = vModelWeights.at(0)->fRows;
nWeightsColumnsFirst = vModelWeights.at(0)->fColumns;

double mFirstWeightsGradients[nWeightsRowsFirst][nWeightsColumnsFirst]={{0.0}};

double firstStepSum=0.0;

for (int i = 0 ; i < nWeightsRowsFirst ; i++){
 for(int j = 0 ; j < nWeightsColumnsFirst ; j++){

   for(int k = 0 ; k < vModelLayers.at(nLayers-1)->fSize ; k++){

    firstStepSum = firstStepSum + (1.0/vModelLayers.at(nLayers-1)->fSize)*
                   (vModelLayers.at(nLayers-1)->vLayerOutput.at(k)- vLabel->at(k))*
                   SigmoidDer(vModelLayers.at(nLayers-1)->vNeurons.at(k).fInput)*
                   vModelWeights.at(1)->mWeightMatrix[j][k];

    }

   mFirstWeightsGradients[i][j] = SigmoidDer(vModelLayers.at(nLayers-2)->vNeurons.at(j).fInput)*
                                  vModelLayers.at(0)->vLayerOutput.at(i)*firstStepSum;
  }


}




/* ------ Updating Weights ------*/
for (int i = 0 ; i < nWeightsRows ; i++){
 for(int j = 0 ; j < nWeightsColumns ; j++){

   vModelWeights.at(1)->mWeightMatrix[i][j] = vModelWeights.at(1)->mWeightMatrix[i][j] - nLearningRate*mWeightsGradients[i][j];

 }
}

for (int i = 0 ; i < nWeightsRowsFirst ; i++){
 for(int j = 0 ; j < nWeightsColumnsFirst ; j++){

   vModelWeights.at(0)->mWeightMatrix[i][j] = vModelWeights.at(0)->mWeightMatrix[i][j] - nLearningRate*mFirstWeightsGradients[i][j];

 }
}


}

float SKModel::Accuracy(){


float counter = 0.0;


for (int i = 0 ; i < mInputSample->size() ; i++){

  vInput = &mInputSample->at(i);
  vLabel = &mInputLabels->at(i);

  float maxLabel=0.0,maxOutput=0.0;

  maxLabel =  std::distance(vLabel->begin(),std::max_element(vLabel->begin(), vLabel->end()));

  propagator->Feed(vInput,vModelLayers.at(0));

  for(int i = 1 ; i < nLayers ; i++)
    propagator->Propagate(vModelLayers.at(i-1),vModelLayers.at(i),vModelWeights.at(i-1));


  vector<double> layerOut = vModelLayers.at(nLayers-1)->vLayerOutput;
  maxOutput = std::distance(layerOut.begin(),std::max_element(layerOut.begin(), layerOut.end()));



  if(maxOutput==maxLabel)
  counter++;

  Clear();

  }


  return 100*counter/mInputSample->size();


}













double SKModel::SigmoidDer(double arg) {

     return (1.0/(1.0 + exp(-1.0*arg)))*(1.0-1.0/(1.0 + exp(-1.0*arg)));

}
