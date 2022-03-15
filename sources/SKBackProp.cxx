#include "SKBackProp.h"

/* ----- Standard Constructor ----- */
SKBackProp::SKBackProp() {

}

/* ----- Standard Destructor ----- */
SKBackProp::~SKBackProp(){}


/* ----- Public Method Get Loss ----- */
float SKBackProp::Loss(SKLayer *outputLayer, vector<float> *targetVector){

   float loss=0.0;

   if(sLoss=="Quadratic"){

     for(int n = 0 ; n < targetVector->size() ; n++)
         loss = loss + 0.5*pow((targetVector->at(n)-outputLayer->vLayerOutput.at(n)),2);

   }


   if(sLoss=="Abs"){

     for(int n = 0 ; n < targetVector->size() ; n++)
         loss = loss + abs((targetVector->at(n)-outputLayer->vLayerOutput.at(n)));

   }


   return loss/targetVector->size();

}

void SKBackProp::CalculateGradients(SKWeights *weights, SKLayer *prevLayer , SKLayer *nextLayer,vector<float> *targetVector){

    // float weight_gradient_11,weight_gradient_21;
    //
    // weight_gradient_11 = (nextLayer->vLayerOutput.at(0) - targetVector->at(0))
    // * SigmoidDer(nextLayer->vNeurons.at(0).fInput) * prevLayer->vLayerOutput.at(0);
    //
    // weight_gradient_21 = (nextLayer->vLayerOutput.at(0) - targetVector->at(0))
    // * SigmoidDer(nextLayer->vNeurons.at(0).fInput) * prevLayer->vLayerOutput.at(1);
    //
    // cout<<" Old Value for Weight 11 : "<<weights->mWeightMatrix[0][0]<<" W_11 Gradient : "
    // <<weight_gradient_11<<" New Value for Weight 11 :  "<<weights->mWeightMatrix[0][0] - 0.01*weight_gradient_11<<endl;
    //
    // cout<<" Old Value for Weight 21 : "<<weights->mWeightMatrix[1][0]<<" W_21 Gradient : "
    // <<weight_gradient_21<<" New Value for Weight 21 :  "<<weights->mWeightMatrix[1][0] - 0.01*weight_gradient_21<<endl;
    //
    // weights->mWeightMatrix[0][0] = weights->mWeightMatrix[0][0] - 0.1*weight_gradient_11;
    // weights->mWeightMatrix[1][0] = weights->mWeightMatrix[1][0] - 0.1*weight_gradient_21;


}








/* ---- Some Derivatives ---- */
float SKBackProp::SigmoidDer(float arg) {

    return (1.0/(1.0 + exp(-1.0*arg)))*(1.0-1.0/(1.0 + exp(-1.0*arg)));

}


float SKBackProp::TanhDer(float arg) {

    return 1.0 - tanh(arg)*tanh(arg);

}


float SKBackProp::LinearDer(float arg) {

    return 1 ;
}


float SKBackProp::ReLUDer(float arg) {

  if(arg>0.0)
   return 1 ;

  else
   return 0.0;

}


float SKBackProp::LeakyReLUDer(float arg) {

  if(arg>0.0)
   return 1 ;

  else
   return 0.01;

}
