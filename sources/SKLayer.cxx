#include "SKLayer.h"


double EvaluateSoftmax(vector<double> &vec , int index){

 double denominator = 0.0;

 for(int i = 0 ; i < vec.size() ; i++)
  denominator += exp(vec.at(i));

 return exp(vec.at(index))/denominator;

}


/* ----- Standard Constructor ----- */
SKLayer::SKLayer(int size, string activation) {

  sActivationFunction = activation;
  fSize = size;

  SKNeuron neuron(activation);

  for ( int i = 0; i < fSize ; i++)
    vNeurons.push_back(neuron);


}

/* ----- Standard Destructor ----- */
SKLayer::~SKLayer(){}


/* ----- Public Method WriteOutput ----- */
void SKLayer::WriteOutput(){

  for ( int i = 0 ; i < fSize ; i++ )
    vLayerOutput.push_back(vNeurons.at(i).Output());

}


/* ----- Public Method Print ----- */
void SKLayer::Print(){

  for ( int i = 0 ; i < vLayerOutput.size() ; i++ )
    cout<<vLayerOutput.at(i)<<" ";

   cout<<endl;
}


/* ----- Public Method Clear ----- */
void SKLayer::Clear(){

  for ( int n = 0 ; n < vNeurons.size() ; n++ )
    vNeurons.at(n).Clear();

  vLayerOutput.clear();

}


double SKLayer::TanhDer(double arg){

  return (1-tanh(arg)*tanh(arg));

}

double SKLayer::LinearDer(double arg){

  return 1.0;

}

double SKLayer::ReLUDer(double arg){

 return (arg > 0.0) ?  arg : 0.0;

}

double SKLayer::LeakyReLUDer(double arg){

 return (arg > 0.0) ?  arg : 0.01;

}



void SKLayer::RearrangeSoftmax() {

  vector<double> vAuxVector;

  for ( int i = 0; i < fSize ; i++)
    vAuxVector.push_back(EvaluateSoftmax(vLayerOutput,i));

  for ( int i = 0; i < fSize ; i++)
    vLayerOutput.at(i) = vAuxVector.at(i);


}



double SKLayer::LayerDer(int neuron) {

  if(sActivationFunction == "Sigmoid")
   return SigmoidDer(vNeurons.at(neuron).fInput);


  if(sActivationFunction == "Tanh")
   return TanhDer(vNeurons.at(neuron).fInput);


  if(sActivationFunction == "Linear")
   return LinearDer(vNeurons.at(neuron).fInput);


  if(sActivationFunction == "ReLU")
   return ReLUDer(vNeurons.at(neuron).fInput);


  if(sActivationFunction == "LeakyReLU")
   return LeakyReLUDer(vNeurons.at(neuron).fInput);

  else
   return 0.0;
}
