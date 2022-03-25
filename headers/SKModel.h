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
 void SetInputSample(vector<vector<double>> *input);

 /* ----- Public Set Input Label ----- */
 void SetInputLabels(vector<vector<double>> *labels);


 /* ----- Public Method Propagate ----- */
 void Propagate(int n);


 /* ----- Public Method Clear ----- */
 void Clear();

 /* ----- Public Method Quadratic Loss ----- */
 void QuadraticLoss(vector<double> *outputVector, vector<double> *targetVector);

 /* ----- Public Method Backpropagate -----*/
 void Backpropagate();

 float Accuracy();

 void SetLearningRate(float learningRate){nLearningRate=learningRate;};

 void CheckDimensions();

 TH2F * ShowMe();

private:

 vector<SKLayer*> vModelLayers;
 vector<SKWeights*> vModelWeights;

 vector<vector<double>> *mInputLabels;
 vector<vector<double>> *mInputSample;

 /* ----- Aux Vectors -----*/
 vector<double> *vInput;
 vector<double> *vLabel;


 vector<double> vLossVector;
 SKPropagator *propagator;

 int nDataSize;
 int nDataNRows;
 int nDataNColumns;
 int nTotalWeights;
 int nLayers;
 int nIterations;
 float nAccuracy;
 float nLearningRate;
 TCanvas *modelCanvas;
 TH2F *modelHistogram;

 double SigmoidDer(double arg);



};

 #endif
