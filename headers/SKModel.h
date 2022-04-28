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

 /* ----- Public Method Add Gradients -----*/
 void AddGradients(SKWeights *gradients);


 /* ----- Public Set Input Sample ----- */
 void SetInputSample(vector<vector<double>> *input);

 /* ----- Public Set Input Label ----- */
 void SetInputLabels(vector<vector<double>> *labels);


 /* ----- Public Method Train ----- */
 void Train(int n);

 /* ----- Public Method Set Batch Size ----- */
 void SetBatchSize(int bSize){ nBatchSize = bSize;};

 /* ----- Public Method Clear ----- */
 void Clear();

 /* ----- Public Method Quadratic Loss ----- */
 double QuadraticLoss();

 /* ----- Public Method Abs Loss ----- */
 double AbsoluteLoss();


 /* ----- Public Method Backpropagate -----*/
 void Backpropagate();

 /* ----- Public Method Propagate -----*/
 vector<double> Propagate(int n);


 float Accuracy();

 void SetLearningRate(float learningRate){nLearningRate=learningRate;};

 void SetLossFunction(string lossFunc){sLossFuction=lossFunc;};


 void CheckDimensions();

 TH2F * ShowMe();

private:

 vector<SKLayer*> vModelLayers;
 vector<SKWeights*> vModelWeights;
 vector<SKWeights*> vModelGradients;
 vector<vector<vector<int>>> mWeightsPaths;

 vector<vector<double>> *mInputLabels;
 vector<vector<double>> *mInputSample;

 /* ----- Aux Vectors -----*/
 vector<double> *vInput;
 vector<double> *vLabel;


 vector<double> vLossVector;
 SKPropagator *propagator;

 vector<double> vModelOutput;

 int nDataSize;
 int nDataNRows;
 int nDataNColumns;
 int nTotalWeights;
 int nLayers;
 int nIterations;
 int nBatchSize;
 float nAccuracy;
 float nLearningRate;
 TCanvas *modelCanvas;
 TH2F *modelHistogram;
 string sLossFuction;


};

 #endif
