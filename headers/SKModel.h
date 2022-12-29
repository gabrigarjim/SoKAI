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
  SKModel(string model);

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

 /* ----- Public Method Add Moment -----*/
 void AddFirstMoments(SKWeights *firstMoments);

 /* ----- Public Method Add Moment -----*/
 void AddSecondMoments(SKWeights *secondMoments);

 /* ----- Public Set Input Sample ----- */
 void SetInputSample(vector<vector<double>> *input);

 /* ----- Public Set Input Label ----- */
 void SetInputLabels(vector<vector<double>> *labels);


 /* ----- Public Method Train ----- */
 void Train(int n);

 /* ----- Public Method Set Batch Size ----- */
 void SetBatchSize(int bSize){ nBatchSize = bSize;};

 /* ----- Public Method Write Weights */
 void SaveWeights(string file);


 /* ----- Public Method Clear ----- */
 void Clear();

 /* ----- Public Method Quadratic Loss ----- */
 double QuadraticLoss();

 /* ----- Public Method Abs Loss ----- */
 double AbsoluteLoss();

 /* ----- Public Method Cross Entropy Loss ----- */
 double CrossEntropyLoss();

 /* ----- Public Method Backpropagate -----*/
 void Backpropagate();

 /* ----- Public Method Propagate -----*/
 vector<double> Propagate(int n);

 /* ----- Public Method Load Weights -----*/
 void LoadWeights(string file);

 void SetOptimizer(string opt){ sOptimizer = opt;};

 void SetSummaryFile(string file, string modelNumber){

   sSummaryFile = file;
   sModelNumber = modelNumber;
 };



  float Accuracy();

  void SetLearningRate(float learningRate){nLearningRate=learningRate;};

  void SetLossFunction(string lossFunc){sLossFunction=lossFunc;};

  void CheckDimensions();

  TH2F * ShowMe();

private:

 vector<SKLayer*> *vModelLayers = new vector<SKLayer*>();
 vector<SKWeights*> vModelWeights;
 vector<SKWeights*> vModelGradients;
 vector<vector<vector<int>>> mWeightsPaths;

 vector<SKWeights*> vModelFirstMoment;
 vector<SKWeights*> vModelSecondMoment;

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
 int nTotalIterations;
 int nBatchSize;
 float nAccuracy;
 float nLearningRate;
 TCanvas *modelCanvas;
 TH2F *modelHistogram;
 string sLossFunction;
 string sModelType;
 string sOptimizer;

 float nBeta1;
 float nBeta2;
 float nEpsilon;

 double nFirstHatMoment;
 double nSecondHatMoment;
 string sSummaryFile;
 string sModelNumber;

 ofstream *sum_file=NULL;

};

 #endif
