#include "SKModel.h"

void get_comb(int w ,int first,int second,vector<vector<int>>& arr,vector<vector<vector<int>>> &mWeightsPaths)
{
    int n = arr.size();

    vector<int> path;
    vector<vector<int>> pathMatrix;

    vector<int> indices(n,0);

    while (1) {

      path.push_back(first);
      path.push_back(second);

      for (int i = 0; i < n; i++)
        path.push_back(arr[i][indices[i]]);


      pathMatrix.push_back(path);

      path.clear();

      int next = n - 1;

      while (next >= 0 && (indices[next] + 1 >= arr[next].size()))
        next--;


        if (next < 0){

            indices.clear();
            mWeightsPaths.push_back(pathMatrix);
            return;
       }

        indices[next]++;

        for (int i = next + 1; i < n; i++)
            indices[i] = 0;

    }
  }


/* ----- Standard Constructor ----- */
SKModel::SKModel() :
 nLearningRate(0.001),
 nIterations(0),
 sLossFuction("Quadratic"){}

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

/* ----- Public Method Add Gradients ----- */
void SKModel::AddGradients(SKWeights *gradients){

  vModelGradients.push_back(gradients);

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

  nDataSize = mInputSample->size();
  nDataNRows = nDataSize;
  nDataNColumns = mInputSample->at(0).size();

  for(int i = 0 ; i < vModelWeights.size() ; i++)
   nTotalWeights = nTotalWeights + (vModelWeights.at(i)->fRows)*(vModelWeights.at(i)->fColumns);

  CheckDimensions();

  LOG(INFO)<<"Initializing Model .......";
  LOG(INFO)<<"Feed forward model with "<<nLayers<<" layers";
  LOG(INFO)<<"Number of trainable parameters : "<<nTotalWeights;
  LOG(INFO)<<"Data Size : "<<nDataSize<<" Input Samples";
  LOG(INFO)<<"Number of Features : "<<nDataNColumns;

  propagator = new SKPropagator();

  int maxSize=0;

  for (int i = 0 ; i < nLayers ; i++)
   if(vModelLayers.at(i)->fSize > maxSize)
    maxSize = vModelLayers.at(i)->fSize;

  modelHistogram = new TH2F("modelHistogram","Model",1000,0,vModelLayers.size() + 1,1000,0,maxSize+1);


  modelHistogram->GetXaxis()->SetTitle("Layer");
  modelHistogram->GetYaxis()->SetTitle("Neuron");
  modelHistogram->GetXaxis()->SetNdivisions(nLayers+1);
  modelHistogram->GetYaxis()->SetNdivisions(maxSize+1);

  LOG(INFO)<<"Calculating weight paths .......";

  vector <vector<int>> comb_vec;

  for (int w = vModelWeights.size()-1 ; w >= 0 ; w--) {
   for (int i = 0 ; i < vModelWeights.at(w)->fRows ; i++) {
     for (int j = 0 ; j < vModelWeights.at(w)->fColumns ; j++) {

         int firstLayer = w + 2;
         int lastLayer  = nLayers - 1;

         vector <int> layer_coord;

         /* ---- Calculate all possible paths for a given weight ----- */
         for(int n = firstLayer ; n <= lastLayer ; n++){
           for(int s = 0 ; s < vModelLayers.at(n)->fSize ; s++){

              layer_coord.push_back(s);

           }

           comb_vec.push_back(layer_coord);
           layer_coord.clear();

         }

           get_comb(w,i,j,comb_vec,mWeightsPaths);

           comb_vec.clear();

       }
      }
     }

}




void SKModel::Clear(){

   for (int i = 0 ; i < nLayers ; i++)
     vModelLayers.at(i)->Clear();

     vLossVector.clear();
 }


void SKModel::CheckDimensions(){


   bool isRight=1;

   for (int i = 0 ; i < (nLayers-1) ; i++ ) {

     isRight = (vModelLayers.at(i)->fSize == vModelWeights.at(i)->fRows);

     if(!isRight)
      LOG(FATAL)<<"Incompatible row dimensions :  Layer "<<i+1;

  }

   for (int i = (nLayers-1) ; i > 0 ; i-- ) {

    isRight = (vModelLayers.at(i)->fSize == vModelWeights.at(i-1)->fColumns);

    if(!isRight)
     LOG(FATAL)<<"Incompatible column dimensions :  Layer "<<i;
 }

}




vector<double> SKModel::Propagate(int n){


  vModelOutput.clear();

  vInput = &mInputSample->at(n);
  vLabel = &mInputLabels->at(n);


  propagator->Feed(vInput,vModelLayers.at(0));

  for(int i = 1 ; i < nLayers ; i++)
    propagator->Propagate(vModelLayers.at(i-1),vModelLayers.at(i),vModelWeights.at(i-1));


  for(int i = 0 ; i < vModelLayers.at(nLayers-1)->vLayerOutput.size() ; i++)
    vModelOutput.push_back(vModelLayers.at(nLayers-1)->vLayerOutput.at(i));



    return vModelOutput;

}




void SKModel::Train(int n){

  vInput = &mInputSample->at(n);
  vLabel = &mInputLabels->at(n);


  propagator->Feed(vInput,vModelLayers.at(0));

  for(int i = 1 ; i < nLayers ; i++)
    propagator->Propagate(vModelLayers.at(i-1),vModelLayers.at(i),vModelWeights.at(i-1));



    nIterations++;
    Backpropagate();

}



void SKModel::Backpropagate(){

  // First we calculate the gradient contribution for each path
  double pathGradient;

  // Then we add all path gradients to compute the change for a given weight
  double gradientSum;

  vector<double> lossDerivatives;
  vector <vector<int>> comb_vec;

  vector <vector<int>> path_matrix;

  int counter=0;
  int batchCounter;

  if(sLossFuction=="Quadratic"){

   for (int i = 0 ; i < vModelLayers.at(nLayers-1)->fSize ; i++)
     lossDerivatives.push_back((1.0/vModelLayers.at(nLayers-1)->fSize)*(vModelLayers.at(nLayers-1)->vLayerOutput.at(i)-vLabel->at(i)));

  }

  else if(sLossFuction=="Absolute"){

   for (int i = 0 ; i < vModelLayers.at(nLayers-1)->fSize ; i++)

     if(vModelLayers.at(nLayers-1)->vLayerOutput.at(i)-vLabel->at(i)>=0.0)
      lossDerivatives.push_back((1.0/vModelLayers.at(nLayers-1)->fSize)*(1.0));

     else
      lossDerivatives.push_back((1.0/vModelLayers.at(nLayers-1)->fSize)*(-1.0));

  }


  else{

     LOG(FATAL)<<"Loss function "<<sLossFuction<<" does not exist (in SoKAI)!";

  }


  // Init gradients to 0
  if(nIterations%(nBatchSize) == 1){
  for (int i = 0 ; i < vModelGradients.size() ; i++)
    vModelGradients.at(i)->ZeroGradients();

   batchCounter=1;
 }

   for (int w = vModelWeights.size()-1 ; w >= 0 ; w--) {
    for (int i = 0 ; i < vModelWeights.at(w)->fRows ; i++) {
      for (int j = 0 ; j < vModelWeights.at(w)->fColumns ; j++) {

        gradientSum = 0.0;
        path_matrix = mWeightsPaths[counter];


         /*-------- Now compute all gradients for all paths ---------*/
         for(int path = 0 ; path < path_matrix.size() ; path++){



            pathGradient=1;
            pathGradient = pathGradient*lossDerivatives.at(path_matrix[path][path_matrix[path].size()-1])*vModelLayers.at(w)->vLayerOutput.at(i);


          for(int r = 1 ; r < path_matrix[path].size() ; r++) {

             pathGradient=pathGradient*vModelLayers.at(r + w)->LayerDer(path_matrix[path][r]);

         }


           for(int r = 1 ; r < path_matrix[path].size()-1 ; r++){

             pathGradient=pathGradient*vModelWeights.at(w + r)->mWeightMatrix[path_matrix[path][r]][path_matrix[path][r+1]];

         }


             gradientSum = gradientSum + pathGradient;

        }

               vModelGradients.at(w)->mWeightMatrix[i][j] = vModelGradients.at(w)->mWeightMatrix[i][j] + (1.0/batchCounter)*(gradientSum - vModelGradients.at(w)->mWeightMatrix[i][j]);
               counter++;


     }
    }
   }

   batchCounter++;

   if(nIterations%nBatchSize==0) {

   for (int w = vModelWeights.size()-1 ; w >= 0 ; w--) {
    for (int i = 0 ; i < vModelWeights.at(w)->fRows ; i++) {
      for (int j = 0 ; j < vModelWeights.at(w)->fColumns ; j++) {

          vModelWeights.at(w)->mWeightMatrix[i][j] = vModelWeights.at(w)->mWeightMatrix[i][j]
          - nLearningRate*vModelGradients.at(w)->mWeightMatrix[i][j];
        }
     }
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


double SKModel::QuadraticLoss() {

    vLossVector.clear();


     for(int i = 0 ; i < vModelLayers.at(nLayers-1)->vLayerOutput.size() ; i++)
       vLossVector.push_back((1.0/2.0)*pow((vModelLayers.at(nLayers-1)->vLayerOutput.at(i)-vLabel->at(i)),2));


     double loss = (std::accumulate(vLossVector.begin(), vLossVector.end(), 0.0))/vLossVector.size();

     return loss;



}

double SKModel::AbsoluteLoss() {

    vLossVector.clear();


     for(int i = 0 ; i < vModelLayers.at(nLayers-1)->vLayerOutput.size() ; i++)
       vLossVector.push_back(TMath::Abs((vModelLayers.at(nLayers-1)->vLayerOutput.at(i)-vLabel->at(i))));


     double loss = (std::accumulate(vLossVector.begin(), vLossVector.end(), 0.0))/vLossVector.size();

     return loss;

}



TH2F * SKModel::ShowMe(){

   double x_start,x_end,y_start,y_end,m,n_const;
   double x,y;
   double true_smear;

   TRandom3 gen(0);

   for ( int n = 0 ; n < nLayers-1 ; n++) {
    for( int i = 0 ; i < vModelLayers.at(n)->fSize ; i++) {
     for( int j = 0 ; j < vModelLayers.at(n+1)->fSize ; j++) {

        x_start = n + 1 - 0.01;
        x_end   = n + 2 + 0.01;

        y_start = float(i) + 1.0;
        y_end   = float(j) + 1.0;

        m = (y_end - y_start)/(x_end - x_start);
        n_const = (y_start - m*x_start);

        for(int z = 0 ; z < abs(vModelWeights.at(n)->mWeightMatrix[i][j])*10000 ; z++) {

           x = (x_end-x_start)* gen.Rndm() + x_start;

           y = m*x + n_const;

           true_smear = 0.02/cos(TMath::Pi() - atan(m));

           y = gen.Gaus(y,true_smear);

           modelHistogram->Fill(x,y);

       }
     }
   }
 }

 return modelHistogram;

}
