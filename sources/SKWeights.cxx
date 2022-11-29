#include "SKWeights.h"


/* ----- Standard Constructor ----- */
SKWeights::SKWeights(int rows, int columns) {

 fRows = rows;
 fColumns = columns;

}

/* ----- Standard Destructor ----- */
SKWeights::~SKWeights(){}

/* ----- Public Method Init ----- */
void SKWeights::Init(int seed) {

  TRandom3 gen(seed);
  vector<double> row;


  for(int i = 0 ; i < fRows ; i++) {
   for(int j = 0 ; j < fColumns ; j++){

      row.push_back(gen.Gaus(0,1));
   }

   mWeightMatrix.push_back(row);
   row.clear();

   }
 }



void SKWeights::Print(){

 for(int i = 0 ; i < fRows ; i++){
   for(int j = 0 ; j < fColumns ; j++){
     cout<<mWeightMatrix[i][j]<<" ";

    }
     cout<<"\n";
  }
}


void SKWeights::InitGradients() {

  vector<double> row;

  for(int i = 0 ; i < fRows ; i++) {
   for(int j = 0 ; j < fColumns ; j++){

      row.push_back(0.0);
   }

   mWeightMatrix.push_back(row);
   row.clear();

  }
}

void SKWeights::InitMoment() {

  vector<double> row;

  for(int i = 0 ; i < fRows ; i++) {
   for(int j = 0 ; j < fColumns ; j++){

      row.push_back(0.0);
   }

   mWeightMatrix.push_back(row);
   row.clear();

  }
}



void SKWeights::ZeroGradients() {


  for(int i = 0 ; i < fRows ; i++) {
   for(int j = 0 ; j < fColumns ; j++){

      mWeightMatrix[i][j]=0.0;

   }
  }
 }
