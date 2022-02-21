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


  // GG - As recommended in many sources, weights should be initialize
  // with values around a gaussian distribution with mean = 0 and
  // sigma = 1

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
