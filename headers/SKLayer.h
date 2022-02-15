#ifndef SKLAYER_H
#define SKLAYER_H

#include "SKLibraries.h"
#include "SKNeuron.h"

using namespace std;

class SKLayer {

 public:

  /* ----- Standard Constructor ----- */

  SKLayer(int size, string activation);


  /* ----- Standard Destructor ----- */
  ~SKLayer();


 private:

  int fSize;
  vector<SKNeuron> vNeurons;


};

#endif
