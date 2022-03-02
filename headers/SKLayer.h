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

  /* ----- Public Method Write Out ----- */
  void WriteOutput();


  /* ----- Public Method Print ----- */
  void Print();

  /* ----- Public Method Clear ----- */
  void Clear();




 private:

  int fSize;
  vector<SKNeuron> vNeurons;
  vector<double> vLayerOutput;

  friend class SKPropagator;
  friend class SKBackProp;


};

#endif
