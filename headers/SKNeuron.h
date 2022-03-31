#ifndef SKNEURON_H
#define SKNEURON_H

#include "SKLibraries.h"

using namespace std;

class SKNeuron {

public:

  /* ----- Standard Constructor ----- */
  SKNeuron(string activation);

 /* ----- Standard Destructor ----- */
 ~SKNeuron();

 /* ----- Public Method Input ----- */
  void Input(float input){

    fInput = fInput + input;

  };


/* ----- Public Method Output ----- */
 float Output();


 /* ----- Public Method Clear ----- */
 void Clear(){fInput=0.0;};




private:

  string sActivationFunction;
  float fInput;

friend class SKBackProp;
friend class SKModel;
friend class SKLayer;



};

 #endif
