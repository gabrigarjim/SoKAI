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

/* ----- Public Method Output ----- */
 float Output();

 /* ----- Public Method Input ----- */
 void Input(double input) {

   fInput = input ;

 }


private:

  string sActivationFunction;
  double fInput;

 };

 #endif
