#include "SKNeuron.h"

/* ----- Standard Constructor ----- */
SKNeuron::SKNeuron(string activation) :
  fInput(0.0) {

 sActivationFunction = activation;

}

/* ----- Standard Destructor ----- */
SKNeuron::~SKNeuron(){}


/* ----- Public Method Output ----- */
float SKNeuron::Output() {

 float out;

    if (sActivationFunction == "Sigmoid" ) {

      out = 1.0/(1.0 + exp(-1.0*fInput));

    }

    else if (sActivationFunction == "Tanh"){

      out=tanh(fInput);

    }

    else if (sActivationFunction == "Linear") {

      out=fInput;

    }

    else if (sActivationFunction == "ReLU") {

      (fInput >= 0.0) ? out = fInput : out = 0.0;

    }

    else if (sActivationFunction == "LeakyReLU") {

      (fInput >= 0.0) ? out = fInput : out = 0.01*fInput;

    }


 return out;

}
