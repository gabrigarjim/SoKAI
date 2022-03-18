# SoKAI
Some Kind of Artificial Intelligence

Useful reminders :

 - In class attributes ,
    - values begin with   "n" (fSize)
    - vectors begin with  "v" (vNeurons)
    - strings begin with  "s" (sActivationFunction)
    - matrices begin with "m" (mWeightMatrix)


Code structure :

  - The framework is based on objects :

    - Data structure objects : SKNeuron, SKLayer

     - SKNeuron : Processes data as f(Input), with f = Sigmoid, Linear reLU.....
     - SKLayer  : Vector of SKNeurons. Stores inputs and outputs to perform
       backpropagation.

    - Procedure objects : SKModel, SKPropagator

      - SKModel : Contains the structure + the methods to perform the training
      - SKPropagator : Feeds and propagates data between layers  
