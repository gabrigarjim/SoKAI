# SoKAI
Some Kind of Artificial Intelligence

Useful reminders :

 - In class attributes ,
    - numerical values begin with   "n" (nLayers) or "f" (fSize)
    - vectors begin with            "v" (vNeurons)
    - strings begin with            "s" (sActivationFunction)
    - matrices begin with           "m" (mWeightMatrix)
    

Code structure :

  - The framework is based on objects :

    - Data structure objects : SKNeuron, SKLayer

     - SKNeuron  : Processes data as f(Input), with f = Sigmoid, Linear reLU.....
     - SKLayer   : Vector of SKNeurons. Stores inputs and outputs to perform
       backpropagation.
     - SKWeights : Weight matrices. 
   
    - Procedure objects : SKModel, SKPropagator

      - SKModel : Contains the structure + the methods to perform the training
      - SKPropagator : Feeds and propagates data between layers  



Dependencies:
   - ROOT. Any installation is valid, tested with standalone build and also with FairSoft build.
   - GLOG. Google Logging Library. For MAC users : brew install glog. Linux : apt install libgoogle-glog-dev
   - CMAKE . Tested with version 3.16 and 3.13. Any recent version should work.
 

TO DO LIST :
 - Implement ADAM
 - Start working on Convolutional and Pooling Layers














