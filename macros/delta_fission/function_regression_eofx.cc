#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

#include "eofx_generator_ia.C"

int main () {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("FunctionRegressionEOFX");

  TApplication* theApp = new TApplication("regression", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed            = 2022;
  int epochs          = 2000;
  int nSamples        = 120; //number of training cases
  int nTrainingSize   = (7.0/10.0)*nSamples;
  int nTestSize       = (3.0/10.0)*nSamples;
  int nMiniBatchSize  = 4;
  float fLearningRate = 0.001;

  SKColorScheme();

  real_start = clock();

  /*---- Input Data ---- */
  vector<vector<double>> data_sample;
  vector<vector<double>> input_labels;

  vector<double> data_instance;
  vector<double> label_instance;

  /*---- For training results ----*/
  vector<double> loss_vec;
  vector<double> output_vec;
  vector<double> output_model;

  vector<double> x_vec;
  vector<double> target_vec;
  vector<double> epoch_vec;

  double loss=0.0;

  TRandom3 gen(seed);


  /* -------- Put this on a header or something..... ----------*/
  gStyle->SetOptStat(0);
   /* --------------------------------------------------------*/

   vector<double> x;
   vector<double> y;

   for (int i = 0 ; i < nSamples ; i++){

     x.clear();
     y.clear();
     eofx_generator_ia(x,y);

      data_sample.push_back(x); // x is already a vector, filled by eofx

      input_labels.push_back(y); // Same here

      data_instance.clear();
      label_instance.clear();

   }

  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(30,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(30,4);
  SKWeights *gradients_12 = new SKWeights(30,4);
  SKWeights *firstMoment_12 = new SKWeights(30,4);
  SKWeights *secondMoment_12 = new SKWeights(30,4);


  SKLayer   *layer_2 = new SKLayer(4,"Sigmoid");
  SKWeights *weights_23 = new SKWeights(4,4);
  SKWeights *gradients_23 = new SKWeights(4,4);
  SKWeights *firstMoment_23 = new SKWeights(4,4);
  SKWeights *secondMoment_23 = new SKWeights(4,4);

  SKLayer   *layer_3 = new SKLayer(4,"Sigmoid");
  SKWeights *weights_34 = new SKWeights(4,30);
  SKWeights *gradients_34 = new SKWeights(4,30);
  SKWeights *firstMoment_34 = new SKWeights(4,30);
  SKWeights *secondMoment_34 = new SKWeights(4,30);


  SKLayer   *layer_4 = new SKLayer(30,"Linear");


  weights_12->Init(seed);
  gradients_12->InitGradients();
  firstMoment_12->InitMoment();
  secondMoment_12->InitMoment();


  weights_23->Init(seed);
  gradients_23->InitGradients();
  firstMoment_23->InitMoment();
  secondMoment_23->InitMoment();


  weights_34->Init(seed);
  gradients_34->InitGradients();
  firstMoment_34->InitMoment();
  secondMoment_34->InitMoment();



  SKModel *model = new SKModel("Regression");

  model->SetOptimizer("Adam");

  model->AddLayer(layer_1);
  model->AddWeights(weights_12);
  model->AddGradients(gradients_12);
  model->AddFirstMoments(firstMoment_12);
  model->AddSecondMoments(secondMoment_12);


  model->AddLayer(layer_2);
  model->AddWeights(weights_23);
  model->AddGradients(gradients_23);
  model->AddFirstMoments(firstMoment_23);
  model->AddSecondMoments(secondMoment_23);

  model->AddLayer(layer_3);
  model->AddWeights(weights_34);
  model->AddGradients(gradients_34);
  model->AddFirstMoments(firstMoment_34);
  model->AddSecondMoments(secondMoment_34);



  model->AddLayer(layer_4);


  model->SetInputSample(&data_sample);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction("Quadratic");

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);


  /* ---------- Pass Data Through Model ----------*/

  TCanvas *model_canvas = new TCanvas("model_canvas","Model");
  TGraph *loss_graph=new TGraph();
  model_canvas->cd();
  loss_graph->SetTitle("Model Loss");
  loss_graph->GetXaxis()->SetTitle("Epochs");
  loss_graph->GetYaxis()->SetTitle("Loss (Quadratic)");
  // loss_graph->Draw("al");

  // model_canvas->Modified();


   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < nTrainingSize ; j++){


      // Using  7/10 of the dataset to train the network
      int sample_number = nTrainingSize*gen.Rndm();

      model->Train(j);

      loss =  model->QuadraticLoss();

      model->Clear();

   }

    if(i%10==0){

     LOG(INFO)<<" Loss : "<<loss<<" . Epoch : "<<i;
     loss_vec.push_back(loss);
     epoch_vec.push_back(i);

    }

}

real_end = clock();

cout<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s"<<endl;

vector<vector<double>> delta_output;
vector<vector<double>> label_output;
vector<vector<double>> yields_input;

vector<double> charges_input;

for(int i = 30 ; i < 60 ; i++)
 charges_input.push_back(i);

/* --------- Testing the model --------- */

 for (int j = 0 ; j < 3 ; j++){


    // Using only 3/10 of the dataset to test the network
    int sample_number = nTrainingSize + nTestSize*gen.Rndm();


    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    delta_output.push_back(output_vec);
    label_output.push_back(input_labels.at(sample_number));
    yields_input.push_back(data_sample.at(sample_number));

    model->Clear();


  }


TCanvas *delta_canvas = new TCanvas("delta_canvas","Deltas");
 delta_canvas->Divide(2,3);

TGraph *deltaGraph_modelOut_1 = new TGraph(charges_input.size(),&(charges_input[0]),&(delta_output.at(0).at(0)));
TGraph *deltaGraph_labelOut_1 = new TGraph(charges_input.size(),&(charges_input[0]),&(label_output.at(0).at(0)));
TGraph *yieldGraph_1 = new TGraph(charges_input.size(),&(charges_input[0]),&(yields_input.at(0).at(0)));

TGraph *deltaGraph_modelOut_2 = new TGraph(charges_input.size(),&(charges_input[0]),&(delta_output.at(1).at(0)));
TGraph *deltaGraph_labelOut_2 = new TGraph(charges_input.size(),&(charges_input[0]),&(label_output.at(1).at(0)));
TGraph *yieldGraph_2 = new TGraph(charges_input.size(),&(charges_input[0]),&(yields_input.at(1).at(0)));


TGraph *deltaGraph_modelOut_3 = new TGraph(charges_input.size(),&(charges_input[0]),&(delta_output.at(2).at(0)));
TGraph *deltaGraph_labelOut_3 = new TGraph(charges_input.size(),&(charges_input[0]),&(label_output.at(2).at(0)));
TGraph *yieldGraph_3 = new TGraph(charges_input.size(),&(charges_input[0]),&(yields_input.at(2).at(0)));

delta_canvas->cd(1);
yieldGraph_1->Draw("AC*");
yieldGraph_1->SetTitle("YIELDS");
yieldGraph_1->GetXaxis()->SetTitle("Z");
yieldGraph_1->GetYaxis()->SetTitle("Yields");


delta_canvas->cd(2);
deltaGraph_labelOut_1->Draw("AC*");
deltaGraph_labelOut_1->SetTitle("Target");
deltaGraph_labelOut_1->GetXaxis()->SetTitle("Z");
deltaGraph_labelOut_1->GetYaxis()->SetTitle("Delta (target)");
deltaGraph_labelOut_1->SetMarkerColor(2);
deltaGraph_labelOut_1->SetMarkerStyle(24);
deltaGraph_labelOut_1->SetMarkerSize(1.0);


delta_canvas->cd(2);
deltaGraph_modelOut_1->Draw("SAME*");
deltaGraph_modelOut_1->SetTitle("Model Prediction");
deltaGraph_modelOut_1->GetXaxis()->SetTitle("Z");
deltaGraph_modelOut_1->GetYaxis()->SetTitle("Delta (predicted)");
deltaGraph_modelOut_1->SetMarkerColor(1);
deltaGraph_modelOut_1->SetMarkerStyle(24);
deltaGraph_modelOut_1->SetMarkerSize(1.0);


delta_canvas->cd(3);
yieldGraph_2->Draw("AC*");
yieldGraph_2->SetTitle("YIELDS");
yieldGraph_2->GetXaxis()->SetTitle("Z");
yieldGraph_2->GetYaxis()->SetTitle("Yields");

delta_canvas->cd(4);
deltaGraph_labelOut_2->Draw("AC*");
deltaGraph_labelOut_2->SetTitle("Target");
deltaGraph_labelOut_2->GetXaxis()->SetTitle("Z");
deltaGraph_labelOut_2->GetYaxis()->SetTitle("Delta (target)");
deltaGraph_labelOut_2->SetMarkerColor(2);
deltaGraph_labelOut_2->SetMarkerStyle(24);
deltaGraph_labelOut_2->SetMarkerSize(1.0);


delta_canvas->cd(4);
deltaGraph_modelOut_2->Draw("SAME*");
deltaGraph_modelOut_2->SetTitle("Model Prediction");
deltaGraph_modelOut_2->GetXaxis()->SetTitle("Z");
deltaGraph_modelOut_2->GetYaxis()->SetTitle("Delta (predicted)");
deltaGraph_modelOut_2->SetMarkerColor(1);
deltaGraph_modelOut_2->SetMarkerStyle(24);
deltaGraph_modelOut_2->SetMarkerSize(1.0);


delta_canvas->cd(5);
yieldGraph_3->Draw("AC*");
yieldGraph_3->SetTitle("YIELDS");
yieldGraph_3->GetXaxis()->SetTitle("Z");
yieldGraph_3->GetYaxis()->SetTitle("Yields");


delta_canvas->cd(6);
deltaGraph_labelOut_3->Draw("AC*");
deltaGraph_labelOut_3->SetTitle("Target");
deltaGraph_labelOut_3->GetXaxis()->SetTitle("Z");
deltaGraph_labelOut_3->GetYaxis()->SetTitle("Delta (target)");
deltaGraph_labelOut_3->SetMarkerColor(2);
deltaGraph_labelOut_3->SetMarkerStyle(24);
deltaGraph_labelOut_3->SetMarkerSize(1.0);


delta_canvas->cd(6);
deltaGraph_modelOut_3->Draw("SAME*");
deltaGraph_modelOut_3->SetTitle("Model Prediction");
deltaGraph_modelOut_3->GetXaxis()->SetTitle("Z");
deltaGraph_modelOut_3->GetYaxis()->SetTitle("Delta (predicted)");
deltaGraph_modelOut_3->SetMarkerColor(1);
deltaGraph_modelOut_3->SetMarkerStyle(24);
deltaGraph_modelOut_3->SetMarkerSize(1.0);



/* --------- Plots and so on ....... ------*/

TCanvas *weight_canvas = new TCanvas("weight_canvas","Weights");

/*TGraph *target_graph = new TGraph(x_vec.size(),&x_vec[0],&target_vec[0]);
TGraph *out_graph = new TGraph(x_vec.size(),&x_vec[0],&output_model[0]);

model_canvas->cd(1);
target_graph->Draw("AC*");
target_graph->SetTitle("Target");
target_graph->GetXaxis()->SetTitle("X");
target_graph->GetYaxis()->SetTitle("Y(Target)");
target_graph->SetMarkerColor(2);
target_graph->SetMarkerStyle(24);
target_graph->SetMarkerSize(0.7);


model_canvas->cd(1);
out_graph->Draw("SAME*");
out_graph->SetTitle("Model Prediction");
out_graph->GetXaxis()->SetTitle("X");
out_graph->GetYaxis()->SetTitle("Y(Predicted)");
out_graph->SetMarkerColor(0);
out_graph->SetMarkerStyle(24);
out_graph->SetMarkerSize(0.7);

*/


/*TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);

model_canvas->cd(2);
loss_graph->Draw("AC");
loss_graph->SetTitle("Model Loss");
loss_graph->GetXaxis()->SetTitle("Epochs");
loss_graph->GetYaxis()->SetTitle("Loss (Quadratic)");
loss_graph->SetLineColor(0);*/

TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();

weight_canvas->cd();
 model_histo->Draw("COLZ");

theApp->Run();

return 0;

}
