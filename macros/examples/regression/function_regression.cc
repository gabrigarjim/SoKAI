#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKFancyPlots.h"

int main () {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("FunctionRegression");

  TApplication* theApp = new TApplication("regression", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed = 2022;
  int epochs = 100000;
  int nSamples = 400;


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

  TRandom3 gen(2022);


  /* -------- Put this on a header or something..... ----------*/
  gStyle->SetOptStat(0);
  Double_t Red[5]    = { 0.06, 0.25, 0.50, 0.75, 1.0};
  Double_t Green[5]  = {0.01, 0.1, 0.15, 0.20, 0.8};
  Double_t Blue[5]   = { 0.00, 0.00, 0.00, 0.0, 0.0};
  Double_t Length[5] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
  Int_t nb=250;

  TColor::CreateGradientColorTable(5,Length,Red,Green,Blue,nb);
   gStyle->SetCanvasColor(1);
   gStyle->SetTitleFillColor(1);
   gStyle->SetStatColor(1);

   gStyle->SetFrameLineColor(0);
   gStyle->SetGridColor(0);
   gStyle->SetStatTextColor(0);
   gStyle->SetTitleTextColor(0);
   gStyle->SetLabelColor(0,"xyz");
   gStyle->SetTitleColor(0,"xyz");
   gStyle->SetAxisColor(0,"xyz");
   /* --------------------------------------------------------*/

   double x,y;

   for (int i = 0 ; i < nSamples ; i++){

      x = gen.Uniform(-2,2);
      y = x*x;

      data_instance.push_back(x);
      data_sample.push_back(data_instance);

      label_instance.push_back(y);
      input_labels.push_back(label_instance);


      data_instance.clear();
      label_instance.clear();

   }




  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(1,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(1,4);
  SKWeights *gradients_12 = new SKWeights(1,4);

  SKLayer   *layer_2 = new SKLayer(4,"Sigmoid");
  SKWeights *weights_23 = new SKWeights(4,1);
  SKWeights *gradients_23 = new SKWeights(4,1);


  SKLayer   *layer_3 = new SKLayer(1,"Linear");


  weights_12->Init(seed);
  gradients_12->InitGradients();

  weights_23->Init(seed);
  gradients_23->InitGradients();


  SKModel *model = new SKModel();

  model->AddLayer(layer_1);
  model->AddWeights(weights_12);
  model->AddGradients(gradients_12);

  model->AddLayer(layer_2);
  model->AddWeights(weights_23);
  model->AddGradients(gradients_23);

  model->AddLayer(layer_3);


  model->SetInputSample(&data_sample);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(0.01);
  model->SetLossFunction("Quadratic");

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(8);


  int iterCounter=0;

  /* ---------- Pass Data Through Model ----------*/

   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < data_sample.size()/2 ; j++){


      // Using only half of the dataset to train the network
      int sample_number = (data_sample.size()/2)*gen.Rndm();


      model->Train(sample_number);

      loss =  model->AbsoluteLoss();

      model->Clear();
   }

    if(i%1000==0){

     LOG(INFO)<<" Loss : "<<loss<<" . Epoch : "<<i;
     loss_vec.push_back(loss);
     epoch_vec.push_back(i);

   }

}

real_end = clock();

cout<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s"<<endl;

/* --------- Testing the model --------- */

  for (int j = 0 ; j < data_sample.size()/2 ; j++){


    // Using only the other half of the dataset to test the network
    int sample_number = (data_sample.size()/2)*gen.Rndm() + (data_sample.size()/2);

    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    output_model.push_back(output_vec.at(output_vec.size()-1));

    x_vec.push_back(data_sample.at(sample_number).at(0));

    target_vec.push_back(data_sample.at(sample_number).at(0)*data_sample.at(sample_number).at(0));

    model->Clear();


}




/* --------- Plots and so on ....... ------*/

TCanvas *model_canvas = new TCanvas("model_canvas","Model");
model_canvas->Divide(2,1);



TGraph *target_graph = new TGraph(x_vec.size(),&x_vec[0],&target_vec[0]);
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




TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);

model_canvas->cd(2);
loss_graph->Draw("AC");
loss_graph->SetTitle("Model Loss");
loss_graph->GetXaxis()->SetTitle("Epochs");
loss_graph->GetYaxis()->SetTitle("Loss (Quadratic)");
loss_graph->SetLineColor(0);




theApp->Run();

return 0;



}
