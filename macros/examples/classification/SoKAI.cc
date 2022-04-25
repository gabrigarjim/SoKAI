#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKFancyPlots.h"

int main () {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("SoKAI");

  TApplication* theApp = new TApplication("SoKAI", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed = 2022;
  int epochs = 20000;

  ifstream *iris_data = new ifstream("../SoKAI/data/iris.csv");

  if(!iris_data->is_open())
   LOG(ERROR)<<"File not opened!!!";

  real_start = clock();
  vector<vector<double>> data_sample;
  vector<double> data_instance;
  vector<double> accuracy_vec;
  vector<double> epoch_vec;
  double accuracy;

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



  TH2F *model_histo;

  vector<vector<double>> input_labels;

  double feature_1,feature_2,feature_3,feature_4;
  int label;
  double max_feature_1=0.0;
  double max_feature_2=0.0;
  double max_feature_3=0.0;
  double max_feature_4=0.0;



  LOG(INFO)<<"Filling input vector.....";

  TH1F *feature_1_histo = new TH1F("feature_1_histo","Feature 1 ",40,0,10);
  TH1F *feature_2_histo = new TH1F("feature_2_histo","Feature 2 ",40,0,6);
  TH1F *feature_3_histo = new TH1F("feature_3_histo","Feature 3 ",40,0,8);
  TH1F *feature_4_histo = new TH1F("feature_4_histo","Feature 4 ",40,0,6);
  TH1F *epoch_time = new TH1F("epoch_time","Epoch Time",200,15,30);

  while(*iris_data>>feature_1>>feature_2>>feature_3>>feature_4>>label){

      data_instance.push_back(feature_1);
      data_instance.push_back(feature_2);
      data_instance.push_back(feature_3);
      data_instance.push_back(feature_4);

      feature_1_histo->Fill(feature_1);
      feature_2_histo->Fill(feature_2);
      feature_3_histo->Fill(feature_3);
      feature_4_histo->Fill(feature_4);

      if(feature_1 > max_feature_1)
       max_feature_1 = feature_1;

      if(feature_2 > max_feature_2)
       max_feature_2 = feature_2;

      if(feature_3 > max_feature_3)
       max_feature_3 = feature_3;

      if(feature_4 > max_feature_4)
       max_feature_4=feature_4;


      data_sample.push_back(data_instance);

      vector<double> label_instance(3,0.0);

      /* ------ One Hot Encoding ------*/
      switch (label) {
        case 1:
        label_instance.at(0)=0.75;
        break;

        case 2:
        label_instance.at(1)=0.75;
        break;

        case 3:
        label_instance.at(2)=0.75;
        break;

      }

      input_labels.push_back(label_instance);

      label_instance.clear();
      data_instance.clear();

    }


    cout<<"Max value feature 1 : "<<max_feature_1<<endl;
    cout<<"Max value feature 2 : "<<max_feature_2<<endl;
    cout<<"Max value feature 3 : "<<max_feature_3<<endl;
    cout<<"Max value feature 4 : "<<max_feature_4<<endl;


    for (int i = 0 ; i < data_sample.size() ; i++){

        data_sample[i][0]=data_sample[i][0]/max_feature_1;
        data_sample[i][1]=data_sample[i][1]/max_feature_2;
        data_sample[i][2]=data_sample[i][2]/max_feature_3;
        data_sample[i][3]=data_sample[i][3]/max_feature_4;

    }



  SKLayer   *layer_1 = new SKLayer(4,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(4,10);
  SKWeights *gradients_12 = new SKWeights(4,10);

  SKLayer   *layer_2 = new SKLayer(10,"Sigmoid");
  SKWeights *weights_23 = new SKWeights(10,3);
  SKWeights *gradients_23 = new SKWeights(10,3);

  SKLayer   *layer_3 = new SKLayer(3,"Sigmoid");


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

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(1);


  TRandom3 gen(0);

  /* ---------- Pass Data Through Model ----------*/

   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < data_sample.size() ; j++){

      int sample_number = data_sample.size()*gen.Rndm();



      model->Train(sample_number);
      model->Clear();


   }

    if(i%100 == 0){

     accuracy = model->Accuracy();

     accuracy_vec.push_back(accuracy);
     epoch_vec.push_back(i);
     end = clock();

     LOG(INFO)<<"Time per 1000 epochs : "<<((float) end -start)/CLOCKS_PER_SEC<<" s "<<" Accuraccy : "<<accuracy<<" % . Epoch : "<<i;
     start = clock();

 }
}
real_end = clock();

model_histo = (TH2F*)model->ShowMe();
cout<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s"<<endl;

TCanvas *model_canvas = new TCanvas("model_canvas","Model");
model_canvas->Divide(2,1);

model_canvas->cd(1);
model_histo->Draw("COLZ");

TGraph *myGraph = new TGraph(epoch_vec.size(),&epoch_vec[0],&accuracy_vec[0]);

model_canvas->cd(2);
myGraph->Draw("AC");
myGraph->SetTitle("Model Accuracy");
myGraph->GetXaxis()->SetTitle("Epochs");
myGraph->GetYaxis()->SetTitle("Accuracy %");
myGraph->SetLineColor(0);


theApp->Run();

return 0;





}
