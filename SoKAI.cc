#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKBackProp.h"
#include "SKModel.h"

int main () {

  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("SoKAI");

  TApplication* theApp = new TApplication("SoKAI", 0, 0);

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed = 2021;

  ifstream *iris_data = new ifstream("/home/gabri/CODE/SoKAI/data/iris.csv");

  if(!iris_data->is_open())
   LOG(ERROR)<<"File not opened!!!";


  vector<vector<float>> data_sample;
  vector<float> data_instance;

  float feature_1,feature_2,feature_3,feature_4,label;


  LOG(INFO)<<"Filling input vector.....";

  TH1F *feature_1_histo = new TH1F("feature_1_histo","Feature 1 ",40,0,10);
  TH1F *feature_2_histo = new TH1F("feature_2_histo","Feature 2 ",40,0,6);
  TH1F *feature_3_histo = new TH1F("feature_3_histo","Feature 3 ",40,0,8);
  TH1F *feature_4_histo = new TH1F("feature_4_histo","Feature 4 ",40,0,6);

  while(*iris_data>>feature_1>>feature_2>>feature_3>>feature_4>>label){

      data_instance.push_back(feature_1);
      data_instance.push_back(feature_2);
      data_instance.push_back(feature_3);
      data_instance.push_back(feature_4);
      data_instance.push_back(label);

      feature_1_histo->Fill(feature_1);
      feature_2_histo->Fill(feature_2);
      feature_3_histo->Fill(feature_3);
      feature_4_histo->Fill(feature_4);

      data_sample.push_back(data_instance);

      data_instance.clear();

    }

  TCanvas *canvas = new TCanvas("canvas","canvas");
  canvas->Divide(2,2);

  canvas->cd(1);
  feature_1_histo->Draw();

  canvas->cd(2);
  feature_2_histo->Draw();

  canvas->cd(3);
  feature_3_histo->Draw();

  canvas->cd(4);
  feature_4_histo->Draw();

  SKWeights *input_layer_weights = new SKWeights(4,1);
  SKLayer   *layer_1 = new SKLayer(4,"Sigmoid");
  SKWeights *weights_12 = new SKWeights(4,3);

  SKLayer   *layer_2 = new SKLayer(4,"Sigmoid");

  SKPropagator * prop_i1 = new SKPropagator();
  SKPropagator * prop_12 = new SKPropagator();

  SKBackProp *backPropagator = new SKBackProp();
  backPropagator->SetLoss("Quadratic");

  input_layer_weights->Init(seed);
  weights_12->Init(seed);

  SKModel *model = new SKModel();

  model->AddWeights(input_layer_weights);
  model->AddLayer(layer_1);
  model->AddWeights(weights_12);
  model->AddLayer(layer_2);

  model->Init();

  theApp->Run();

  return 0;





}
