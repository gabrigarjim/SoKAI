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

  clock_t start, end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed = 2022;
  int epochs = 20000;

  ifstream *iris_data = new ifstream("/home/gabri/CODE/SoKAI/data/iris.csv");

  if(!iris_data->is_open())
   LOG(ERROR)<<"File not opened!!!";


  vector<vector<double>> data_sample;
  vector<double> data_instance;
  vector<double> accuracy_vec;
  vector<double> epoch_vec;
  double accuracy;

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
  SKWeights *weights_12 = new SKWeights(4,16);

  SKLayer   *layer_2 = new SKLayer(16,"Sigmoid");
  SKWeights *weights_23 = new SKWeights(16,3);

  SKLayer   *layer_3 = new SKLayer(3,"Sigmoid");

  weights_12->Init(seed);
  weights_23->Init(seed);

  SKModel *model = new SKModel();

  model->AddLayer(layer_1);
  model->AddWeights(weights_12);

  model->AddLayer(layer_2);
  model->AddWeights(weights_23);

  model->AddLayer(layer_3);

  model->SetInputSample(&data_sample);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(0.01);

  TRandom3 gen(0);

  /* ---------- Pass Data Through Model ----------*/
start = clock();

for (int j = 0 ; j < epochs ; j++){


 for (int i = 0 ; i < data_sample.size() ; i++){

  int sample_number = data_sample.size()*gen.Rndm();

  model->Propagate(sample_number);

  model->Backpropagate();

  model->Clear();

 }

if(j%1000==0){

  LOG(INFO)<<"Epoch : "<<j;
  accuracy = model->Accuracy();
  LOG(INFO)<<"Model Accuracy : "<<accuracy<<" %";

  accuracy_vec.push_back(accuracy);
  epoch_vec.push_back(j);
  end = clock();

  LOG(INFO)<<"Time per 1000 epochs : "<<((float) end - start)/CLOCKS_PER_SEC<<" s";
  start = clock();

 }

}

TGraph *myGraph = new TGraph(epoch_vec.size(),&epoch_vec[0],&accuracy_vec[0]);
myGraph->Draw("AC*");
myGraph->SetTitle("Model Accuracy");
myGraph->GetXaxis()->SetTitle("Epochs");
myGraph->GetYaxis()->SetTitle("Accuracy %");


theApp->Run();

return 0;





}
