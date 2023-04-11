#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

using namespace std::chrono;

/* Instructions

For running:

./CrossSectionPrediction Epochs Samples LearningRate BatchSize H1 f1 f2 f3 Loss ModelNumber

Example : ./CrossSectionPrediction 1000 4000 10 8 16 Sigmoid LeakyReLU LeakyReLU Quadratic 12

*/

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("CrossSectionPrediction");

  TApplication* theApp = new TApplication("Reconstruction", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed            = 2022;
  int epochs          = stoi(argv[1]);
  int nSamples        = stoi(argv[2]);
  int nMiniBatchSize  = stoi(argv[4]);
  float fLearningRate = stoi(argv[3])/1000.;

  real_start = clock();

  /*---- Input Data (Use this format!) ---- */
  vector<vector<double>> data_sample;
  vector<vector<double>> input_labels;

  vector<vector<double>> data_sample_shuffled;
  vector<vector<double>> input_labels_shuffled;



  vector<double> data_instance;
  vector<double> label_instance;


  /*---- For training results ----*/
  vector<double> loss_vec;
  vector<double> output_vec;
  vector<double> output_model;

  vector<double> epoch_vec;

  double quadraticLoss = 0.0;
  double absoluteLoss = 0.0;

  TRandom3 gen(seed);

  /* ------ Normalization : Fill me with correct values!!! ------ */
  float fEnergyMax = 1E9;
  float fFissioningSystemMax = 1E9;
  float fTargetMax = 1E9;
  float fCrossSectionMax = 1E9;


  SKColorScheme();

  /* ------- Reading Root Data -------- */
  TString fileList = "Here goes your file";

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  Float_t rEnergy;
  TBranch  *energyBranch = eventTree->GetBranch("Energy");
  energyBranch->SetAddress(&rEnergy);

  Float_t rFissioningSystem;
  TBranch  *systemBranch = eventTree->GetBranch("FissioningSystem");
  systemBranch->SetAddress(&rFissioningSystem);

  Float_t rTarget;
  TBranch  *targetBranch = eventTree->GetBranch("Target");
  targetBranch->SetAddress(&rTarget);

  Float_t rCrossSection;
  TBranch  *crossBranch = eventTree->GetBranch("CrossSection");
  crossBranch->SetAddress(&rCrossSection);


  /* ----- F. System and Target Names ----- */
  TString sFissioningSystem;
  TBranch  *systemNameBranch = eventTree->GetBranch("sFissioningSystem");
  systemNameBranch->SetAddress(&sFissioningSystem);

  TString sTarget;
  TBranch  *targetNameBranch = eventTree->GetBranch("sTarget");
  targetNameBranch->SetAddress(&sTarget);




  int nEvents = eventTree->GetEntries();

  if(nEvents < nSamples)
  LOG(FATAL)<<"More number of samples than avalaible!!!";


  int eventCounter = 0;


  while (eventCounter < nSamples){

    eventTree->GetEvent(eventCounter);

    eventCounter++;

    data_instance.push_back( rEnergy / fEnergyMax);
    data_instance.push_back( rFissioningSystem / fFissioningSystemMax);
    data_instance.push_back( rTarget / fTargetMax);

    data_sample.push_back(data_instance);

    label_instance.push_back(rCrossSection / fCrossSectionMax);

    input_labels.push_back(label_instance);

    data_instance.clear();
    label_instance.clear();

   }




  int nTrainingSize   = (7.0/10.0)*data_sample.size();
  int nTestSize       = (3.0/10.0)*data_sample.size();

  LOG(INFO)<<"Training Size : "<<nTrainingSize<<" events. Test size : "<<nTestSize<<endl;


  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(3,argv[6]);
  SKWeights *weights_12 = new SKWeights(3,stoi(argv[5]));
  SKWeights *gradients_12 = new SKWeights(3,stoi(argv[5]));
  SKWeights *firstMoment_12 = new SKWeights(3,stoi(argv[5]));
  SKWeights *secondMoment_12 = new SKWeights(3,stoi(argv[5]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[5]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *gradients_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[5]),1);


  SKLayer   *layer_3 = new SKLayer(1,argv[8]);

  weights_12->Init(seed);
  gradients_12->InitGradients();
  firstMoment_12->InitMoment();
  secondMoment_12->InitMoment();

  weights_23->Init(seed);
  gradients_23->InitGradients();
  firstMoment_23->InitMoment();
  secondMoment_23->InitMoment();


  SKModel *model = new SKModel("Regression");

  model->SetOptimizer("Adam");
  model->SetSummaryFile("summary_cross_section_prediction",argv[10]);

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

  model->Init();
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction(argv[9]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<data_sample.size()<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[9];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[6]<<" "<<"3";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[5];
  LOG(INFO)<<"L3 : "<<argv[8]<<" "<<"1";

  /* ---------- Pass Data Through Model ----------*/
   absoluteLoss = 0.0;
   quadraticLoss = 0.0;

   LOG(INFO)<<"Training! (Eye of the tiger sounds in the background...)";

   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < nTrainingSize ; j++){


       int sample_number = nTrainingSize*gen.Rndm();

       model->Train(j);

       absoluteLoss  = absoluteLoss  +  model->AbsoluteLoss();
       quadraticLoss = quadraticLoss +  model->QuadraticLoss();

       model->Clear();
    }

    if((i+1)%10==0){

     LOG(INFO)<<" Quadratic Loss : "<<quadraticLoss/(10*nTrainingSize)<<" Absolute Loss : "<<absoluteLoss/(10*nTrainingSize)<<" Absolute Loss (mb) : "<<fCrossSectionMax*(absoluteLoss)/(10*nTrainingSize)<<". Epoch : "<<i+1;
     loss_vec.push_back(fCrossSectionMax*(absoluteLoss)/(10*nTrainingSize));
     epoch_vec.push_back(i+1);

     quadraticLoss = 0.0;
     absoluteLoss  = 0.0;

   }
 }

real_end = clock();

LOG(INFO)<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s";

/* --------- Testing the model --------- */
TH1F *hAllResidues = new TH1F("hAllResidues","Cross Section prediction - Real",400,-400,400);
/* --- This you have to fill as you want (Maybe a Graph of Energy Vs Neural Network Output for Every FS ?) ---- */

  for (int j = 0 ; j < nTestSize ; j++){

    int sample_number =  nTrainingSize + nTestSize*gen.Rndm();

    output_vec.clear();

    output_vec = model->Propagate(sample_number); /* ----- This is the NN Output ....*/

    hAllResidues->Fill(fCrossSectionMax * (output_vec.at(0) - input_labels.at(sample_number).at(0)));

    model->Clear();


}


TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);


TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();

string weight_filename = "model_weights_cross_sections_regression_";
weight_filename = weight_filename + argv[10] + ".txt";

model->SaveWeights(weight_filename);

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

TCanvas *results_canvas = new TCanvas("results_canvas","Results Canvas");
 results_canvas->cd();

hAllResidues->Draw();

TString filename = "training_results_cross_sections_regression_";
 filename = filename + argv[10] + ".root";

TFile resultsFile(filename,"RECREATE");

 summary_canvas->Write();
 results_canvas->Write();


 theApp->Run();

 return 0;

}
