#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

using namespace std::chrono;

/* ------- Instructions

For running:

./MassReconstruction Epochs Samples LearningRate BatchSize H1 f1 f2 f3 Loss ModelNumber

Example : ./MassReconstruction 1000 4000 10 8 16 Sigmoid LeakyReLU LeakyReLU Quadratic 12

*/

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("MassReconstruction");

  TApplication* theApp = new TApplication("MassReconstruction", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed            = 2023;
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
  float fFragmentMaxCharge = 50;
  float fXMaxPosTwim = 6;
  float fMaxPolarTwim = 0.045;
  float fMaxPosMwpc = 40;
  float fMaxPosToFWall = 30;
  float fMaxToF = 35.15;

  float fMaxMass = 105;
  float fMaxBRho = 10;
  float fMaxLength = 756;

  SKColorScheme();

  /* ------- Reading Root Data -------- */
  TString fileList = "/home/gabri/CODE/SoKAI/macros/mass_reconstruction/files/Training_data_z40_17ps.root";

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  Float_t rFragmentCharge;
  TBranch  *fragmentBranch = eventTree->GetBranch("FragmentCharge");
  fragmentBranch->SetAddress(&rFragmentCharge);

  Float_t rTwimPosition;
  TBranch  *twimBranch = eventTree->GetBranch("XPosTwim");
  twimBranch->SetAddress(&rTwimPosition);

  Float_t rPolarTwim;
  TBranch  *polarBranch = eventTree->GetBranch("ThetaTwim");
  polarBranch->SetAddress(&rPolarTwim);

  Float_t rPositionMwpc;
  TBranch  *mwpcBranch = eventTree->GetBranch("XPosMwpc3");
  mwpcBranch->SetAddress(&rPositionMwpc);


  Float_t rPositionToFWall;
  TBranch  *tofwallBranch = eventTree->GetBranch("YPosTofWall");
  tofwallBranch->SetAddress(&rPositionToFWall);


  Float_t rToF;
  TBranch  *tofBranch = eventTree->GetBranch("TofTofWall");
  tofBranch->SetAddress(&rToF);



  /* ----- Labels ----- */

  Float_t rMass;
  TBranch  *massBranch = eventTree->GetBranch("Mass");
  massBranch->SetAddress(&rMass);

  Float_t rBRho;
  TBranch  *brhoBranch = eventTree->GetBranch("Bp");
  brhoBranch->SetAddress(&rBRho);

  Float_t rTrackLenght;
  TBranch  *lenghtBranch = eventTree->GetBranch("Length");
  lenghtBranch->SetAddress(&rTrackLenght);


  int nEvents = eventTree->GetEntries();

  if(nEvents < nSamples)
  LOG(FATAL)<<"More number of samples than avalaible!!!";


  int eventCounter = 0;


  while (eventCounter < nSamples){

    eventTree->GetEvent(eventCounter);

    eventCounter++;

    data_instance.push_back( rFragmentCharge / fFragmentMaxCharge);
    data_instance.push_back( rTwimPosition / fXMaxPosTwim);
    data_instance.push_back( rPolarTwim / fMaxPolarTwim);
    data_instance.push_back( rPositionMwpc / fMaxPosMwpc);
    data_instance.push_back( rPositionToFWall / fMaxPosToFWall);
    data_instance.push_back( rToF / fMaxToF);

    data_sample.push_back(data_instance);

    label_instance.push_back(rMass / fMaxMass);
    label_instance.push_back(rBRho / fMaxBRho);
    label_instance.push_back(rTrackLenght / fMaxLength);


    input_labels.push_back(label_instance);

    data_instance.clear();
    label_instance.clear();

   }




  int nTrainingSize   = (6.0/10.0)*data_sample.size();
  int nTestSize       = (4.0/10.0)*data_sample.size();

  LOG(INFO)<<"Training Size : "<<nTrainingSize<<" events. Test size : "<<nTestSize<<endl;


  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(6,argv[6]);
  SKWeights *weights_12 = new SKWeights(6,stoi(argv[5]));
  SKWeights *gradients_12 = new SKWeights(6,stoi(argv[5]));
  SKWeights *firstMoment_12 = new SKWeights(6,stoi(argv[5]));
  SKWeights *secondMoment_12 = new SKWeights(6,stoi(argv[5]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[5]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[5]),3);
  SKWeights *gradients_23 = new SKWeights(stoi(argv[5]),3);
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[5]),3);
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[5]),3);


  SKLayer   *layer_3 = new SKLayer(3,argv[8]);

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
  model->SetSummaryFile("summary_mass_reconstruction_prediction",argv[10]);

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
  model->SetInputSample(&data_sample);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(0.001);
  model->SetLossFunction(argv[9]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);


  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<nTrainingSize<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[9];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[6]<<" "<<"6";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[5];
  LOG(INFO)<<"L3 : "<<argv[8]<<" "<<"3";

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

     LOG(INFO)<<" Quadratic Loss : "<<quadraticLoss/(10*nTrainingSize)<<" Absolute Loss : "<<absoluteLoss/(10*nTrainingSize)<<". Epoch : "<<i+1;
     loss_vec.push_back((absoluteLoss)/(10*nTrainingSize));
     epoch_vec.push_back(i+1);

     quadraticLoss = 0.0;
     absoluteLoss  = 0.0;

   }
 }

real_end = clock();

LOG(INFO)<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s";

/* --------- Testing the model --------- */
TH1F *hMassResidues = new TH1F("hMassResidues","Mass Reconstructed - Real",400,-10,10);
 hMassResidues->GetXaxis()->SetTitle("Reconstructed Mass - Real");
 hMassResidues->GetYaxis()->SetTitle("Counts");
 hMassResidues->SetLineColor(1);

TH1F *hBRhoResidues = new TH1F("hBRhoResidues","B Rho Reconstructed - Real",400,-10,10);
 hBRhoResidues->GetXaxis()->SetTitle("Reconstructed BRho - Real");
 hBRhoResidues->GetYaxis()->SetTitle("Counts");
 hBRhoResidues->SetLineColor(1);

TH1F *hTrackLengthResidues = new TH1F("hTrackLengthResidues","Track Length Reconstructed - Real",400,-40,40);
 hTrackLengthResidues->GetXaxis()->SetTitle("Reconstructed Length - Real");
 hTrackLengthResidues->GetYaxis()->SetTitle("Counts");
 hTrackLengthResidues->SetLineColor(1);

TH2F *hCorr_Aq_Z = new TH2F("hCorr_Aq_Z","A/Q Vs Z",400,35,45,400,0,0);
 hCorr_Aq_Z->GetXaxis()->SetTitle("Z");
 hCorr_Aq_Z->GetYaxis()->SetTitle("A/Q");
 hCorr_Aq_Z->SetLineColor(1);

TH1F *hMassSpectrum = new TH1F("hMassSpectrum","Reconstructed Mass",400,85,105);
 hMassSpectrum->GetXaxis()->SetTitle("Reconstructed Mass");
 hMassSpectrum->GetYaxis()->SetTitle("Counts");
 hMassSpectrum->SetLineColor(1);



  for (int j = 0 ; j < nTestSize ; j++){

    int sample_number =  nTrainingSize + nTestSize*gen.Rndm();

    output_vec.clear();

    output_vec = model->Propagate(sample_number); /* ----- This is the NN Output ....*/

    hMassResidues->Fill(fMaxMass * (output_vec.at(0) - input_labels.at(sample_number).at(0)));
    hBRhoResidues->Fill(fMaxBRho * (output_vec.at(1) - input_labels.at(sample_number).at(1)));
    hTrackLengthResidues->Fill(fMaxLength * (output_vec.at(2) - input_labels.at(sample_number).at(2)));
    hCorr_Aq_Z->Fill(fFragmentMaxCharge*data_sample.at(sample_number).at(0),fMaxMass * output_vec.at(0) / fFragmentMaxCharge*data_sample.at(sample_number).at(0));
    hMassSpectrum->Fill(fMaxMass * output_vec.at(0));
    model->Clear();


}


TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);


TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();

string weight_filename = "model_weights_mass_reconstruction_";
weight_filename = weight_filename + argv[10] + ".txt";

model->SaveWeights(weight_filename);

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

TCanvas *results_canvas = new TCanvas("results_canvas","Results Canvas");
results_canvas->Divide(3,1);

results_canvas->cd(1);
 hMassResidues->Draw("");

results_canvas->cd(2);
 hBRhoResidues->Draw("");

results_canvas->cd(3);
 hTrackLengthResidues->Draw("");

TCanvas *id_canvas = new TCanvas("id_canvas","Results Canvas");
 id_canvas->Divide(2,1);

 id_canvas->cd(1);
  hCorr_Aq_Z->Draw("COLZ");

 id_canvas->cd(2);
  hMassSpectrum->Draw("");

TString filename = "training_mass_reconstruction_";
 filename = filename + argv[10] + ".root";

TFile resultsFile(filename,"RECREATE");

 summary_canvas->Write();
 results_canvas->Write();


 theApp->Run();

 return 0;

}
