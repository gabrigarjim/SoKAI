#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"


using namespace std::chrono;

int main (int argc, char** argv) {

 FLAGS_alsologtostderr = 1;
setenv("GLOG_logtostderr", "1", 0);
    setenv("GLOG_colorlogtostderr", "1", 0);
 google::InitGoogleLogging("Mass Reconstruction Check");

 TApplication* theApp = new TApplication("Reconstruction_check", 0, 0);

 clock_t start, end,real_start,real_end;

 LOG(INFO)<<"#============================================================#";
 LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
 LOG(INFO)<<"#============================================================#";

 real_start = clock();

 /*---- Input Data (Use this format!) ---- */
 vector<vector<double>> data_sample;
 vector<double> data_instance;
 vector<vector<double>> input_labels;
 vector<double> label_instance;

 /*---- For training results ----*/
 vector<double> output_vec;
 int seed = 2022;

 TRandom3 gen(seed);


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


  /* ------ Reconstruction Model ------ */

  SKLayer   *layer_1_reco = new SKLayer(6,"LeakyReLU");
  SKWeights *weights_12_reco = new SKWeights(6,20);
  SKWeights *gradients_12_reco = new SKWeights(6,20);

  SKLayer   *layer_2_reco = new SKLayer(20,"Sigmoid");
  SKWeights *weights_23_reco = new SKWeights(20,3);
  SKWeights *gradients_23_reco = new SKWeights(20,3);

  SKLayer   *layer_3_reco = new SKLayer(3,"Sigmoid");




  weights_12_reco->Init(seed);
  gradients_12_reco->InitGradients();

  weights_23_reco->Init(seed);
  gradients_23_reco->InitGradients();


  SKModel *model_reco = new SKModel("Regression");

  model_reco->AddLayer(layer_1_reco);
  model_reco->AddWeights(weights_12_reco);
  model_reco->AddGradients(gradients_12_reco);


  model_reco->AddLayer(layer_2_reco);
  model_reco->AddWeights(weights_23_reco);
  model_reco->AddGradients(gradients_23_reco);

  model_reco->AddLayer(layer_3_reco);

  model_reco->SetInputSample(&data_sample);

  model_reco->SetSummaryFile("test_reconstruction","4");

  model_reco->Init();


  model_reco->LoadWeights("training_antia/model_weights_mass_reconstruction_11.txt");

  Int_t nEvents = eventTree->GetEntries();

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;


  /*----------- A lot of Plots ----------- */
  TH1F *hMassResidues = new TH1F("hMassResidues","Mass Reconstructed - Real",400,-10,10);
   hMassResidues->GetXaxis()->SetTitle("Reconstructed Mass - Real");
   hMassResidues->GetYaxis()->SetTitle("Counts");
   hMassResidues->SetLineColor(1);

  TH1F *hBRhoResidues = new TH1F("hBRhoResidues","B Rho Reconstructed - Real",400,-2,2);
   hBRhoResidues->GetXaxis()->SetTitle("Reconstructed BRho - Real");
   hBRhoResidues->GetYaxis()->SetTitle("Counts");
   hBRhoResidues->SetLineColor(1);

  TH1F *hTrackLengthResidues = new TH1F("hTrackLengthResidues","Track Length Reconstructed - Real",400,-10,10);
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



  for (Int_t j = 0.7*nEvents; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;



    data_instance.push_back( rFragmentCharge / fFragmentMaxCharge);
    data_instance.push_back( rTwimPosition / fXMaxPosTwim);
    data_instance.push_back( rPolarTwim / fMaxPolarTwim);
    data_instance.push_back( rPositionMwpc / fMaxPosMwpc);
    data_instance.push_back( rPositionToFWall / fMaxPosToFWall);
    data_instance.push_back( rToF / fMaxToF);

    label_instance.push_back(rMass / fMaxMass);
    label_instance.push_back(rBRho / fMaxBRho);
    label_instance.push_back(rTrackLenght / fMaxLength);


    input_labels.push_back(label_instance);


    data_sample.push_back(data_instance);

    output_vec = model_reco->Propagate(0);

    hMassResidues->Fill(fMaxMass * (output_vec.at(0) - input_labels.at(0).at(0)));
    hBRhoResidues->Fill(fMaxBRho * (output_vec.at(1) - input_labels.at(0).at(1)));
    hTrackLengthResidues->Fill(fMaxLength * (output_vec.at(2) - input_labels.at(0).at(2)));
    hCorr_Aq_Z->Fill(fFragmentMaxCharge*data_sample.at(0).at(0),fMaxMass * output_vec.at(0) / fFragmentMaxCharge*data_sample.at(0).at(0));
    hMassSpectrum->Fill(fMaxMass * output_vec.at(0));

    data_instance.clear();
    model_reco->Clear();
    output_vec.clear();
    data_sample.clear();
    label_instance.clear();
    input_labels.clear();


   }


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

   TString s = "training_antia/results_file_training_11.root";

   TFile resultsFile(s,"RECREATE");

   id_canvas->Write();
   results_canvas->Write();

   resultsFile.Close();
   theApp->Run();

  return 0;

}
