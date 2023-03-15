#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"
#include "exc_energy.h"

using namespace std::chrono;

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("Exp Data Classification");

  TApplication* theApp = new TApplication("Classification_check", 0, 0);


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


  float fPolarMax = TMath::Pi();
  float fAzimuthalMax = 2*TMath::Pi();
  float fClusterEnergyMax = 400;
  float fSingleCrystalEnergyMax = 340;
  float fPrimEnergyMax = 700;

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "../SoKAI/macros/quasifree_reconstruction/files/238U_Quasifree_Exp_Data.root";
  TString crystalString = "../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

  TH2F *hCorr_crystal_distribution;

  TFile *crystalFile;
  crystalFile = TFile::Open(crystalString);

  char name[100];
  Double_t randTheta_1,randPhi_1,randTheta_2,randPhi_2;

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  Float_t rClusterEnergy[2];
  TBranch  *energyBranch = eventTree->GetBranch("ClusterEnergy");
  energyBranch->SetAddress(rClusterEnergy);

  Float_t rPolar[2];
  TBranch  *polarBranch = eventTree->GetBranch("ClusterTheta");
  polarBranch->SetAddress(rPolar);

  Float_t rAzimuthal[2];
  TBranch  *aziBranch = eventTree->GetBranch("ClusterPhi");
  aziBranch->SetAddress(rAzimuthal);


  Float_t rMotherCrystalEnergy[2];
  TBranch  *singleBranch = eventTree->GetBranch("MotherCrystalEnergy");
  singleBranch->SetAddress(rMotherCrystalEnergy);


  Int_t rMotherId[2];
  TBranch *idBranch = eventTree->GetBranch("MotherId");
  idBranch->SetAddress(rMotherId);



  int nEvents = eventTree->GetEntries();

  TH2F **hCorr_classified_kinematics;
  hCorr_classified_kinematics = new TH2F*[8];


  for(int i = 0 ; i < 4 ; i++){

      sprintf(name, "hCorr_classified_kinematics_%i_0", i + 1);
      hCorr_classified_kinematics[2*i] = new TH2F(name,name,250,18,70,200,0,400);
      hCorr_classified_kinematics[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

      sprintf(name, "hCorr_classified_kinematics_%i_1", i + 1);
      hCorr_classified_kinematics[2*i+1] = new TH2F(name,name,250,18,70,200,0,400);
      hCorr_classified_kinematics[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

  }

  TH2F *hCorr_punched = new TH2F("hCorr_punched","Classification: Punched",250,18,70,200,0,400);
  TH2F *hCorr_stopped = new TH2F("hCorr_stopped","Classification: Stopped",250,18,70,200,0,400);

  TH2F *hCorr_final_kinematics = new TH2F("hCorr_final_kinematics","Final Reconstruction",250,18,70,350,0,700);
  TH2F *hCorr_raw_kinematics = new TH2F("hCorr_raw_kinematics","Raw Kinematics",250,18,70,350,0,700);

  TH1F *hRaw_excitation_energy = new TH1F("hRaw_excitation_energy","Raw Excitation Energy",400,-100,100);
  TH1F *hReconstructed_excitation_energy = new TH1F("hReconstructed_excitation_energy","Reconstructed Excitation Energy",400,-100,100);


  /* ------ Classification Model ------ */

  SKLayer   *layer_1_class = new SKLayer(6,"LeakyReLU");
  SKWeights *weights_12_class = new SKWeights(6,10);
  SKWeights *gradients_12_class = new SKWeights(6,10);


  SKLayer   *layer_2_class = new SKLayer(10,"LeakyReLU");
  SKWeights *weights_23_class = new SKWeights(10,10);
  SKWeights *gradients_23_class = new SKWeights(10,10);

  SKLayer   *layer_3_class = new SKLayer(10,"LeakyReLU");
  SKWeights *weights_34_class = new SKWeights(10,4);
  SKWeights *gradients_34_class = new SKWeights(10,4);


  SKLayer   *layer_4_class = new SKLayer(4,"LeakyReLU");

  weights_12_class->Init(seed);
  gradients_12_class->InitGradients();

  weights_23_class->Init(seed);
  gradients_23_class->InitGradients();

  weights_34_class->Init(seed);
  gradients_34_class->InitGradients();


  SKModel *model_class = new SKModel("Classification");

  model_class->AddLayer(layer_1_class);
  model_class->AddWeights(weights_12_class);
  model_class->AddGradients(gradients_12_class);


  model_class->AddLayer(layer_2_class);
  model_class->AddWeights(weights_23_class);
  model_class->AddGradients(gradients_23_class);

  model_class->AddLayer(layer_3_class);
  model_class->AddWeights(weights_34_class);
  model_class->AddGradients(gradients_34_class);

  model_class->AddLayer(layer_4_class);

  model_class->SetInputSample(&data_sample);

  model_class->SetSummaryFile("test","1");

  model_class->Init();


  model_class->LoadWeights("model_weights_classification_2.txt");


  /* ----- Reconstruction Model ----- */

  SKLayer   *layer_1_reco = new SKLayer(6,"LeakyReLU");
  SKWeights *weights_12_reco = new SKWeights(6,12);
  SKWeights *gradients_12_reco = new SKWeights(6,12);


  SKLayer   *layer_2_reco = new SKLayer(12,"Sigmoid");
  SKWeights *weights_23_reco = new SKWeights(12,12);
  SKWeights *gradients_23_reco = new SKWeights(12,12);

  SKLayer   *layer_3_reco = new SKLayer(12,"LeakyReLU");
  SKWeights *weights_34_reco = new SKWeights(12,1);
  SKWeights *gradients_34_reco = new SKWeights(12,1);


  SKLayer   *layer_4_reco = new SKLayer(1,"LeakyReLU");

  weights_12_reco->Init(seed);
  gradients_12_reco->InitGradients();

  weights_23_reco->Init(seed);
  gradients_23_reco->InitGradients();

  weights_34_reco->Init(seed);
  gradients_34_reco->InitGradients();


  SKModel *model_reco = new SKModel("Regression");

  model_reco->AddLayer(layer_1_reco);
  model_reco->AddWeights(weights_12_reco);
  model_reco->AddGradients(gradients_12_reco);


  model_reco->AddLayer(layer_2_reco);
  model_reco->AddWeights(weights_23_reco);
  model_reco->AddGradients(gradients_23_reco);

  model_reco->AddLayer(layer_3_reco);
  model_reco->AddWeights(weights_34_reco);
  model_reco->AddGradients(gradients_34_reco);

  model_reco->AddLayer(layer_4_reco);

  model_reco->SetInputSample(&data_sample);

  model_reco->SetSummaryFile("final_result","1");

  model_reco->Init();


  model_reco->LoadWeights("weights_reconstruction_30.txt");



  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%1000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);

    data_instance.push_back((rMotherCrystalEnergy[0])/fSingleCrystalEnergyMax);
    data_instance.push_back((rMotherCrystalEnergy[1])/fSingleCrystalEnergyMax);

    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rPolar[1]/fPolarMax);


    sprintf(name, "distributionCrystalID_%i",rMotherId[0] - 2432);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi_1,randTheta_1);

    sprintf(name, "distributionCrystalID_%i",rMotherId[1] - 2432);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi_2,randTheta_2);

    hCorr_raw_kinematics->Fill(randTheta_1,rClusterEnergy[0]);
    hCorr_raw_kinematics->Fill(randTheta_2,rClusterEnergy[1]);

    hRaw_excitation_energy->Fill(exc_energy(rClusterEnergy[0],rClusterEnergy[1],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));

    data_sample.push_back(data_instance);

    output_vec = model_class->Propagate(0);

    int highest_index_training = distance(output_vec.begin(),max_element(output_vec.begin(), output_vec.end()));

    output_vec.clear();

    if(highest_index_training == 0){

     hCorr_stopped->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_stopped->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));

     hReconstructed_excitation_energy->Fill(exc_energy(rClusterEnergy[0],rClusterEnergy[1],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));
     hCorr_final_kinematics->Fill(randTheta_1,rClusterEnergy[0]);
     hCorr_final_kinematics->Fill(randTheta_2,rClusterEnergy[1]);

    }


    if(highest_index_training == 1){

     hCorr_stopped->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_punched->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));

     data_instance.clear();
     data_sample.clear();

     data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[1]/fPolarMax);

     data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[0]/fPolarMax);

     hCorr_final_kinematics->Fill(randTheta_1,rClusterEnergy[0]);

     data_sample.push_back(data_instance);
     output_vec = model_reco->Propagate(0);

     hCorr_final_kinematics->Fill(randTheta_2,fPrimEnergyMax*output_vec.at(0));
     hReconstructed_excitation_energy->Fill(exc_energy(rClusterEnergy[0],fPrimEnergyMax*output_vec.at(0),TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));


    }

    if(highest_index_training == 2){

     hCorr_punched->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_stopped->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));


     data_instance.clear();
     data_sample.clear();

     data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[0]/fPolarMax);

     data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[1]/fPolarMax);

     hCorr_final_kinematics->Fill(randTheta_2,rClusterEnergy[1]);

     data_sample.push_back(data_instance);
     output_vec = model_reco->Propagate(0);

     hCorr_final_kinematics->Fill(randTheta_1,fPrimEnergyMax*output_vec.at(0));
     hReconstructed_excitation_energy->Fill(exc_energy(fPrimEnergyMax*output_vec.at(0),rClusterEnergy[0],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));





    }

    if(highest_index_training == 3){
     hCorr_punched->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_punched->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));


     data_instance.clear();
     data_sample.clear();

     data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[0]/fPolarMax);

     data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[1]/fPolarMax);


     data_sample.push_back(data_instance);
     output_vec = model_reco->Propagate(0);

     Float_t firstEnergy = output_vec.at(0);

     hCorr_final_kinematics->Fill(randTheta_1,fPrimEnergyMax*firstEnergy);

     model_reco->Clear();
     output_vec.clear();

     data_sample.clear();
     data_instance.clear();

     data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[1]/fPolarMax);

     data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
     data_instance.push_back(rPolar[0]/fPolarMax);

     data_sample.push_back(data_instance);
     output_vec = model_reco->Propagate(0);

     Float_t secondEnergy = output_vec.at(0);

     hCorr_final_kinematics->Fill(randTheta_2,fPrimEnergyMax*secondEnergy);

     hReconstructed_excitation_energy->Fill(exc_energy(fPrimEnergyMax*firstEnergy,fPrimEnergyMax*secondEnergy,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));


    }



    hCorr_classified_kinematics[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));



    data_instance.clear();
    label_instance.clear();

    model_class->Clear();
    model_reco->Clear();

    data_sample.clear();
    input_labels.clear();


  }


  TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Model");
  kinematics_canvas->Divide(4,2);

  for(int h = 0 ; h < 4 ; h ++){

     kinematics_canvas->cd(h+1);

     hCorr_classified_kinematics[2*h]->Draw("COLZ");

     kinematics_canvas->cd(4 + h + 1);

     hCorr_classified_kinematics[2*h + 1]->Draw("COLZ");


   }

  TCanvas *everything_canvas = new TCanvas("everything_canvas","Model");
  everything_canvas->Divide(2,1);

  everything_canvas->cd(1);
   hCorr_punched->Draw("COLZ");

  everything_canvas->cd(2);
   hCorr_stopped->Draw("COLZ");


  TCanvas *final_kinematics_canvas = new TCanvas("final_kinematics_canvas","Model");
  final_kinematics_canvas->Divide(2,1);

  final_kinematics_canvas->cd(1);
  hCorr_raw_kinematics->Draw("COLZ");

  final_kinematics_canvas->cd(2);
  hCorr_final_kinematics->Draw("COLZ");

  TCanvas *final_exc_canvas = new TCanvas("final_exc_canvas","Model");
  final_exc_canvas->Divide(2,1);

  final_exc_canvas->cd(1);
  hRaw_excitation_energy->Draw("");

  final_exc_canvas->cd(2);
  hReconstructed_excitation_energy->Draw("COLZ");





theApp->Run();

return 0;



}
