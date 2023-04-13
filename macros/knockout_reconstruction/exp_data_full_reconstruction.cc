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
 google::InitGoogleLogging("Knockout Reconstruction Check");

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


  float fPolarMax = TMath::Pi()/2.0;
  float fAzimuthalMax = 2*TMath::Pi();
  float fClusterEnergyMax = 400;
  float fSingleCrystalEnergyMax = 360;
  float fNfMax = 280;
  float fNsMax = 340;
  float fAngularDeviationMax = 0.1;
  float fPrimEnergyMax = 1200;

 SKColorScheme();

 TString fileList = "../../SoKAI/macros/knockout_reconstruction/files/U238_Fission_560AMeV_Experimental.root";

 TString crystalString = "../../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

 TH2F *hCorr_crystal_distribution;

 double exc_energy_raw,exc_energy_reco;

 TFile *crystalFile;
 crystalFile = TFile::Open(crystalString);

 char name[100];
 Double_t randTheta,randPhi;

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

  Float_t rMotherNf[2];
  TBranch  *nfBranch = eventTree->GetBranch("MotherCrystalNf");
  nfBranch->SetAddress(rMotherNf);

  Float_t rMotherNs[2];
  TBranch  *nsBranch = eventTree->GetBranch("MotherCrystalNs");
  nsBranch->SetAddress(rMotherNs);

  Float_t rAngularDeviation[2];
  TBranch  *angBranch = eventTree->GetBranch("AngularDeviation");
  angBranch->SetAddress(rAngularDeviation);

  Float_t rFissionCharges[2];
  TBranch  *chargesBranch = eventTree->GetBranch("FissionCharges");
  chargesBranch->SetAddress(rFissionCharges);

  Int_t rMotherId[2];
  TBranch *idBranch = eventTree->GetBranch("MotherId");
  idBranch->SetAddress(rMotherId);


  /* ------ Reconstruction Model ------ */

  SKLayer   *layer_1_reco = new SKLayer(14,"LeakyReLU");
  SKWeights *weights_12_reco = new SKWeights(14,8);
  SKWeights *gradients_12_reco = new SKWeights(14,8);

  SKLayer   *layer_2_reco = new SKLayer(8,"Sigmoid");
  SKWeights *weights_23_reco = new SKWeights(8,12);
  SKWeights *gradients_23_reco = new SKWeights(8,12);

  SKLayer   *layer_3_reco = new SKLayer(12,"LeakyReLU");
  SKWeights *weights_34_reco = new SKWeights(12,8);
  SKWeights *gradients_34_reco = new SKWeights(12,8);

  SKLayer   *layer_4_reco = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_45_reco = new SKWeights(8,1);
  SKWeights *gradients_45_reco = new SKWeights(8,1);

  SKLayer   *layer_5_reco = new SKLayer(1,"LeakyReLU");


  weights_12_reco->Init(seed);
  gradients_12_reco->InitGradients();

  weights_23_reco->Init(seed);
  gradients_23_reco->InitGradients();

  weights_34_reco->Init(seed);
  gradients_34_reco->InitGradients();

  weights_45_reco->Init(seed);
  gradients_45_reco->InitGradients();



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
  model_reco->AddWeights(weights_45_reco);
  model_reco->AddGradients(gradients_45_reco);


  model_reco->AddLayer(layer_5_reco);

  model_reco->SetInputSample(&data_sample);

  model_reco->SetSummaryFile("test_reconstruction","12");

  model_reco->Init();


  model_reco->LoadWeights("/home/gabri/Analysis/s455/nn_results/knockout_reconstruction/model_weights_knockout_regression_15.txt");

  Int_t nEvents = eventTree->GetEntries();

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;


  /*----------- A lot of Plots ----------- */
  TH2F * hCorr_raw_kinematics = new TH2F("hCorr_raw_kinematics","Raw Kinematics",300,20,80,300,0,650);
  TH2F * hCorr_reconstructed_kinematics = new TH2F("hCorr_reconstructed_kinematics","Reconstructed Kinematics",300,20,80,300,0,650);

  TH1F * hRaw_exc_energy = new TH1F("hRaw_exc_energy","Raw E*",400,-300,600);
  TH1F * hReconstructed_exc_energy = new TH1F("hReconstructed_exc_energy","Reconstructed E*",400,-300,600);

  TH2F * hCorr_exc_energy_charges = new TH2F("hCorr_exc_energy_charges","Excitation Energy Vs Charges",300,-200,500,300,20,90);



  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%1000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    Double_t finalEnergy_1 = rClusterEnergy[0];
    Double_t finalEnergy_2 = rClusterEnergy[1];

    exc_energy_raw = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    hRaw_exc_energy->Fill(exc_energy_raw);

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    data_instance.push_back(rMotherNf[0]/fNfMax);
    data_instance.push_back(rMotherNs[0]/fNsMax);
    data_instance.push_back(rAngularDeviation[0]/fAngularDeviationMax);

    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
    data_instance.push_back(rPolar[1]/fPolarMax);
    data_instance.push_back(rAzimuthal[1]/fAzimuthalMax);
    data_instance.push_back(rMotherNf[1]/fNfMax);
    data_instance.push_back(rMotherNs[1]/fNsMax);
    data_instance.push_back(rAngularDeviation[1]/fAngularDeviationMax);

    data_sample.push_back(data_instance);

    output_vec = model_reco->Propagate(0);
    finalEnergy_1 = output_vec.at(0)*fPrimEnergyMax;


    sprintf(name, "distributionCrystalID_%i",rMotherId[0]-2432);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi,randTheta);

    hCorr_raw_kinematics->Fill(randTheta,fClusterEnergyMax*data_instance.at(0));
    hCorr_reconstructed_kinematics->Fill(randTheta,fPrimEnergyMax*output_vec.at(0));

    data_instance.clear();
    model_reco->Clear();
    output_vec.clear();
    data_sample.clear();


    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
    data_instance.push_back(rPolar[1]/fPolarMax);
    data_instance.push_back(rAzimuthal[1]/fAzimuthalMax);
    data_instance.push_back(rMotherNf[1]/fNfMax);
    data_instance.push_back(rMotherNs[1]/fNsMax);
    data_instance.push_back(rAngularDeviation[1]/fAngularDeviationMax);


    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    data_instance.push_back(rMotherNf[0]/fNfMax);
    data_instance.push_back(rMotherNs[0]/fNsMax);
    data_instance.push_back(rAngularDeviation[0]/fAngularDeviationMax);

    data_sample.push_back(data_instance);

    output_vec = model_reco->Propagate(0);
    finalEnergy_2= output_vec.at(0)*fPrimEnergyMax;


    sprintf(name, "distributionCrystalID_%i",rMotherId[1]-2432);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi,randTheta);

    hCorr_raw_kinematics->Fill(randTheta,fClusterEnergyMax*data_instance.at(0));
    hCorr_reconstructed_kinematics->Fill(randTheta,fPrimEnergyMax*output_vec.at(0));

    data_instance.clear();
    model_reco->Clear();
    data_sample.clear();
    output_vec.clear();

    exc_energy_reco = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    hReconstructed_exc_energy->Fill(exc_energy_reco);
    hCorr_exc_energy_charges->Fill(exc_energy_reco,rFissionCharges[0]);
    hCorr_exc_energy_charges->Fill(exc_energy_reco,rFissionCharges[1]);

     }



    TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Kinematics Canvas");
    kinematics_canvas->Divide(2,1);

    kinematics_canvas->cd(1);
     hCorr_raw_kinematics->Draw("COLZ");

    kinematics_canvas->cd(2);
     hCorr_reconstructed_kinematics->Draw("COLZ");


    TCanvas *exc_energy_canvas = new TCanvas("exc_energy_canvas","exc_energy_canvas");
    exc_energy_canvas->Divide(2,1);

    exc_energy_canvas->cd(1);
    hRaw_exc_energy->Draw("");

    exc_energy_canvas->cd(2);
    hReconstructed_exc_energy->Draw("");

    TCanvas *exc_charges_canvas = new TCanvas("exc_charges_canvas","exc_charges_canvas");
    exc_charges_canvas->cd();

    hCorr_exc_energy_charges->Draw("COLZ");


    theApp->Run();

    return 0;

}
