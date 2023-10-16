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

 TString fileList = "../../SoKAI/macros/knockout_reconstruction/files/U238_Fission_560AMeV_Experimental_Masses.root";

 TString crystalString = "../../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

 TH2F *hCorr_crystal_distribution;

 double exc_energy_raw,exc_energy_reco;

 TFile *crystalFile;
 crystalFile = TFile::Open(crystalString);

 char name[100];
 Double_t randTheta,randPhi;

  TFile *eventFile;
  TTree* eventTree;

  TFile* thefile = new TFile("exc_energy_masses_trampeado.root");
  TCutG *mycutg ;
  thefile->GetObject("CUTG",mycutg);



  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  TFile *f = new TFile("all_charges.root","RECREATE");

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

  Float_t rFissionMasses[2];
  TBranch  *massBranch = eventTree->GetBranch("FissionMasses");
  massBranch->SetAddress(rFissionMasses);


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


  model_reco->LoadWeights("/home/gabri/Analysis/s455/nn_results/knockout_reconstruction/model_weights_knockout_regression_12.txt");

  Int_t nEvents = eventTree->GetEntries();

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;


  /*----------- A lot of Plots ----------- */
  TH2F * hCorr_raw_kinematics = new TH2F("hCorr_raw_kinematics","Raw Kinematics",300,20,80,300,0,650);
  TH2F * hCorr_reconstructed_kinematics = new TH2F("hCorr_reconstructed_kinematics","Reconstructed Kinematics",300,20,80,300,0,650);

  TH1F * hRaw_exc_energy = new TH1F("hRaw_exc_energy","Raw E*",400,-300,600);
  TH1F * hReconstructed_exc_energy = new TH1F("hReconstructed_exc_energy","Reconstructed E*",400,-300,600);

  TH2F * hCorr_exc_energy_opening = new TH2F("hCorr_exc_energy_opening","Excitation Energy Vs Opening Angle",300,0,100,300,-200,500);
  TH2F * hCorr_exc_energy_charges = new TH2F("hCorr_exc_energy_charges","Excitation Energy Vs Charges",300,-200,500,300,20,90);
  TH2F * hCorr_exc_energy_masses = new TH2F("hCorr_exc_energy_masses","Excitation Energy Vs Masses",300,-200,500,300,40,180);

  TH1F * hFissioning_system = new TH1F("hFissioning_system","Z1 + Z2*",400,60,100);
  TH2F * hCorr_exc_energy_mass_sum = new TH2F("hCorr_exc_energy_mass_sum","Excitation Energy Vs Mass Sum",300,190,250,300,-200,500);

  TH2F * hCorr_charges_mass_sum = new TH2F("hCorr_charges_mass_sum","Charges Vs Mass Sum",300,20,90,300,190,250);

  TH1F * hMassSum_1 = new TH1F("hMassSum_1","A1 + A2 : E* -> (-20,30)",400,190,250);
  TH1F * hMassSum_2 = new TH1F("hMassSum_2","A1 + A2 : E* -> (30,80)",400,190,250);
  TH1F * hMassSum_3 = new TH1F("hMassSum_3","A1 + A2 : E* -> (80,130)",400,190,250);
  TH1F * hMassSum_4 = new TH1F("hMassSum_4","A1 + A2 : E* -> (130,180)",400,190,250);
  TH1F * hMassSum_5 = new TH1F("hMassSum_5","A1 + A2 : E* -> (180,230)",400,190,250);
  TH1F * hMassSum_6 = new TH1F("hMassSum_6","A1 + A2 : E* -> (230,280)",400,190,250);
  TH1F * hMassSum_7 = new TH1F("hMassSum_7","A1 + A2 : E* -> (280,330)",400,190,250);


  TH1F * hFragmentMasses_1 = new TH1F("hFragmentMasses_1","Fragment Masses : E* -> (-20,30)",400,60,160);
  TH1F * hFragmentMasses_2 = new TH1F("hFragmentMasses_2","Fragment Masses : E* -> (30,80)",400,60,160);
  TH1F * hFragmentMasses_3 = new TH1F("hFragmentMasses_3","Fragment Masses : E* -> (80,130)",400,60,160);
  TH1F * hFragmentMasses_4 = new TH1F("hFragmentMasses_4","Fragment Masses : E* -> (130,180)",400,60,160);
  TH1F * hFragmentMasses_5 = new TH1F("hFragmentMasses_5","Fragment Masses : E* -> (180,230)",400,60,160);
  TH1F * hFragmentMasses_6 = new TH1F("hFragmentMasses_6","Fragment Masses : E* -> (230,280)",400,60,160);
  TH1F * hFragmentMasses_7 = new TH1F("hFragmentMasses_7","Fragment Masses : E* -> (280,330)",400,60,160);


  TH1F * hFragmentCharges_1 = new TH1F("hFragmentCharges_1","Fragment Charges : E* -> (-20,30)",400,25,65);
  TH1F * hFragmentCharges_2 = new TH1F("hFragmentCharges_2","Fragment Charges : E* -> (30,80)",400,25,65);
  TH1F * hFragmentCharges_3 = new TH1F("hFragmentCharges_3","Fragment Charges : E* -> (80,130)",400,25,65);
  TH1F * hFragmentCharges_4 = new TH1F("hFragmentCharges_4","Fragment Charges : E* -> (130,180)",400,25,65);
  TH1F * hFragmentCharges_5 = new TH1F("hFragmentCharges_5","Fragment Charges : E* -> (180,230)",400,25,65);
  TH1F * hFragmentCharges_6 = new TH1F("hFragmentCharges_6","Fragment Charges : E* -> (230,280)",400,25,65);
  TH1F * hFragmentCharges_7 = new TH1F("hFragmentCharges_7","Fragment Charges : E* -> (280,330)",400,25,65);

  TH1F * hFragmentCharges_Mass_Cut = new TH1F("hFragmentCharges_Mass_Cut","Fragment Charges : M = 236",400,25,65);
  TH1F * hExcitationEnergy = new TH1F("hExcitationEnergy","Excitation Energy : M = 236 ",100,-200,500);

  hFragmentCharges_Mass_Cut->SetLineColor(1);
  hExcitationEnergy->SetLineColor(1);

  hFragmentCharges_Mass_Cut->GetXaxis()->SetTitle("Fission Fragment Charges (Z Units)");
  hExcitationEnergy->GetXaxis()->SetTitle("Reconstructed E* (MeV)");

  // std::vector<int> vMasses = {220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237};
  std::vector<int> vMasses = {220};

  Float_t polarFActor = 4*0.017453293;
  Float_t nFissions;

  std::vector<Double_t> vExc_energy;
  std::vector<Double_t> vFissions;


  for(int p = 0 ; p < vMasses.size() ; p++){


   for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    Double_t finalEnergy_1 = rClusterEnergy[0];
    Double_t finalEnergy_2 = rClusterEnergy[1];

    exc_energy_raw = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    // hRaw_exc_energy->Fill(exc_energy_raw);

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

    Float_t openingAngle = TMath::Sin(rPolar[0])*TMath::Sin(rPolar[1])*TMath::Cos(rAzimuthal[1]-rAzimuthal[0]) + TMath::Cos(rPolar[0])*TMath::Cos(rPolar[1]);
    openingAngle = TMath::RadToDeg()*TMath::ACos(openingAngle);


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

    exc_energy_reco = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0] -2,TMath::RadToDeg()*rPolar[1]-2,TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    hReconstructed_exc_energy->Fill(exc_energy_reco);

    hCorr_exc_energy_charges->Fill(exc_energy_reco,rFissionCharges[0]);
    hCorr_exc_energy_charges->Fill(exc_energy_reco,rFissionCharges[1]);

    hCorr_exc_energy_masses->Fill(exc_energy_reco,rFissionMasses[0]);
    hCorr_exc_energy_masses->Fill(exc_energy_reco,rFissionMasses[1]);
    hFissioning_system->Fill(rFissionCharges[1] + rFissionCharges[0]);

    hCorr_exc_energy_opening->Fill(openingAngle,exc_energy_reco);

    if( (rFissionMasses[0] + rFissionMasses[1]) > 220 && (rFissionMasses[0] + rFissionMasses[1]) < 237){
     hCorr_charges_mass_sum->Fill(rFissionCharges[1],rFissionMasses[0] + rFissionMasses[1]);
     hCorr_charges_mass_sum->Fill(rFissionCharges[0],rFissionMasses[0] + rFissionMasses[1]);
   }

    // if(mycutg->IsInside(rFissionMasses[0] + rFissionMasses[1] , exc_energy_reco))

    if(openingAngle>78)
    hCorr_exc_energy_mass_sum->Fill(rFissionMasses[0] + rFissionMasses[1] , exc_energy_reco);


    if((rFissionMasses[0] + rFissionMasses[1]) > vMasses.at(p) - 1 && (rFissionMasses[0] + rFissionMasses[1]) < vMasses.at(p) + 1){

      hFragmentCharges_Mass_Cut->Fill(round(rFissionCharges[0]));
      hFragmentCharges_Mass_Cut->Fill(round(rFissionCharges[1]));
      hExcitationEnergy->Fill(exc_energy_reco);

    }

    if(exc_energy_reco > -20 && exc_energy_reco < 30){

     hFragmentMasses_1->Fill(rFissionMasses[0]);
     hFragmentMasses_1->Fill(rFissionMasses[1]);

     hFragmentCharges_1->Fill(rFissionCharges[0]);
     hFragmentCharges_1->Fill(rFissionCharges[1]);

     hMassSum_1->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }

    if(exc_energy_reco > 30 && exc_energy_reco < 80){

     hFragmentMasses_2->Fill(rFissionMasses[0]);
     hFragmentMasses_2->Fill(rFissionMasses[1]);

     hFragmentCharges_2->Fill(rFissionCharges[0]);
     hFragmentCharges_2->Fill(rFissionCharges[1]);

     hMassSum_2->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }


    if(exc_energy_reco > 80 && exc_energy_reco < 130){

     hFragmentMasses_3->Fill(rFissionMasses[0]);
     hFragmentMasses_3->Fill(rFissionMasses[1]);

     hFragmentCharges_3->Fill(rFissionCharges[0]);
     hFragmentCharges_3->Fill(rFissionCharges[1]);

     hMassSum_3->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }


    if(exc_energy_reco > 130 && exc_energy_reco < 180){

     hFragmentMasses_4->Fill(rFissionMasses[0]);
     hFragmentMasses_4->Fill(rFissionMasses[1]);

     hFragmentCharges_4->Fill(rFissionCharges[0]);
     hFragmentCharges_4->Fill(rFissionCharges[1]);

     hMassSum_4->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }


    if(exc_energy_reco > 180 && exc_energy_reco < 230){

     hFragmentMasses_5->Fill(rFissionMasses[0]);
     hFragmentMasses_5->Fill(rFissionMasses[1]);

     hFragmentCharges_5->Fill(rFissionCharges[0]);
     hFragmentCharges_5->Fill(rFissionCharges[1]);

     hMassSum_5->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }


    if(exc_energy_reco > 230 && exc_energy_reco < 280){

     hFragmentMasses_6->Fill(rFissionMasses[0]);
     hFragmentMasses_6->Fill(rFissionMasses[1]);

     hFragmentCharges_6->Fill(rFissionCharges[0]);
     hFragmentCharges_6->Fill(rFissionCharges[1]);

     hMassSum_6->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }

    if(exc_energy_reco > 280 && exc_energy_reco < 330){

     hFragmentMasses_7->Fill(rFissionMasses[0]);
     hFragmentMasses_7->Fill(rFissionMasses[1]);

     hFragmentCharges_7->Fill(rFissionCharges[0]);
     hFragmentCharges_7->Fill(rFissionCharges[1]);

     hMassSum_7->Fill(rFissionMasses[0] + rFissionMasses[1]);

    }



     }

      Double_t calib = 13233.5 -105.87 * vMasses.at(p) + 0.211144 *  vMasses.at(p) *  vMasses.at(p);

      TString s = "Fission Charges : M = " + to_string(vMasses.at(p)) + " , <E*> = " + to_string(calib);
      hFragmentCharges_Mass_Cut->SetTitle(s);

      // TCanvas *c = new TCanvas("canvas","canvas",1000,1000);
      // hFragmentCharges_Mass_Cut->Draw("");
      //
      // TString sp = "/home/gabri/Escritorio/fission_charges_M_" + to_string(vMasses.at(p)) + ".png";
      // c->Print(sp);

      vExc_energy.push_back(calib);
      vFissions.push_back(hFragmentCharges_Mass_Cut->GetEntries());
      nFissions += hFragmentCharges_Mass_Cut->GetEntries();

      // s = "hCharges_" + to_string(vMasses.at(p));
      // hFragmentCharges_Mass_Cut->SetName(s);
      // hFragmentCharges_Mass_Cut->Write();
      //
      hFragmentCharges_Mass_Cut->Reset("ICESM");
      hExcitationEnergy->Reset("ICESM");

      // delete c;
   }

    // /* ------- Getting Fission Probabilities -------- */
    // TH1F *hSlice_histogram;
    // /*300, -200,500*/
    // vector<Double_t> vExcEnergies;
    // vector<Double_t> vProb;
    // int multiplier = 20;
    //
    // for (int i = 0 ; i < 15 ; i++){
    //
    //  hSlice_histogram = (TH1F*)hCorr_exc_energy_charges->ProjectionY("name",multiplier*i,multiplier*i+multiplier,"");
    //  vExcEnergies.push_back(hCorr_exc_energy_charges->GetXaxis()->GetBinCenter( (multiplier*i + multiplier*i+multiplier)/2.0) );
    //  vProb.push_back(hSlice_histogram->GetEntries()/hCorr_exc_energy_charges->GetEntries());
    // }
    //

     for(int p = 0 ; p < vFissions.size() ; p++)
      vFissions.at(p) = vFissions.at(p) / nFissions;

     TGraph *gProbability = new TGraph(vExc_energy.size(), &(vExc_energy.at(0)),&(vFissions.at(0)));
     gProbability->SetTitle("Fission Probability");
     gProbability->GetXaxis()->SetTitle("Excitation Energy");
     gProbability->GetYaxis()->SetTitle("Probability (Counts / Total Counts)");
    //
    // TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Kinematics Canvas");
    // kinematics_canvas->Divide(2,1);
    //
    // kinematics_canvas->cd(1);
    //  hCorr_raw_kinematics->Draw("COLZ");
    //
    // kinematics_canvas->cd(2);
    //  hCorr_reconstructed_kinematics->Draw("COLZ");
    //
    //
    // TCanvas *exc_energy_canvas = new TCanvas("exc_energy_canvas","exc_energy_canvas");
    // exc_energy_canvas->Divide(2,1);
    //
    // exc_energy_canvas->cd(1);
    // hRaw_exc_energy->Draw("");
    //
    // exc_energy_canvas->cd(2);
    // hReconstructed_exc_energy->Draw("");
    //
    // TCanvas *exc_charges_canvas = new TCanvas("exc_charges_canvas","exc_charges_canvas");
    // exc_charges_canvas->Divide(2,1);
    //
    // exc_charges_canvas->cd(1);
    // hCorr_exc_energy_charges->Draw("COLZ");
    //
    // exc_charges_canvas->cd(2);
    // hCorr_exc_energy_masses->Draw("COLZ");
    //
    TCanvas *myCanvas = new TCanvas("canvas_prob","canvas_prob",1000,800);
    // // myCanvas->Divide(2,1);
    // //
    myCanvas->cd();
    hCorr_charges_mass_sum->Draw("COLZ");
    //
    // myCanvas->cd();
    // hCorr_exc_energy_mass_sum->Draw("COLZ");
    //
    // TCanvas *cut_canvas = new TCanvas("cut_canvas","cut_canvas");
    // cut_canvas->Divide(2,1);
    //
    // cut_canvas->cd(1);
    // hFragmentCharges_Mass_Cut->Draw("");
    //
    // cut_canvas->cd(2);
    // hExcitationEnergy->Draw("");
    //
    //
    // cout<<"Mean Excitation Energy : "<<hExcitationEnergy->GetMean()<<endl;
    //
    // TCanvas *all_canvas = new TCanvas("all_canvas","all_canvas");
    // all_canvas->Divide(7,3);
    //
    // all_canvas->cd(1);
    // hFragmentMasses_1->Draw("");
    //
    // all_canvas->cd(2);
    // hFragmentMasses_2->Draw("");
    //
    // all_canvas->cd(3);
    // hFragmentMasses_3->Draw("");
    //
    // all_canvas->cd(4);
    // hFragmentMasses_4->Draw("");
    //
    // all_canvas->cd(5);
    // hFragmentMasses_5->Draw("");
    //
    // all_canvas->cd(6);
    // hFragmentMasses_6->Draw("");
    //
    // all_canvas->cd(7);
    // hFragmentMasses_7->Draw("");
    //
    // all_canvas->cd(8);
    // hMassSum_1->Draw("");
    //
    // all_canvas->cd(9);
    // hMassSum_2->Draw("");
    //
    // all_canvas->cd(10);
    // hMassSum_3->Draw("");
    //
    // all_canvas->cd(11);
    // hMassSum_4->Draw("");
    //
    // all_canvas->cd(12);
    // hMassSum_5->Draw("");
    //
    // all_canvas->cd(13);
    // hMassSum_6->Draw("");
    //
    // all_canvas->cd(14);
    // hMassSum_7->Draw("");
    //
    // all_canvas->cd(15);
    // hFragmentCharges_1->Draw("");
    //
    // all_canvas->cd(16);
    // hFragmentCharges_2->Draw("");
    //
    // all_canvas->cd(17);
    // hFragmentCharges_3->Draw("");
    //
    // all_canvas->cd(18);
    // hFragmentCharges_4->Draw("");
    //
    // all_canvas->cd(19);
    // hFragmentCharges_5->Draw("");
    //
    // all_canvas->cd(20);
    // hFragmentCharges_6->Draw("");
    //
    // all_canvas->cd(21);
    // hFragmentCharges_7->Draw("");


    theApp->Run();

    return 0;

}
