#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

using namespace std::chrono;

int main (int argc, char** argv) {

 FLAGS_alsologtostderr = 1;
 google::InitGoogleLogging("Quasifree Reconstruction Check");

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


 float fPolarMax = TMath::Pi();
 float fAzimuthalMax = 2*TMath::Pi();
 float fClusterEnergyMax = 400;
 float fSingleCrystalEnergyMax = 340;
 float fPrimEnergyMax = 700;

 SKColorScheme();

 TString fileList = "../SoKAI/macros/quasifree_reconstruction/files/U238_Quasifree_560AMeV_NN_chamber_train.root";

 TString crystalString = "../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

 TH2F *hCorr_crystal_distribution;

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

  Float_t rMotherCrystalEnergy[2];
  TBranch  *singleBranch = eventTree->GetBranch("MotherCrystalEnergy");
  singleBranch->SetAddress(rMotherCrystalEnergy);

  Int_t rPunched[4];
  TBranch  *punchedBranch = eventTree->GetBranch("Punched");
  punchedBranch->SetAddress(rPunched);

  Float_t rPrimaryEnergy[2];
  TBranch  *primBranch = eventTree->GetBranch("ProtonEnergy");
  primBranch->SetAddress(&rPrimaryEnergy);

  Int_t rMotherId[2];
  TBranch *idBranch = eventTree->GetBranch("MotherId");
  idBranch->SetAddress(&rMotherId);

  /* ------ Reconstruction Model ------ */

  SKLayer   *layer_1_reco = new SKLayer(6,"LeakyReLU");
  SKWeights *weights_12_reco = new SKWeights(6,15);
  SKWeights *gradients_12_reco = new SKWeights(6,15);

  SKLayer   *layer_2_reco = new SKLayer(15,"LeakyReLU");
  SKWeights *weights_23_reco = new SKWeights(15,15);
  SKWeights *gradients_23_reco = new SKWeights(15,15);

  SKLayer   *layer_3_reco = new SKLayer(15,"LeakyReLU");
  SKWeights *weights_34_reco = new SKWeights(15,1);
  SKWeights *gradients_34_reco = new SKWeights(15,1);

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

  model_reco->SetSummaryFile("test_reconstruction","12");

  model_reco->Init();


  model_reco->LoadWeights("weights_reconstruction_30.txt");

  Int_t nEvents = eventTree->GetEntries();

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;


  /*----------- A lot of Plots ----------- */
  TH2F * hCorr_raw_kinematics = new TH2F("hCorr_raw_kinematics","Raw Kinematics",300,20,80,300,0,650);
  TH2F * hCorr_reconstructed_kinematics = new TH2F("hCorr_reconstructed_kinematics","Reconstructed Kinematics",300,20,80,300,0,650);
  TH2F * hCorr_perfect_kinematics = new TH2F("hCorr_perfect_kinematics","Perfect Kinematics",300,20,80,300,0,650);

  TH1F * hResolution_300_350 = new TH1F("hResolution_300_350","Resolution: 300 - 350 MeV ",200,-400,400);
  TH1F * hResolution_350_400 = new TH1F("hResolution_350_400","Resolution: 350 - 400 MeV ",200,-400,400);
  TH1F * hResolution_400_450 = new TH1F("hResolution_400_450","Resolution: 400 - 450 MeV ",200,-400,400);
  TH1F * hResolution_450_500 = new TH1F("hResolution_450_500","Resolution: 450 - 500 MeV ",200,-400,400);
  TH1F * hResolution_500_550 = new TH1F("hResolution_500_550","Resolution: 500 - 550 MeV ",200,-400,400);
  TH1F * hResolution_550_600 = new TH1F("hResolution_550_600","Resolution: 550 - 600 MeV ",200,-400,400);



  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

      if(abs(rClusterEnergy[0] - rPrimaryEnergy[0]) > 25 && rPrimaryEnergy[0] > 300){

        data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
        data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
        data_instance.push_back(rPolar[0]/fPolarMax);

        data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
        data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
        data_instance.push_back(rPolar[1]/fPolarMax);

        data_sample.push_back(data_instance);

        output_vec = model_reco->Propagate(0);

        if ( rPrimaryEnergy[0] > 300 && rPrimaryEnergy[0] < 350)
         hResolution_300_350->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);

        if ( rPrimaryEnergy[0] > 350 && rPrimaryEnergy[0] < 400)
         hResolution_350_400->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);

        if ( rPrimaryEnergy[0] > 400 && rPrimaryEnergy[0] < 450)
         hResolution_400_450->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);

        if ( rPrimaryEnergy[0] > 450 && rPrimaryEnergy[0] < 500)
         hResolution_450_500->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);

        if ( rPrimaryEnergy[0] > 500 && rPrimaryEnergy[0] < 550)
         hResolution_500_550->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);

        if ( rPrimaryEnergy[0] > 550 && rPrimaryEnergy[0] < 600)
         hResolution_550_600->Fill(fPrimEnergyMax*output_vec.at(0)-rPrimaryEnergy[0]);


        sprintf(name, "distributionCrystalID_%i",rMotherId[0]);

        hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
        hCorr_crystal_distribution->GetRandom2(randPhi,randTheta);

        hCorr_raw_kinematics->Fill(randTheta,fClusterEnergyMax*data_instance.at(0));
        hCorr_reconstructed_kinematics->Fill(randTheta,fPrimEnergyMax*output_vec.at(0));
        hCorr_perfect_kinematics->Fill(randTheta,rPrimaryEnergy[0]);

        data_instance.clear();
        model_reco->Clear();
        output_vec.clear();
        data_sample.clear();


       }


      if(abs(rClusterEnergy[1] - rPrimaryEnergy[1]) > 25 && rPrimaryEnergy[1] > 300){

        data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
        data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);
        data_instance.push_back(rPolar[1]/fPolarMax);

        data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
        data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
        data_instance.push_back(rPolar[0]/fPolarMax);

        data_sample.push_back(data_instance);

        output_vec = model_reco->Propagate(0);

        if ( rPrimaryEnergy[1] > 300 && rPrimaryEnergy[1] < 350)
         hResolution_300_350->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        if ( rPrimaryEnergy[1] > 350 && rPrimaryEnergy[1] < 400)
         hResolution_350_400->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        if ( rPrimaryEnergy[1] > 400 && rPrimaryEnergy[1] < 450)
         hResolution_400_450->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        if ( rPrimaryEnergy[1] > 450 && rPrimaryEnergy[1] < 500)
         hResolution_450_500->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        if ( rPrimaryEnergy[1] > 500 && rPrimaryEnergy[1] < 550)
         hResolution_500_550->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        if ( rPrimaryEnergy[1] > 550 && rPrimaryEnergy[1] < 600)
         hResolution_550_600->Fill(fPrimEnergyMax*(output_vec.at(0))- rPrimaryEnergy[1]);

        sprintf(name, "distributionCrystalID_%i",rMotherId[1]);

        hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
        hCorr_crystal_distribution->GetRandom2(randPhi,randTheta);

        hCorr_raw_kinematics->Fill(randTheta,fClusterEnergyMax*data_instance.at(0));
        hCorr_reconstructed_kinematics->Fill(randTheta,fPrimEnergyMax*output_vec.at(0));
        hCorr_perfect_kinematics->Fill(randTheta,rPrimaryEnergy[1]);

        data_instance.clear();
        model_reco->Clear();
        data_sample.clear();
        output_vec.clear();


       }

       }




    TF1 *myGaussian  = new TF1("gaus1","gaus(0)",-300,300);

    hResolution_300_350->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 300 - 350 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/325.0;

    hResolution_350_400->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 350 - 400 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/375.0;

    hResolution_400_450->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 400 - 450 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/425.0;

    hResolution_450_500->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 450 - 500 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/475.0;

    hResolution_500_550->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 500 - 550 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/525.0;

    hResolution_550_600->Fit(myGaussian,"Q","",-300,300);
    LOG(INFO)<<"Resolution for 550 - 600 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/575.0;



  TCanvas *reso_canvas = new TCanvas("reso_canvas","Resolution Canvas");
   reso_canvas->Divide(3,2);

   reso_canvas->cd(1);
    hResolution_300_350->Draw("");

   reso_canvas->cd(2);
    hResolution_350_400->Draw("");

   reso_canvas->cd(3);
    hResolution_400_450->Draw("");

   reso_canvas->cd(4);
    hResolution_450_500->Draw("");

   reso_canvas->cd(5);
    hResolution_500_550->Draw("");

   reso_canvas->cd(6);
    hResolution_550_600->Draw("");


    TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Kinematics Canvas");
    kinematics_canvas->Divide(3,1);

    kinematics_canvas->cd(1);
     hCorr_raw_kinematics->Draw("COLZ");

    kinematics_canvas->cd(2);
     hCorr_reconstructed_kinematics->Draw("COLZ");

    kinematics_canvas->cd(3);
     hCorr_perfect_kinematics->Draw("COLZ");



theApp->Run();

return 0;

}
