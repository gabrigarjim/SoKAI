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
  float fSingleCrystalEnergyMax = 340;
  float fPrimEnergyMax = 800;
  float fNfMax = 200;
  float fNsMax = 230;
  float fAngularDeviationMax = 0.25;

 SKColorScheme();

 TString fileList = "../../SoKAI/macros/knockout_reconstruction/files/U238_Fission_560AMeV_NN_train.root";

 TString crystalString = "../../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

 TH2F *hCorr_crystal_distribution;

 double exc_energy_raw,exc_energy_reco,exc_energy_primary;

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

  Float_t rPrimaryEnergy[2];
  TBranch  *primBranch = eventTree->GetBranch("ProtonEnergy");
  primBranch->SetAddress(&rPrimaryEnergy);

  Float_t rMotherNf[2];
  TBranch  *nfBranch = eventTree->GetBranch("MotherCrystalNf");
  nfBranch->SetAddress(&rMotherNf);

  Float_t rMotherNs[2];
  TBranch  *nsBranch = eventTree->GetBranch("MotherCrystalNs");
  nsBranch->SetAddress(&rMotherNs);

  Float_t rAngularDeviation[2];
  TBranch  *angBranch = eventTree->GetBranch("AngularDeviation");
  angBranch->SetAddress(&rAngularDeviation);

  Int_t rMotherId[2];
  TBranch *idBranch = eventTree->GetBranch("MotherId");
  idBranch->SetAddress(&rMotherId);

  Float_t rProtonPolar[2];
  TBranch *polarPrimBranch = eventTree->GetBranch("ProtonPolar");
  polarPrimBranch->SetAddress(rProtonPolar);

  Float_t rProtonAzimuthal[2];
  TBranch *aziPrimBranch = eventTree->GetBranch("ProtonAzimuthal");
  aziPrimBranch->SetAddress(rProtonAzimuthal);





  /* ------ Reconstruction Model ------ */

  SKLayer   *layer_1_reco = new SKLayer(14,"LeakyReLU");
  SKWeights *weights_12_reco = new SKWeights(14,20);
  SKWeights *gradients_12_reco = new SKWeights(14,20);

  SKLayer   *layer_2_reco = new SKLayer(20,"LeakyReLU");
  SKWeights *weights_23_reco = new SKWeights(20,20);
  SKWeights *gradients_23_reco = new SKWeights(20,20);

  SKLayer   *layer_3_reco = new SKLayer(20,"LeakyReLU");
  SKWeights *weights_34_reco = new SKWeights(20,1);
  SKWeights *gradients_34_reco = new SKWeights(20,1);

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


  model_reco->LoadWeights("/home/gabri/Analysis/s455/nn_results/knockout_reconstruction/model_weights_knockout_regression_3.txt");

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

  TH1F * hRaw_exc_energy = new TH1F("hRaw_exc_energy","Raw E*",400,-300,300);
  TH1F * hReconstructed_exc_energy = new TH1F("hReconstructed_exc_energy","Reconstructed E*",400,-300,300);



  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    Float_t fFirstSigma,fSecondSigma;

    fFirstSigma  = rPrimaryEnergy[0]/235;
    fSecondSigma = rPrimaryEnergy[1]/235;

    Double_t finalEnergy_1 = rClusterEnergy[0];
    Double_t finalEnergy_2 = rClusterEnergy[1];

    exc_energy_raw = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    exc_energy_primary = exc_energy(rPrimaryEnergy[0],rPrimaryEnergy[1],TMath::RadToDeg()*rProtonPolar[0],TMath::RadToDeg()*rProtonPolar[1],TMath::RadToDeg()*rProtonAzimuthal[0],TMath::RadToDeg()*rProtonAzimuthal[1]);
    hRaw_exc_energy->Fill(exc_energy_raw-exc_energy_primary);

      if(abs(rClusterEnergy[0] - rPrimaryEnergy[0]) > 3.0*fFirstSigma){

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


      if(abs(rClusterEnergy[1] - rPrimaryEnergy[1]) > 3.0*fSecondSigma){

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

       if( (abs(rClusterEnergy[1] - rPrimaryEnergy[1]) > 3.0*fSecondSigma) || (abs(rClusterEnergy[0] - rPrimaryEnergy[0]) > 3.0*fFirstSigma) ){
         cout<<"Cluster Energies : "<<rClusterEnergy[0]<<" "<<rClusterEnergy[1]<<" Reconstructed: "<<finalEnergy_1<<" "<<finalEnergy_2<<endl;

         exc_energy_reco = exc_energy(finalEnergy_1,finalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
         hReconstructed_exc_energy->Fill(exc_energy_reco-exc_energy_primary);

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

    TCanvas *exc_energy_canvas = new TCanvas("exc_energy_canvas","exc_energy_canvas");
    exc_energy_canvas->Divide(2,1);

    exc_energy_canvas->cd(1);
    hRaw_exc_energy->Draw("");

    exc_energy_canvas->cd(2);
    hReconstructed_exc_energy->Draw("");



  theApp->Run();

  return 0;

}
