#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"
#include "exc_energy.h"

int main () {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("ExcitationEnergy");

  TApplication* theApp = new TApplication("ExcitationEnergy", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  /*---- Input Data (Use this format!) ---- */
  vector<vector<double>> data_sample;
  vector<double> data_instance;

  /*---- For training results ----*/
  vector<double> loss_vec;
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
  TString fileList = "/home/gabri/Analysis/s455/simulation/punch_through/files/nn_file.root";

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  Float_t rClusterEnergy[2];
  TBranch  *energyBranch = eventTree->GetBranch("ClusterEnergy");
  energyBranch->SetAddress(&rClusterEnergy);

  Float_t rPolar[2];
  TBranch  *polarBranch = eventTree->GetBranch("ClusterTheta");
  polarBranch->SetAddress(&rPolar);

  Float_t rAzimuthal[2];
  TBranch  *aziBranch = eventTree->GetBranch("ClusterPhi");
  aziBranch->SetAddress(&rAzimuthal);

  Float_t rMotherCrystalEnergy[2];
  TBranch  *singleBranch = eventTree->GetBranch("MotherCrystalEnergy");
  singleBranch->SetAddress(&rMotherCrystalEnergy);

  Float_t rPrimaryEnergy[2];
  TBranch  *primBranch = eventTree->GetBranch("ProtonEnergy");
  primBranch->SetAddress(&rPrimaryEnergy);

  Float_t fFinalEnergy_1, fFinalEnergy_2, fTheta_1, fTheta_2, fPhi_1, fPhi_2;
  Float_t fExc_energy_califa, fExc_energy_nn, fExc_energy_mix;


  // /*------- Classification Model -------*/
  // SKLayer   *layer_1_class = new SKLayer(6,"Linear");
  // SKWeights *weights_12_class = new SKWeights(6,11);
  // SKWeights *gradients_12_class = new SKWeights(6,11);
  //
  // SKLayer   *layer_2_class = new SKLayer(11,"Linear");
  // SKWeights *weights_23_class = new SKWeights(11,11);
  // SKWeights *gradients_23_class = new SKWeights(11,11);
  //
  // SKLayer   *layer_3_class = new SKLayer(11,"Sigmoid");
  // SKWeights *weights_34_class = new SKWeights(11,4);
  // SKWeights *gradients_34_class = new SKWeights(11,4);
  //
  // SKLayer   *layer_4_class = new SKLayer(4,"Sigmoid");
  //
  // weights_12_class->Init(seed);
  // gradients_12_class->InitGradients();
  //
  // weights_23_class->Init(seed);
  // gradients_23_class->InitGradients();
  //
  // weights_34_class->Init(seed);
  // gradients_34_class->InitGradients();
  //
  //
  // SKModel *model_class = new SKModel("Classification");
  //
  // model_class->AddLayer(layer_1_class);
  // model_class->AddWeights(weights_12_class);
  // model_class->AddGradients(gradients_12_class);
  //
  // model_class->AddLayer(layer_2_class);
  // model_class->AddWeights(weights_23_class);
  // model_class->AddGradients(gradients_23_class);
  //
  // model_class->AddLayer(layer_3_class);
  // model_class->AddWeights(weights_34_class);
  // model_class->AddGradients(gradients_34_class);
  //
  // model_class->AddLayer(layer_4_class);
  //
  // model_class->SetInputSample(&data_sample);
  //
  // model_class->Init();
  //
  // model_class->LoadWeights("model_weights_7.txt");

  /*------- Regression Model -------*/
  SKLayer   *layer_1_reg = new SKLayer(2,"Linear");
  SKWeights *weights_12_reg = new SKWeights(2,16);
  SKWeights *gradients_12_reg = new SKWeights(2,16);

  SKLayer   *layer_2_reg = new SKLayer(16,"Linear");
  SKWeights *weights_23_reg = new SKWeights(16,4);
  SKWeights *gradients_23_reg = new SKWeights(16,4);

  SKLayer   *layer_3_reg = new SKLayer(4,"Sigmoid");
  SKWeights *weights_34_reg = new SKWeights(4,1);
  SKWeights *gradients_34_reg = new SKWeights(4,1);

  SKLayer   *layer_4_reg = new SKLayer(1,"Linear");

  weights_12_reg->Init(seed);
  gradients_12_reg->InitGradients();

  weights_23_reg->Init(seed);
  gradients_23_reg->InitGradients();

  weights_34_reg->Init(seed);
  gradients_34_reg->InitGradients();


  SKModel *model_reg = new SKModel("Regression");

  model_reg->AddLayer(layer_1_reg);
  model_reg->AddWeights(weights_12_reg);
  model_reg->AddGradients(gradients_12_reg);

  model_reg->AddLayer(layer_2_reg);
  model_reg->AddWeights(weights_23_reg);
  model_reg->AddGradients(gradients_23_reg);

  model_reg->AddLayer(layer_3_reg);
  model_reg->AddWeights(weights_34_reg);
  model_reg->AddGradients(gradients_34_reg);

  model_reg->AddLayer(layer_4_reg);

  model_reg->SetInputSample(&data_sample);

  model_reg->Init();

  model_reg->LoadWeights("weights_reconstruction_use_this.txt");


/* --------- Testing the model --------- */
TH1F *hExcEnergy_califa = new TH1F("hExcEnergy_califa","Excitation Energy : Califa",400,-200,200);
TH1F *hExcEnergy_nn     = new TH1F("hExcEnergy_nn","Excitation Energy : NN ",400,-200,200);
TH1F *hExcEnergy_Mix_primaries      = new TH1F("hExcEnergy_Mix_primaries","Excitation Energy : Califa Angle + MC Energies ",400,-200,200);

TH2F *hCorrKinematics_califa = new TH2F("hCorrKinematics_califa","Kinematics",400,0,100,400,0,700);
TH2F *hCorrKinematics_reconstruction = new TH2F("hCorrKinematics_reconstruction","Kinematics",400,0,100,400,0,700);

  int nEvents = eventTree->GetEntries();


  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);
    if(!(j%100))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;


    fFinalEnergy_1 = rClusterEnergy[0];
    fFinalEnergy_2 = rClusterEnergy[1];

    if(rClusterEnergy[0] < 20 || rClusterEnergy[1] < 20 )
    continue;

    fTheta_1 = rPolar[0];
    fTheta_2 = rPolar[1];

    fPhi_1 = rAzimuthal[0];
    fPhi_2 = rAzimuthal[1];


    if(TMath::Abs(rPrimaryEnergy[0] - rClusterEnergy[0]) > 10){

     data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
     data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);

     data_sample.push_back(data_instance);

     output_vec = model_reg->Propagate(0);

     data_instance.clear();
     data_sample.clear();

     fFinalEnergy_1 = fPrimEnergyMax*output_vec.at(0);

     output_vec.clear();

     model_reg->Clear();

    }

   if(TMath::Abs(rPrimaryEnergy[1] - rClusterEnergy[1]) > 10){

    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);

    data_sample.push_back(data_instance);

    output_vec = model_reg->Propagate(0);

    data_instance.clear();
    data_sample.clear();

    fFinalEnergy_2 = fPrimEnergyMax*output_vec.at(0);

    output_vec.clear();

    model_reg->Clear();

   }

    fExc_energy_califa = exc_energy(rClusterEnergy[0],rClusterEnergy[1],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    fExc_energy_nn = exc_energy(fFinalEnergy_1,fFinalEnergy_2,TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);
    fExc_energy_mix = exc_energy(rPrimaryEnergy[0],rPrimaryEnergy[1],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]);

    hExcEnergy_nn->Fill(fExc_energy_nn);
    hExcEnergy_califa->Fill(fExc_energy_califa);
    hExcEnergy_Mix_primaries->Fill(fExc_energy_mix);

    hCorrKinematics_califa->Fill(TMath::RadToDeg()*rPolar[0],rClusterEnergy[0]);
    hCorrKinematics_califa->Fill(TMath::RadToDeg()*rPolar[1],rClusterEnergy[1]);

    hCorrKinematics_reconstruction->Fill(TMath::RadToDeg()*rPolar[0],fFinalEnergy_1);
    hCorrKinematics_reconstruction->Fill(TMath::RadToDeg()*rPolar[1],fFinalEnergy_2);
    cout<<"Reconstructed Energy : "<<fFinalEnergy_1<<" "<<fFinalEnergy_2<<" Original : "<<rPrimaryEnergy[0]<<" "<<rPrimaryEnergy[1]<<endl;

   }






/* --------- Plots and so on ....... ------*/


TCanvas *model_canvas = new TCanvas("model_canvas","Model");
model_canvas->Divide(3,1);

model_canvas->cd(1);
hExcEnergy_califa->Draw("");
hExcEnergy_califa->GetXaxis()->SetTitle("Excitation Energy (MeV)");
hExcEnergy_califa->GetYaxis()->SetTitle("Counts");

model_canvas->cd(2);
hExcEnergy_nn->Draw("");
hExcEnergy_nn->GetXaxis()->SetTitle("Excitation Energy (MeV)");
hExcEnergy_nn->GetYaxis()->SetTitle("Counts");

model_canvas->cd(3);
hExcEnergy_Mix_primaries->Draw("");
hExcEnergy_Mix_primaries->GetXaxis()->SetTitle("Excitation Energy (MeV)");
hExcEnergy_Mix_primaries->GetYaxis()->SetTitle("Counts");




TCanvas *reconstruction_canvas = new TCanvas("reconstruction_canvas","Reconstruction");

reconstruction_canvas->Divide(2,1);

reconstruction_canvas->cd(1);
hCorrKinematics_califa->Draw("COLZ");
hCorrKinematics_califa->GetXaxis()->SetTitle("Polar Angle (degrees)");
hCorrKinematics_califa->GetYaxis()->SetTitle("Energy (MeV)");



reconstruction_canvas->cd(2);
hCorrKinematics_reconstruction->Draw("COLZ");
hCorrKinematics_reconstruction->GetXaxis()->SetTitle("Polar Angle (degrees)");
hCorrKinematics_reconstruction->GetYaxis()->SetTitle("Energy (MeV)");


theApp->Run();

return 0;



}
