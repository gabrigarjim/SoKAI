#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKFancyPlots.h"
#include "exc_energy.h"


int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("ProtonReconstruction");

  TApplication* theApp = new TApplication("reconstruction", 0, 0);


  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed            = 2022;
  int epochs          = stoi(argv[1]);
  int nSamples        = stoi(argv[2]);
  int nTrainingSize   = (7.0/10.0)*nSamples;
  int nTestSize       = (3.0/10.0)*nSamples;
  int nMiniBatchSize  = stoi(argv[4]);;
  float fLearningRate = stoi(argv[3])/100.;

  real_start = clock();

  /*---- Input Data (Use this format!) ---- */
  vector<vector<double>> data_sample;
  vector<vector<double>> input_labels;
  vector<vector<double>> azimut_sample;


  vector<double> data_instance;
  vector<double> label_instance;
  vector<double> azimuth_instance;

  /*---- For training results ----*/
  vector<double> loss_vec;
  vector<double> output_vec;
  vector<double> output_model;

  vector<double> x_vec;
  vector<double> target_vec;
  vector<double> epoch_vec;
  vector<double> reconstruction_resolution_vec;

  double quadraticLoss = 0.0;
  double absoluteLoss = 0.0;

  TRandom3 gen(seed);

  float fPolarMax = TMath::Pi();
  float fAzimuthalMax = 2*TMath::Pi();
  float fClusterEnergyMax = 400;
  float fSingleCrystalEnergyMax = 340;
  float fPrimEnergyMax = 700;



  /* -------- Put this on a header or something..... ----------*/
  gStyle->SetOptStat(1111);
  Double_t Red[5]    = { 0.06, 0.25, 0.50, 0.75, 1.0};
  Double_t Green[5]  = {0.01, 0.1, 0.15, 0.20, 0.8};
  Double_t Blue[5]   = { 0.00, 0.00, 0.00, 0.0, 0.0};
  Double_t Length[5] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
  Int_t nb=250;

  TColor::CreateGradientColorTable(5,Length,Red,Green,Blue,nb);
   gStyle->SetCanvasColor(1);
   gStyle->SetTitleFillColor(1);
   gStyle->SetStatColor(1);

   gStyle->SetFrameLineColor(0);
   gStyle->SetGridColor(0);
   gStyle->SetStatTextColor(0);
   gStyle->SetTitleTextColor(0);
   gStyle->SetLabelColor(0,"xyz");
   gStyle->SetTitleColor(0,"xyz");
   gStyle->SetAxisColor(0,"xyz");
   /* --------------------------------------------------------*/


   /* ------- Reading Root Data -------- */
  TString fileList = "/home/gabri/Analysis/s455/simulation/punch_through/files/pca_file_matched.root";

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
  TBranch  *primBranch = eventTree->GetBranch("PrimaryEnergy");
  primBranch->SetAddress(rPrimaryEnergy);

  int nEvents = eventTree->GetEntries();

  float fExcEnergyCalifa,fExcEnergyNN;

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nSamples; j++) {


    eventTree->GetEvent(j);
    if(!(j%100))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nSamples)<<" % ) "<<endl;

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);

    data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);

    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rPolar[1]/fPolarMax);

    // data_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    // data_instance.push_back(rAzimuthal[1]/fAzimuthalMax);
    azimuth_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    azimuth_instance.push_back(rAzimuthal[1]/fAzimuthalMax);



    label_instance.push_back(rPrimaryEnergy[0]/fPrimEnergyMax);
    label_instance.push_back(rPrimaryEnergy[1]/fPrimEnergyMax);


    data_sample.push_back(data_instance);
    input_labels.push_back(label_instance);
    azimut_sample.push_back(azimuth_instance);

    data_instance.clear();
    label_instance.clear();
    azimuth_instance.clear();
   }

  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(6,argv[5]);
  SKWeights *weights_12 = new SKWeights(6,stoi(argv[6]));
  SKWeights *gradients_12 = new SKWeights(6,stoi(argv[6]));

  SKLayer   *layer_2 = new SKLayer(stoi(argv[6]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));
  SKWeights *gradients_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));

  SKLayer   *layer_3 = new SKLayer(stoi(argv[8]),argv[9]);
  SKWeights *weights_34 = new SKWeights(stoi(argv[8]),2);
  SKWeights *gradients_34 = new SKWeights(stoi(argv[8]),2);

  SKLayer   *layer_4 = new SKLayer(2,argv[10]);



  weights_12->Init(seed);
  gradients_12->InitGradients();

  weights_23->Init(seed);
  gradients_23->InitGradients();

  weights_34->Init(seed);
  gradients_34->InitGradients();





  SKModel *model = new SKModel("Regression");

  model->AddLayer(layer_1);
  model->AddWeights(weights_12);
  model->AddGradients(gradients_12);

  model->AddLayer(layer_2);
  model->AddWeights(weights_23);
  model->AddGradients(gradients_23);

  model->AddLayer(layer_3);
  model->AddWeights(weights_34);
  model->AddGradients(gradients_34);

  model->AddLayer(layer_4);

  model->SetInputSample(&data_sample);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction(argv[11]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<argv[2]<<" Learning Rate : "<<stoi(argv[3])/100.0<<" Metric : "<<argv[11];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[5]<<" "<<"8";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[6];
  LOG(INFO)<<"H2 : "<<argv[9]<<" "<<argv[8];
  LOG(INFO)<<"L4 : "<<argv[10]<<" "<<"1";

  /* ---------- Pass Data Through Model ----------*/

   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < nTrainingSize ; j++){


      // Using  7/10 of the dataset to train the network
      int sample_number = nTrainingSize*gen.Rndm();

      model->Train(j);

    if(i%10 == 0 && j == nTrainingSize-1){
      absoluteLoss  =  model->AbsoluteLoss();
      quadraticLoss =  model->QuadraticLoss();

     }

      model->Clear();
   }

    if(i%10==0){

     LOG(INFO)<<" Quadratic Loss : "<<quadraticLoss<<" Absolute Loss : "<<absoluteLoss<<" . Epoch : "<<i;
     loss_vec.push_back(quadraticLoss);
     epoch_vec.push_back(i);

   }

}

real_end = clock();

cout<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s"<<endl;

/* --------- Testing the model --------- */
TH2F *hCorrReconstruction_energy = new TH2F("hCorrReconstruction_energy","Reconstructed Energy Difference Vs Energy",400,-400,400,400,0,600);
TH2F *hCorrReconstruction_results = new TH2F("hCorrReconstruction_results","Reconstructed Energy Vs Primary Energy",400,-600,800,400,0,600);

TH2F *hCorrKinematics_califa = new TH2F("hCorrKinematics_califa","Kinematics",400,0,100,400,0,700);
TH2F *hCorrKinematics_reconstruction = new TH2F("hCorrKinematics_reconstruction","Kinematics",400,0,100,400,0,700);

/* ----- Exc Energy ----- */

TH1F *hExcEnergy_califa = new TH1F("hExcEnergy_califa","Excitation Energy (Califa)",400,-200,200);
TH1F *hExcEnergy_network = new TH1F("hExcEnergy_network","Excitation Energy (Neural Network)",400,-200,200);

  for (int j = 0 ; j < nTestSize ; j++){


    // Using only 3/10 of the dataset to test the network
    int sample_number = nTrainingSize + nTestSize*gen.Rndm();


    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    hCorrReconstruction_results->Fill(fPrimEnergyMax*(output_vec.at(0)),fPrimEnergyMax*input_labels.at(sample_number).at(0));
    hCorrReconstruction_results->Fill(fPrimEnergyMax*(output_vec.at(1)),fPrimEnergyMax*input_labels.at(sample_number).at(1));

    hCorrReconstruction_energy->Fill(fPrimEnergyMax*(output_vec.at(0))-fPrimEnergyMax*input_labels.at(sample_number).at(0),fPrimEnergyMax*input_labels.at(sample_number).at(0));
    hCorrReconstruction_energy->Fill(fPrimEnergyMax*(output_vec.at(1))-fPrimEnergyMax*input_labels.at(sample_number).at(1),fPrimEnergyMax*input_labels.at(sample_number).at(1));


    hCorrKinematics_califa->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(4),fClusterEnergyMax*data_sample.at(sample_number).at(0));
    hCorrKinematics_califa->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(5),fClusterEnergyMax*data_sample.at(sample_number).at(1));

    hCorrKinematics_reconstruction->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(4),fPrimEnergyMax*(output_vec.at(0)));
    hCorrKinematics_reconstruction->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(5),fPrimEnergyMax*(output_vec.at(1)));


    fExcEnergyCalifa = exc_energy(fClusterEnergyMax*data_sample.at(sample_number).at(0),fClusterEnergyMax*data_sample.at(sample_number).at(1),TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(4),TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(5),TMath::RadToDeg()*fAzimuthalMax*azimut_sample.at(sample_number).at(0),TMath::RadToDeg()*fAzimuthalMax*azimut_sample.at(sample_number).at(1));
    fExcEnergyNN     = exc_energy(fPrimEnergyMax*(output_vec.at(0)),fPrimEnergyMax*(output_vec.at(1)),TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(4),TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(5),TMath::RadToDeg()*fAzimuthalMax*azimut_sample.at(sample_number).at(0),TMath::RadToDeg()*fAzimuthalMax*azimut_sample.at(sample_number).at(1));


    hExcEnergy_califa->Fill(fExcEnergyCalifa);
    hExcEnergy_network->Fill(fExcEnergyNN);


    model->Clear();


}





/* --------- Plots and so on ....... ------*/


TCanvas *model_canvas = new TCanvas("model_canvas","Model");
model_canvas->Divide(2,1);

model_canvas->cd(1);
hCorrReconstruction_results->Draw("COLZ");
hCorrReconstruction_results->GetXaxis()->SetTitle("Reconstructed Energy");
hCorrReconstruction_results->GetYaxis()->SetTitle("Primary Energy (MeV)");

model_canvas->cd(2);
hCorrReconstruction_energy->Draw("COLZ");
hCorrReconstruction_energy->GetXaxis()->SetTitle("Reconstructed Energy - Primary Energy (MeV)");
hCorrReconstruction_energy->GetYaxis()->SetTitle("Primary Energy (MeV)");



TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);


TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();


TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

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



TCanvas *excitation_energy_canvas = new TCanvas("excitation_energy_canvas","Reconstruction");
excitation_energy_canvas->Divide(2,1);

excitation_energy_canvas->cd(1);
hExcEnergy_califa->Draw();

excitation_energy_canvas->cd(2);
hExcEnergy_network->Draw();


TString name = "training_results_";
 name = name + argv[12] + ".root";

TFile resultsFile(name,"RECREATE");

 reconstruction_canvas->Write();
 excitation_energy_canvas->Write();
 model_canvas->Write();
 summary_canvas->Write();


theApp->Run();

return 0;



}
