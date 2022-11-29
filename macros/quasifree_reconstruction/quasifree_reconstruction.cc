#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

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
  int nMiniBatchSize  = stoi(argv[4]);;
  float fLearningRate = stoi(argv[3])/1000.;

  real_start = clock();

  /*---- Input Data (Use this format!) ---- */
  vector<vector<double>> data_sample;
  vector<vector<double>> input_labels;


  vector<double> data_instance;
  vector<double> label_instance;

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

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "/home/gabri/Analysis/s455/simulation/punch_through/files/punched_sorted.root";

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("data");

  Float_t rClusterEnergy;
  TBranch  *energyBranch = eventTree->GetBranch("PunchedEnergy");
  energyBranch->SetAddress(&rClusterEnergy);

  Float_t rPolar;
  TBranch  *polarBranch = eventTree->GetBranch("PunchedTheta");
  polarBranch->SetAddress(&rPolar);

  Float_t rAzimuthal;
  TBranch  *aziBranch = eventTree->GetBranch("PunchedPhi");
  aziBranch->SetAddress(&rAzimuthal);

  Float_t rMotherCrystalEnergy;
  TBranch  *singleBranch = eventTree->GetBranch("PunchedCrystalEnergy");
  singleBranch->SetAddress(&rMotherCrystalEnergy);

  Float_t rPrimaryEnergy;
  TBranch  *primBranch = eventTree->GetBranch("PrimaryEnergy");
  primBranch->SetAddress(&rPrimaryEnergy);

  int nEvents = eventTree->GetEntries();


  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nSamples; j++) {


    eventTree->GetEvent(j);
    if(!(j%100))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nSamples)<<" % ) "<<endl;

     if(TMath::Abs(700*rPrimaryEnergy - 400*rClusterEnergy) < 10 || 700*rPrimaryEnergy < 300 && 400*rClusterEnergy < 150)
     continue;

    data_instance.push_back(rClusterEnergy);

    data_instance.push_back(rMotherCrystalEnergy);

    // data_instance.push_back(rPolar);
    //
    // data_instance.push_back(rAzimuthal);

    label_instance.push_back(rPrimaryEnergy);


    data_sample.push_back(data_instance);
    input_labels.push_back(label_instance);

    data_instance.clear();
    label_instance.clear();

   }

  int nTrainingSize   = (5.0/10.0)*data_sample.size();
  int nTestSize       = (5.0/10.0)*data_sample.size();


  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(2,argv[5]);
  SKWeights *weights_12 = new SKWeights(2,stoi(argv[6]));
  SKWeights *gradients_12 = new SKWeights(2,stoi(argv[6]));

  SKLayer   *layer_2 = new SKLayer(stoi(argv[6]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));
  SKWeights *gradients_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));

  SKLayer   *layer_3 = new SKLayer(stoi(argv[8]),argv[9]);
  SKWeights *weights_34 = new SKWeights(stoi(argv[8]),1);
  SKWeights *gradients_34 = new SKWeights(stoi(argv[8]),1);

  SKLayer   *layer_4 = new SKLayer(1,argv[10]);



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

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<argv[2]<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[11];
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

  for (int j = 0 ; j < nTestSize ; j++){


    // Using only 3/10 of the dataset to test the network
    int sample_number = nTrainingSize + nTestSize*gen.Rndm();


    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    hCorrReconstruction_results->Fill(fPrimEnergyMax*(output_vec.at(0)),fPrimEnergyMax*input_labels.at(sample_number).at(0));

    hCorrReconstruction_energy->Fill(fPrimEnergyMax*(output_vec.at(0))-fPrimEnergyMax*input_labels.at(sample_number).at(0),fPrimEnergyMax*input_labels.at(sample_number).at(0));

    //
    // hCorrKinematics_califa->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fClusterEnergyMax*data_sample.at(sample_number).at(0));
    //
    // hCorrKinematics_reconstruction->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fPrimEnergyMax*(output_vec.at(0)));

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

model->SaveWeights("weights_reconstruction.txt");

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

// TCanvas *reconstruction_canvas = new TCanvas("reconstruction_canvas","Reconstruction");
//
// reconstruction_canvas->Divide(2,1);
//
// reconstruction_canvas->cd(1);
// hCorrKinematics_califa->Draw("COLZ");
// hCorrKinematics_califa->GetXaxis()->SetTitle("Polar Angle (degrees)");
// hCorrKinematics_califa->GetYaxis()->SetTitle("Energy (MeV)");
//
//
//
// reconstruction_canvas->cd(2);
// hCorrKinematics_reconstruction->Draw("COLZ");
// hCorrKinematics_reconstruction->GetXaxis()->SetTitle("Polar Angle (degrees)");
// hCorrKinematics_reconstruction->GetYaxis()->SetTitle("Energy (MeV)");

TString name = "training_results_regression_";
 name = name + argv[12] + ".root";

TFile resultsFile(name,"RECREATE");

 // reconstruction_canvas->Write();
 model_canvas->Write();
 summary_canvas->Write();


theApp->Run();

return 0;



}
