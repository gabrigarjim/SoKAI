#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

using namespace std::chrono;

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("CosmicTracking");

  TApplication* theApp = new TApplication("CosmicTracking", 0, 0);

  clock_t start, end,real_start,real_end;

  LOG(INFO)<<"#============================================================#";
  LOG(INFO)<<"# Welcome to SoKAI (Some Kind of Artificial Intelligence) !! #";
  LOG(INFO)<<"#============================================================#";

  int seed            = 2022;
  int epochs          = stoi(argv[1]);
  int nSamples        = stoi(argv[2]);
  int nMiniBatchSize  = stoi(argv[4]);
  float fLearningRate = stoi(argv[3])/1000.;

  real_start = clock();

  /*---- Input Data (Use this format!) ---- */
  vector<vector<double>> data_sample;
  vector<vector<double>> input_labels;

  vector<vector<double>> data_sample_shuffled;


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

  float fPolarMax = TMath::Pi();
  float fAzimuthalMax = TMath::Pi();
  float fPositionXMax = 200;
  float fPositionZMax = 200.;

  float fEnergyMax = 100.;
  float fCrystalAzimuthalMax = 190.;
  float fCrystalPolarMax = 140.;

  int nMaxCrystals = 50;

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "../../SoKAI/macros/cosmic_tracking/files/cosmicTree.root";

  TFile *eventFile;
  TTree* eventTree;

  eventFile = TFile::Open(fileList);
  eventTree = (TTree*)eventFile->Get("evt");

  Float_t rCosmicPolar;
  TBranch  *polarBranch = eventTree->GetBranch("fTheta");
  polarBranch->SetAddress(&rCosmicPolar);

  Float_t rCosmicAzimuthal;
  TBranch  *aziBranch = eventTree->GetBranch("fPhi");
  aziBranch->SetAddress(&rCosmicAzimuthal);

  Float_t rCosmicX;
  TBranch  *xBranch = eventTree->GetBranch("fX");
  xBranch->SetAddress(&rCosmicX);

  Float_t rCosmicZ;
  TBranch  *zBranch = eventTree->GetBranch("fZ");
  zBranch->SetAddress(&rCosmicZ);

  std::vector<float> *vCrystalEnergy = 0;
  TBranch  *energyBranch = eventTree->GetBranch("CrystalEnergy");
  energyBranch->SetAddress(&vCrystalEnergy);

  std::vector<float> *vCrystalPhi = 0;
  TBranch  *cryPhiBranch = eventTree->GetBranch("CrystalPhi");
  cryPhiBranch->SetAddress(&vCrystalPhi);

  std::vector<float> *vCrystalTheta = 0;
  TBranch  *cryThetaBranch = eventTree->GetBranch("CrystalTheta");
  cryThetaBranch->SetAddress(&vCrystalTheta);


  int eventCounter = 0;

  int nEvents = eventTree->GetEntries();

  if(nEvents < nSamples)
  LOG(FATAL)<<"More number of samples than avalaible!!!";


  while (eventCounter < nSamples){


    eventTree->GetEvent(eventCounter);

    eventCounter++;

    if(vCrystalEnergy->size() > nMaxCrystals)
    continue;


    for(int i = 0 ; i < 3*nMaxCrystals; i++)
     data_instance.push_back(0.0);


    for(int j = 0 ; j < vCrystalEnergy->size() ; j++){

          data_instance.at(3*j) = vCrystalEnergy->at(j) / fEnergyMax;
          data_instance.at(3*j + 1) = vCrystalTheta->at(j) / fCrystalPolarMax;
          data_instance.at(3*j + 2) = vCrystalPhi->at(j) / fCrystalAzimuthalMax;

    }


    data_sample.push_back(data_instance);

    label_instance.push_back(rCosmicX/fPositionXMax);
    label_instance.push_back(rCosmicZ/fPositionZMax);
    label_instance.push_back(rCosmicPolar/fPolarMax);
    label_instance.push_back(rCosmicAzimuthal/fAzimuthalMax);

    input_labels.push_back(label_instance);

    data_instance.clear();
    label_instance.clear();


  }


  int nTrainingSize   = (7.0/10.0)*data_sample.size();
  int nTestSize       = (3.0/10.0)*data_sample.size();

  LOG(INFO)<<"Training Size : "<<nTrainingSize<<" events. Test size : "<<nTestSize<<endl;


  cout<<"Sample sizes : "<<data_sample.size()<<" times "<<data_sample.at(0).size()<<endl;
  cout<<"Label sizes : "<<input_labels.size()<<" times "<<input_labels.at(0).size()<<endl;

  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(3*nMaxCrystals,argv[6]);
  SKWeights *weights_12 = new SKWeights(3*nMaxCrystals,stoi(argv[5]));
  SKWeights *gradients_12 = new SKWeights(3*nMaxCrystals,stoi(argv[5]));
  SKWeights *firstMoment_12 = new SKWeights(3*nMaxCrystals,stoi(argv[5]));
  SKWeights *secondMoment_12 = new SKWeights(3*nMaxCrystals,stoi(argv[5]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[5]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[5]),4);
  SKWeights *gradients_23 = new SKWeights(stoi(argv[5]),4);
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[5]),4);
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[5]),4);


  SKLayer   *layer_3 = new SKLayer(4,argv[8]);


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
  model->SetSummaryFile("summary_cosmic_tracking",argv[10]);

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
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction(argv[9]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<data_sample.size()<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[9];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[6]<<" "<<"150";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[5];
  LOG(INFO)<<"L5 : "<<argv[8]<<" "<<"4";

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

     LOG(INFO)<<" Quadratic Loss : "<<quadraticLoss/(10*nTrainingSize)<<" Absolute Loss : "<<absoluteLoss/(10*nTrainingSize)<<" Epoch : "<<i+1;
     loss_vec.push_back((absoluteLoss)/(10*nTrainingSize));
     epoch_vec.push_back(i+1);

     quadraticLoss = 0.0;
     absoluteLoss  = 0.0;

   }

}

real_end = clock();

LOG(INFO)<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s";

/* --------- Testing the model --------- */

TH1F * hPolarResidues = new TH1F("hPolarResidues","Residues : Primary Polar - Reconstructed Polar",100,-200,200);
TH1F * hAzimuthalResidues = new TH1F("hAzimuthalResidues","Residues : Primary Azimuthal - Reconstructed Azimuthal",100,-200,200);
TH1F * hPositionXResidues = new TH1F("hPositionXResidues","Residues : Primary Position X - Reconstructed Position X",100,-200,200);
TH1F * hPositionZResidues = new TH1F("hPositionZResidues","Residues : Primary Position Z - Reconstructed Position Z",100,-200,200);





  for (int j = 0 ; j < nTestSize ; j++){

    int sample_number =  nTrainingSize + nTestSize*gen.Rndm();

    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    Float_t fPositionXLabel = fPositionXMax*input_labels.at(sample_number).at(0);
    Float_t fPositionZLabel = fPositionZMax*input_labels.at(sample_number).at(1);
    Float_t fPolarLabel = TMath::RadToDeg()*fPolarMax*input_labels.at(sample_number).at(2);
    Float_t fAzimuthalLabel = TMath::RadToDeg()*fAzimuthalMax*input_labels.at(sample_number).at(3);

    Float_t fPositionXOutput = fPositionXMax*output_vec.at(0);
    Float_t fPositionZOutput = fPositionZMax*output_vec.at(1);
    Float_t fPolarOutput = TMath::RadToDeg()*fPolarMax*output_vec.at(2);
    Float_t fAzimuthalOutput = TMath::RadToDeg()*fAzimuthalMax*output_vec.at(3);

    hPolarResidues->Fill(fPolarLabel - fPolarOutput);
    hAzimuthalResidues->Fill(fAzimuthalLabel - fAzimuthalOutput);
    hPositionXResidues->Fill(fPositionXLabel - fPositionXOutput);
    hPositionZResidues->Fill(fPositionZLabel - fPositionZOutput);

    model->Clear();

 }



/* --------- Plots and so on ....... ------*/


TCanvas *model_canvas = new TCanvas("model_canvas","Model");
model_canvas->Divide(2,2);

model_canvas->cd(1);
hPolarResidues->Draw("");
hPolarResidues->GetXaxis()->SetTitle("Primary Polar - Reconstructed Polar (Degrees)");
hPolarResidues->GetYaxis()->SetTitle("Counts");

model_canvas->cd(2);
hAzimuthalResidues->Draw("");
hAzimuthalResidues->GetXaxis()->SetTitle("Primary Azimuthal - Reconstructed Azimuthal (Degrees)");
hAzimuthalResidues->GetYaxis()->SetTitle("Counts");

model_canvas->cd(3);
hPositionXResidues->Draw("");
hPositionXResidues->GetXaxis()->SetTitle("Primary Position X - Reconstructed Position X (cm)");
hPositionXResidues->GetYaxis()->SetTitle("Counts");

model_canvas->cd(4);
hPositionZResidues->Draw("");
hPositionZResidues->GetXaxis()->SetTitle("Primary Position Z  - Reconstructed Position Z (cm)");
hPositionZResidues->GetYaxis()->SetTitle("Counts");


TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);


TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();

string weight_filename = "model_weights_cosmic_tracking_";
weight_filename = weight_filename + argv[10] + ".txt";

model->SaveWeights(weight_filename);

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");


TString filename = "training_results_cosmic_tracking_";
   filename = filename + argv[10] + ".root";

TFile resultsFile(filename,"RECREATE");


 model_canvas->Write();
 summary_canvas->Write();

theApp->Run();

return 0;



}
