#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"

using namespace std::chrono;

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("KnockoutReconstruction");

  TApplication* theApp = new TApplication("Reconstruction", 0, 0);

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
  vector<vector<double>> input_labels_shuffled;



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

  float fPolarMax = TMath::Pi()/2.0;
  float fAzimuthalMax = 2*TMath::Pi();
  float fClusterEnergyMax = 600;
  float fSingleCrystalEnergyMax = 360;
  float fNfMax = 210;
  float fNsMax = 240;
  float fAngularDeviationMax = 0.20;
  float fPrimEnergyMax = 1000;

  gROOT->SetBatch(kTRUE);
  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "../../SoKAI/macros/knockout_reconstruction/files/U238_Fission_560AMeV_NN_Z91_Final.root";

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

  std::vector<Float_t> myVec;
  TBranch  *neuBranch = eventTree->GetBranch("AngularDeviation");
  angBranch->SetAddress(&myVec);



  int nEvents = eventTree->GetEntries();

  if(nEvents < nSamples)
  LOG(FATAL)<<"More number of samples than avalaible!!!";


  int eventCounter = 0;
  int nSamples_0 = 0, nSamples_1 = 0, nSamples_2 = 0, nSamples_3 = 0,nSamples_4 = 0,nSamples_5 = 0, nSamples_6 = 0 , nSamples_7 = 0;


  while (eventCounter < nSamples){

    eventTree->GetEvent(eventCounter);

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    eventCounter++;


    Float_t fFirstSigma,fSecondSigma;

    fFirstSigma  = rPrimaryEnergy[0]/235;
    fSecondSigma = rPrimaryEnergy[1]/235;



      data_instance.push_back( rClusterEnergy[0] / fClusterEnergyMax);
      data_instance.push_back( rMotherCrystalEnergy[0] / fSingleCrystalEnergyMax);
      data_instance.push_back( rPolar[0] / fPolarMax);
      data_instance.push_back( rAngularDeviation[0] / fAngularDeviationMax);

      data_instance.push_back( rClusterEnergy[1] / fClusterEnergyMax);
      data_instance.push_back( rMotherCrystalEnergy[1] / fSingleCrystalEnergyMax);
      data_instance.push_back( rPolar[1] / fPolarMax);
      data_instance.push_back( rAngularDeviation[1] / fAngularDeviationMax);


      data_sample.push_back(data_instance);

      label_instance.push_back(rPrimaryEnergy[0]/fPrimEnergyMax);

      input_labels.push_back(label_instance);

      data_instance.clear();
      label_instance.clear();


      data_instance.push_back( rClusterEnergy[1] / fClusterEnergyMax);
      data_instance.push_back( rMotherCrystalEnergy[1] / fSingleCrystalEnergyMax);
      data_instance.push_back( rPolar[1] / fPolarMax);
      data_instance.push_back( rAngularDeviation[1] / fAngularDeviationMax);

      data_instance.push_back( rClusterEnergy[0] / fClusterEnergyMax);
      data_instance.push_back( rMotherCrystalEnergy[0] / fSingleCrystalEnergyMax);
      data_instance.push_back( rPolar[0] / fPolarMax);
      data_instance.push_back( rAngularDeviation[0] / fAngularDeviationMax);


      data_sample.push_back(data_instance);

      label_instance.push_back( rPrimaryEnergy[1] / fPrimEnergyMax);

      input_labels.push_back(label_instance);

      data_instance.clear();
      label_instance.clear();

   }




  LOG(INFO)<<"Shuffle events! ";

  std::vector<int> indices(data_sample.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  for( int index : indices){

    data_sample_shuffled.push_back(data_sample[index]);
    input_labels_shuffled.push_back(input_labels[index]);

  }

  cout<<"Sample sizes : "<<data_sample_shuffled.size()<<" times "<<data_sample_shuffled.at(0).size()<<endl;
  cout<<"Label sizes : "<<input_labels_shuffled.size()<<" times "<<input_labels_shuffled.at(0).size()<<endl;

  int nTrainingSize   = (7.0/10.0)*data_sample_shuffled.size();
  int nTestSize       = (3.0/10.0)*data_sample_shuffled.size();

  LOG(INFO)<<"Training Size : "<<nTrainingSize<<" events. Test size : "<<nTestSize<<endl;


  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(8,argv[6]);
  SKWeights *weights_12 = new SKWeights(8,stoi(argv[5]));
  SKWeights *gradients_12 = new SKWeights(8,stoi(argv[5]));
  SKWeights *firstMoment_12 = new SKWeights(8,stoi(argv[5]));
  SKWeights *secondMoment_12 = new SKWeights(8,stoi(argv[5]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[5]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *gradients_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[5]),1);
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[5]),1);


  SKLayer   *layer_3 = new SKLayer(1,argv[8]);



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
  model->SetSummaryFile("summary_knockout_regression_three_layers",argv[10]);

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

  model->SetInputSample(&data_sample_shuffled);
  model->SetInputLabels(&input_labels_shuffled);

  model->Init();
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction(argv[9]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<data_sample.size()<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[9];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[6]<<" "<<"8";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[5];
  LOG(INFO)<<"L3 : "<<argv[8]<<" "<<"1";

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

     LOG(INFO)<<" Quadratic Loss : "<<quadraticLoss/(10*nTrainingSize)<<" Absolute Loss : "<<absoluteLoss/(10*nTrainingSize)<<" Absolute Loss (MeV) : "<<fPrimEnergyMax*(absoluteLoss)/(10*nTrainingSize)<<". Epoch : "<<i+1;
     loss_vec.push_back(fPrimEnergyMax*(absoluteLoss)/(10*nTrainingSize));
     epoch_vec.push_back(i+1);

     quadraticLoss = 0.0;
     absoluteLoss  = 0.0;

   }

}

real_end = clock();

LOG(INFO)<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s";

/* --------- Testing the model --------- */
TH2F *hCorrReconstruction_energy = new TH2F("hCorrReconstruction_energy","Reconstructed Energy Difference Vs Energy",400,-400,400,400,0,600);
TH2F *hCorrReconstruction_results = new TH2F("hCorrReconstruction_results","Reconstructed Energy Vs Primary Energy",400,-600,800,400,0,600);

TH1F * hResolution_250_300 = new TH1F("hResolution_250_300","Resolution: 250 - 300 MeV ",100,-400,400);
TH1F * hResolution_300_350 = new TH1F("hResolution_300_350","Resolution: 300 - 350 MeV ",100,-400,400);
TH1F * hResolution_350_400 = new TH1F("hResolution_350_400","Resolution: 350 - 400 MeV ",100,-400,400);
TH1F * hResolution_400_450 = new TH1F("hResolution_400_450","Resolution: 400 - 450 MeV ",100,-400,400);
TH1F * hResolution_450_500 = new TH1F("hResolution_450_500","Resolution: 450 - 500 MeV ",100,-400,400);
TH1F * hResolution_500_550 = new TH1F("hResolution_500_550","Resolution: 500 - 550 MeV ",100,-400,400);
TH1F * hResolution_550_600 = new TH1F("hResolution_550_600","Resolution: 550 - 600 MeV ",100,-400,400);
TH1F * hResolution_600_650 = new TH1F("hResolution_600_650","Resolution: 600 - 650 MeV ",100,-400,400);


TH2F * hCorr_raw_kinematics = new TH2F("hCorr_raw_kinematics","Raw Kinematics",300,20,80,300,0,600);
TH2F * hCorr_reconstructed_kinematics = new TH2F("hCorr_reconstructed_kinematics","Reconstructed Kinematics",300,20,80,300,0,600);




  for (int j = 0 ; j < nTestSize ; j++){

    int sample_number =  nTrainingSize + nTestSize*gen.Rndm();

    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    Double_t fDestandarizedPrimaryEnergy = fPrimEnergyMax*input_labels_shuffled.at(sample_number).at(0);
    Double_t fDestandarizedOutput = fPrimEnergyMax*output_vec.at(0);
    Double_t fDestandarizedPolar = TMath::RadToDeg()*(fPolarMax*data_sample_shuffled.at(sample_number).at(2) );

    hCorr_raw_kinematics->Fill(fDestandarizedPolar,fClusterEnergyMax*data_sample_shuffled.at(sample_number).at(0) );

    hCorr_reconstructed_kinematics->Fill(fDestandarizedPolar, fDestandarizedOutput);


    hCorrReconstruction_results->Fill(fDestandarizedOutput,fDestandarizedPrimaryEnergy);

    hCorrReconstruction_energy->Fill(fDestandarizedOutput - fDestandarizedPrimaryEnergy,fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 250 && fDestandarizedPrimaryEnergy < 300)
     hResolution_250_300->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 300 && fDestandarizedPrimaryEnergy < 350)
     hResolution_300_350->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 350 && fDestandarizedPrimaryEnergy < 400)
     hResolution_350_400->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 400 && fDestandarizedPrimaryEnergy < 450)
     hResolution_400_450->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 450 && fDestandarizedPrimaryEnergy < 500)
     hResolution_450_500->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 500 && fDestandarizedPrimaryEnergy < 550)
     hResolution_500_550->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 550 && fDestandarizedPrimaryEnergy < 600)
     hResolution_550_600->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    if(fDestandarizedPrimaryEnergy > 600 && fDestandarizedPrimaryEnergy < 650)
     hResolution_600_650->Fill(fDestandarizedOutput-fDestandarizedPrimaryEnergy);

    model->Clear();


}

TF1 *myGaussian  = new TF1("gaus1","gaus(0)",-300,300);

hResolution_250_300->Fit(myGaussian,"Q","",-300,300);
LOG(INFO)<<"Resolution for 250 - 300 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/275.0;

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

hResolution_600_650->Fit(myGaussian,"Q","",-300,300);
LOG(INFO)<<"Resolution for 600 - 650 MeV (FWHM) : "<<235.48*myGaussian->GetParameter(2)/625.0;



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

string weight_filename = "model_weights_knockout_regression_three_layers_";
weight_filename = weight_filename + argv[10] + ".txt";

model->SaveWeights(weight_filename);

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

std::vector<Float_t> vSaveValues;

TCanvas *reso_canvas = new TCanvas("reso_canvas","Resolution Canvas");
 reso_canvas->Divide(4,2);

 reso_canvas->cd(1);
  hResolution_250_300->Draw("");
  vSaveValues.push_back(hResolution_250_300->GetMean());
  vSaveValues.push_back(hResolution_250_300->GetRMS());


 reso_canvas->cd(2);
  hResolution_300_350->Draw("");
  vSaveValues.push_back(hResolution_300_350->GetMean());
  vSaveValues.push_back(hResolution_300_350->GetRMS());

 reso_canvas->cd(3);
  hResolution_350_400->Draw("");
  vSaveValues.push_back(hResolution_350_400->GetMean());
  vSaveValues.push_back(hResolution_350_400->GetRMS());

 reso_canvas->cd(4);
  hResolution_400_450->Draw("");
  vSaveValues.push_back(hResolution_400_450->GetMean());
  vSaveValues.push_back(hResolution_400_450->GetRMS());

 reso_canvas->cd(5);
  hResolution_450_500->Draw("");
  vSaveValues.push_back(hResolution_450_500->GetMean());
  vSaveValues.push_back(hResolution_450_500->GetRMS());

 reso_canvas->cd(6);
  hResolution_500_550->Draw("");
  vSaveValues.push_back(hResolution_500_550->GetMean());
  vSaveValues.push_back(hResolution_500_550->GetRMS());

 reso_canvas->cd(7);
  hResolution_550_600->Draw("");
  vSaveValues.push_back(hResolution_550_600->GetMean());
  vSaveValues.push_back(hResolution_550_600->GetRMS());

 reso_canvas->cd(8);
  hResolution_600_650->Draw("");
  vSaveValues.push_back(hResolution_600_650->GetMean());
  vSaveValues.push_back(hResolution_600_650->GetRMS());

  TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Kinematics Canvas");
  kinematics_canvas->Divide(2,1);

  kinematics_canvas->cd(1);
   hCorr_raw_kinematics->Draw("COLZ");

  kinematics_canvas->cd(2);
   hCorr_reconstructed_kinematics->Draw("COLZ");


 std::ofstream outFile("results_three_layer_model.txt", std::ios::app);
 outFile<<argv[10]<<" ";

 for(int i = 0 ; i < vSaveValues.size() ; i++)
  outFile<<vSaveValues.at(i)<<" ";

 outFile<<endl;

 outFile.close();


 TString filename = "training_results_knockout_regression_three_layers_";
  filename = filename + argv[10] + ".root";

 TFile resultsFile(filename,"RECREATE");


  model_canvas->Write();
  summary_canvas->Write();
  reso_canvas->Write();
  kinematics_canvas->Write();

  resultsFile.Close();

  // theApp->Run();

  // theApp->Terminate();
  exit(0);
  return 0;

}
