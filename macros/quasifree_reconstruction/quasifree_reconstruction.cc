#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKColorScheme.h"
using namespace std::chrono;

int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("Quasifree Reconstruction");

  TApplication* theApp = new TApplication("reconstruction", 0, 0);


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


  vector<double> data_instance;
  vector<double> label_instance;

  vector<vector<double>> data_sample_reconstruction;


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
  TString fileList = "/home/gabri/Analysis/s455/simulation/punch_through/writers/U238_Quasifree_560AMeV_NN.root";

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


  int nEvents = eventTree->GetEntries();

  TH2F **hCorr_classified_kinematics;
  hCorr_classified_kinematics = new TH2F*[8];

  char name [100];

  for(int i = 0 ; i < 4 ; i++){

      sprintf(name, "hCorr_classified_kinematics_%i_0", i + 1);
      hCorr_classified_kinematics[2*i] = new TH2F(name,name,200,18,70,200,0,400);
      hCorr_classified_kinematics[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

      sprintf(name, "hCorr_classified_kinematics_%i_1", i + 1);
      hCorr_classified_kinematics[2*i+1] = new TH2F(name,name,200,18,70,200,0,400);
      hCorr_classified_kinematics[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

  }


  /* ------ Classification Model ------ */

  SKLayer   *layer_1_class = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_12_class = new SKWeights(8,10);
  SKWeights *gradients_12_class = new SKWeights(8,10);


  SKLayer   *layer_2_class = new SKLayer(10,"LeakyReLU");
  SKWeights *weights_23_class = new SKWeights(10,20);
  SKWeights *gradients_23_class = new SKWeights(10,20);


  SKLayer   *layer_3_class = new SKLayer(20,"LeakyReLU");
  SKWeights *weights_34_class = new SKWeights(20,4);
  SKWeights *gradients_34_class = new SKWeights(20,4);

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

  model_class->SetSummaryFile("test","30");

  model_class->Init();


  model_class->LoadWeights("/home/gabri/Analysis/s455/simulation/punch_through/analysis/nn_results/model_weights_classification_28.txt");


  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nSamples; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nSamples<<" ("<<100.0*Float_t(j)/Float_t(nSamples)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);

    data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);

    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rPolar[1]/fPolarMax);

    data_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    data_instance.push_back(rAzimuthal[1]/fAzimuthalMax);


    data_sample.push_back(data_instance);

    output_vec = model_class->Propagate(data_sample.size()-1);

    int highest_index_training = distance(output_vec.begin(),max_element(output_vec.begin(), output_vec.end()));

    hCorr_classified_kinematics[2*highest_index_training]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(4),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics[2*highest_index_training+1]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(5),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));


    if(highest_index_training == 1){

     data_instance.push_back(highest_index_training/4.0);
     data_sample_reconstruction.push_back(data_instance);

     label_instance.push_back(rPrimaryEnergy[1]/fPrimEnergyMax);

     input_labels.push_back(label_instance);


    }

    if(highest_index_training == 2){

     data_instance.push_back(highest_index_training/4.0);
     data_sample_reconstruction.push_back(data_instance);


     label_instance.push_back(rPrimaryEnergy[0]/fPrimEnergyMax);

     input_labels.push_back(label_instance);

    }

    if(highest_index_training == 3){

     data_instance.push_back(highest_index_training/4.0);

     data_sample_reconstruction.push_back(data_instance);

     label_instance.push_back(rPrimaryEnergy[0]/fPrimEnergyMax);
     input_labels.push_back(label_instance);


     label_instance.clear();
     data_sample_reconstruction.push_back(data_instance);

     label_instance.push_back(rPrimaryEnergy[1]/fPrimEnergyMax);
     input_labels.push_back(label_instance);


    }



    data_instance.clear();
    model_class->Clear();
    data_sample.clear();
    label_instance.clear();

   }




  int nTrainingSize   = (1.0/10.0)*data_sample_reconstruction.size();
  int nTestSize       = (9.0/10.0)*data_sample_reconstruction.size();



  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(9,argv[5]);
  SKWeights *weights_12 = new SKWeights(9,stoi(argv[6]));
  SKWeights *gradients_12 = new SKWeights(9,stoi(argv[6]));
  SKWeights *firstMoment_12 = new SKWeights(9,stoi(argv[6]));
  SKWeights *secondMoment_12 = new SKWeights(9,stoi(argv[6]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[6]),argv[7]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));
  SKWeights *gradients_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[6]),stoi(argv[8]));


  SKLayer   *layer_3 = new SKLayer(stoi(argv[8]),argv[9]);
  SKWeights *weights_34 = new SKWeights(stoi(argv[8]),1);
  SKWeights *gradients_34 = new SKWeights(stoi(argv[8]),1);
  SKWeights *firstMoment_34 = new SKWeights(stoi(argv[8]),1);
  SKWeights *secondMoment_34 = new SKWeights(stoi(argv[8]),1);

  SKLayer   *layer_4 = new SKLayer(1,argv[10]);

  weights_12->Init(seed);
  gradients_12->InitGradients();
  firstMoment_12->InitMoment();
  secondMoment_12->InitMoment();


  weights_23->Init(seed);
  gradients_23->InitGradients();
  firstMoment_23->InitMoment();
  secondMoment_23->InitMoment();


  weights_34->Init(seed);
  gradients_34->InitGradients();
  firstMoment_34->InitMoment();
  secondMoment_34->InitMoment();


  SKModel *model = new SKModel("Regression");

  model->SetOptimizer("Adam");
  model->SetSummaryFile("summary_test",argv[12]);

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
  model->AddWeights(weights_34);
  model->AddGradients(gradients_34);
  model->AddFirstMoments(firstMoment_34);
  model->AddSecondMoments(secondMoment_34);


  model->AddLayer(layer_4);

  model->SetInputSample(&data_sample_reconstruction);
  model->SetInputLabels(&input_labels);

  model->Init();
  model->SetLearningRate(fLearningRate);
  model->SetLossFunction(argv[11]);

  /* ---- Number of processed inputs before updating gradients ---- */
  model->SetBatchSize(nMiniBatchSize);

  LOG(INFO)<<"Model Training Hyper Parameters. Epochs : "<<argv[1]<<" Samples : "<<argv[2]<<" Learning Rate : "<<stoi(argv[3])/1000.0<<" Metric : "<<argv[11];
  LOG(INFO)<<"";
  LOG(INFO)<<"/* ---------- Model Structure -----------";
  LOG(INFO)<<"L1 : "<<argv[5]<<" "<<"9";
  LOG(INFO)<<"H1 : "<<argv[7]<<" "<<argv[6];
  LOG(INFO)<<"H2 : "<<argv[9]<<" "<<argv[8];
  LOG(INFO)<<"L4 : "<<argv[10]<<" "<<"1";

  /* ---------- Pass Data Through Model ----------*/

   for (int i = 0 ; i < epochs ; i++){
     for (int j = 0 ; j < nTrainingSize ; j++){


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

LOG(INFO)<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s";

/* --------- Testing the model --------- */
TH2F *hCorrReconstruction_energy = new TH2F("hCorrReconstruction_energy","Reconstructed Energy Difference Vs Energy",400,-400,400,400,0,600);
TH2F *hCorrReconstruction_results = new TH2F("hCorrReconstruction_results","Reconstructed Energy Vs Primary Energy",400,-600,800,400,0,600);

  for (int j = 0 ; j < nTestSize ; j++){


    // Using only 3/10 of the dataset to test the network
    int sample_number = nTrainingSize + nTestSize*gen.Rndm();


    output_vec.clear();

    output_vec = model->Propagate(sample_number);

    hCorrReconstruction_results->Fill(fPrimEnergyMax*(output_vec.at(0)),fPrimEnergyMax*input_labels.at(sample_number).at(0));

    hCorrReconstruction_energy->Fill(fPrimEnergyMax*(output_vec.at(0))-fPrimEnergyMax*input_labels.at(sample_number).at(0),fPrimEnergyMax*input_labels.at(sample_number).at(0));

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

model->SaveWeights("weights_reconstruction_30.txt");

TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");


TString filename = "training_results_regression_";
 filename = filename + argv[12] + ".root";

TFile resultsFile(filename,"RECREATE");


 model_canvas->Write();
 summary_canvas->Write();




  TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Model");
  kinematics_canvas->Divide(4,2);

  for(int h = 0 ; h < 4 ; h ++){

     kinematics_canvas->cd(h+1);

     hCorr_classified_kinematics[2*h]->Draw("COLZ");

     kinematics_canvas->cd(4 + h + 1);

     hCorr_classified_kinematics[2*h + 1]->Draw("COLZ");


 }

theApp->Run();

return 0;



}
