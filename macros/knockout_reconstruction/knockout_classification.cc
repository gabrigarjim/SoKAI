#include "SKNeuron.h"
#include "SKLayer.h"
#include "SKWeights.h"
#include "SKPropagator.h"
#include "SKModel.h"
#include "SKFancyPlots.h"
#include "exc_energy.h"
#include "SKColorScheme.h"


int main (int argc, char** argv) {


  FLAGS_alsologtostderr = 1;
  google::InitGoogleLogging("KnockoutClassification");

  TApplication* theApp = new TApplication("Classification", 0, 0);


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

  vector<double> x_vec;
  vector<double> target_vec;
  vector<double> epoch_vec;

  double crossEntropyLoss = 0.0;

  TRandom3 gen(seed);

  float fPolarMax = TMath::Pi()/2.0;
  float fAzimuthalMax = 2*TMath::Pi();
  float fClusterEnergyMax = 400;
  float fSingleCrystalEnergyMax = 340;
  float fPrimEnergyMax = 700;
  float fNfMax = 200;
  float fNsMax = 230;
  float fAngularDeviationMax = 0.25;

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "../../SoKAI/macros/knockout_reconstruction/files/U238_Fission_560AMeV_NN_train.root";

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

  Int_t rPunched[4];
  TBranch  *punchedBranch = eventTree->GetBranch("Punched");
  punchedBranch->SetAddress(rPunched);


  int nEvents = eventTree->GetEntries();

  int fClass_0 = 0 , fClass_1 = 0, fClass_2 = 0, fClass_3 = 0;

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nSamples; j++) {


    eventTree->GetEvent(j);

    if(!(j%1000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

      if(TMath::RadToDeg()*(rPolar[0] + rPolar[1]) < 77 || TMath::RadToDeg()*(rPolar[0] + rPolar[1]) > 84)
       continue;

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

      label_instance.push_back(rPunched[0]);
      label_instance.push_back(rPunched[1]);
      label_instance.push_back(rPunched[2]);
      label_instance.push_back(rPunched[3]);


      data_sample.push_back(data_instance);
      input_labels.push_back(label_instance);

      data_instance.clear();
      label_instance.clear();

   }




   int nTrainingSize   = (7.0/10.0)*data_sample.size();
   int nTestSize       = (3.0/10.0)*data_sample.size();

   cout<<"Training Size : "<<nTrainingSize<<" Events "<<endl;
   cout<<"Test Size : "<<nTestSize<<" Events "<<endl;

  /*------- The Model Itself -------*/

  SKLayer   *layer_1 = new SKLayer(14,argv[7]);
  SKWeights *weights_12 = new SKWeights(14,stoi(argv[5]));
  SKWeights *gradients_12 = new SKWeights(14,stoi(argv[5]));
  SKWeights *firstMoment_12 = new SKWeights(14,stoi(argv[5]));
  SKWeights *secondMoment_12 = new SKWeights(14,stoi(argv[5]));


  SKLayer   *layer_2 = new SKLayer(stoi(argv[5]),argv[8]);
  SKWeights *weights_23 = new SKWeights(stoi(argv[5]),stoi(argv[6]));
  SKWeights *gradients_23 = new SKWeights(stoi(argv[5]),stoi(argv[6]));
  SKWeights *firstMoment_23 = new SKWeights(stoi(argv[5]),stoi(argv[6]));
  SKWeights *secondMoment_23 = new SKWeights(stoi(argv[5]),stoi(argv[6]));

  SKLayer   *layer_3 = new SKLayer(stoi(argv[6]),argv[9]);
  SKWeights *weights_34 = new SKWeights(stoi(argv[6]),4);
  SKWeights *gradients_34 = new SKWeights(stoi(argv[6]),4);
  SKWeights *firstMoment_34 = new SKWeights(stoi(argv[6]),4);
  SKWeights *secondMoment_34 = new SKWeights(stoi(argv[6]),4);

  SKLayer   *layer_4 = new SKLayer(4,argv[10]);

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


  SKModel *model = new SKModel("Classification");

  model->SetOptimizer("Adam");
  model->SetSummaryFile("summary_knockout_classification_",argv[12]);

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
  LOG(INFO)<<"L1 : "<<argv[7]<<" "<<"14";
  LOG(INFO)<<"H1 : "<<argv[8]<<" "<<argv[5];
  LOG(INFO)<<"H2 : "<<argv[9]<<" "<<argv[6];
  LOG(INFO)<<"L4 : "<<argv[10]<<" "<<"4";

  /* ---------- Pass Data Through Model ----------*/

   for (int i = 0 ; i < epochs ; i++){

     crossEntropyLoss = 0.0;

     for (int j = 0 ; j < nTrainingSize ; j++){


      // Using  7/10 of the dataset to train the network
      int sample_number = nTrainingSize*gen.Rndm();

      model->Train(sample_number);

      crossEntropyLoss  +=  model->CrossEntropyLoss();

      model->Clear();

     }

    if(i%10==0){

     LOG(INFO)<<" Cross Entropy Loss : "<<crossEntropyLoss/nTrainingSize<<" . Epoch : "<<i;
     loss_vec.push_back(crossEntropyLoss/nTrainingSize);
     epoch_vec.push_back(i);

   }

}

real_end = clock();

cout<<"Total training time : "<<((float) real_end - real_start)/CLOCKS_PER_SEC<<" s"<<endl;

Float_t fGoodClassification  = 0.0;


TH2F *hTraining_results = new TH2F("hTraining_results","Confussion Matrix",4,-0.5,3.5,4,-0.5,3.5);

TH2F *hPunched_kinematics = new TH2F("hPunched_kinematics","Punched Kinematics",400,0,100,400,0,600);
TH2F *hStopped_kinematics = new TH2F("hStopped_kinematics","Stopped Kinematics",400,0,100,400,0,600);

Float_t mConfussionMatrix[4][4]={0.0};
Float_t vTotalCases[4] = {0.0};


  for (int j = 0 ; j < nTestSize ; j++){


    // Using only 3/10 of the dataset to test the network
    int sample_number = nTrainingSize + nTestSize*gen.Rndm();


    output_vec.clear();
    output_vec = model->Propagate(sample_number);

    int highest_index_training = distance(output_vec.begin(),max_element(output_vec.begin(), output_vec.end()));
    int highest_index_label = distance(input_labels.at(sample_number).begin(),max_element(input_labels.at(sample_number).begin(), input_labels.at(sample_number).end()));

    mConfussionMatrix[highest_index_training][highest_index_label] += 1;

    if(highest_index_label == highest_index_training)
     fGoodClassification += 2.0;


    if(highest_index_label != highest_index_training){

    if((highest_index_label == 0 && highest_index_training == 1) || (highest_index_label == 0 && highest_index_training == 2))
     fGoodClassification++;

    if((highest_index_label == 3 && highest_index_training == 1) || (highest_index_label == 3 && highest_index_training == 2))
     fGoodClassification++;

    if((highest_index_label == 1 && highest_index_training == 0) || (highest_index_label == 1 && highest_index_training == 3))
     fGoodClassification++;

    if((highest_index_label == 2 && highest_index_training == 0) || (highest_index_label == 2 && highest_index_training == 3))
     fGoodClassification++;

    }


    hTraining_results->Fill(highest_index_training,highest_index_label);

    vTotalCases[highest_index_label] += 1.0;

    if(highest_index_training == 0){

      hStopped_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fClusterEnergyMax*data_sample.at(sample_number).at(0));
      hStopped_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(9),fClusterEnergyMax*data_sample.at(sample_number).at(7));

    }


    if(highest_index_training == 1){

      hStopped_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fClusterEnergyMax*data_sample.at(sample_number).at(0));
      hPunched_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(9),fClusterEnergyMax*data_sample.at(sample_number).at(7));

    }


    if(highest_index_training == 2){

      hPunched_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fClusterEnergyMax*data_sample.at(sample_number).at(0));
      hStopped_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(9),fClusterEnergyMax*data_sample.at(sample_number).at(7));

    }


    if(highest_index_training == 3){

      hPunched_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(2),fClusterEnergyMax*data_sample.at(sample_number).at(0));
      hPunched_kinematics->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(sample_number).at(9),fClusterEnergyMax*data_sample.at(sample_number).at(7));

    }


    model->Clear();


}


LOG(INFO)<<"Accuracy: "<<100*(mConfussionMatrix[0][0] + mConfussionMatrix[1][1] + mConfussionMatrix[2][2] + mConfussionMatrix[3][3])/nTestSize<<" %";
LOG(INFO)<<"Accuracy (pair): "<<100*(fGoodClassification)/(2.0*nTestSize)<<" %";

LOG(INFO)<<"Confussion Matrix : Rows trained, Columns labels ";
 for(int i = 0 ; i < 4 ; i ++){
  for(int j = 0 ; j < 4 ; j ++){
   cout<<mConfussionMatrix[i][j]<<"    ";
  }
   cout<<endl;

}


LOG(INFO)<<"Confussion Matrix (Porcentual): Rows trained, Columns labels ";
 for(int i = 0 ; i < 4 ; i ++){
  for(int j = 0 ; j < 4 ; j ++){
   cout<<100*mConfussionMatrix[i][j]/vTotalCases[j]<<"    ";
  }
   cout<<endl;

}


TGraph *loss_graph = new TGraph(epoch_vec.size(),&epoch_vec[0],&loss_vec[0]);


TH2F* model_histo;
model_histo = (TH2F*)model->ShowMe();


/* ------ Writing weights ------ */

string weight_filename = "model_weights_classification_knockout";
 weight_filename = weight_filename + argv[12] + ".txt";

model->SaveWeights(weight_filename);


TCanvas *summary_canvas = new TCanvas("summary_canvas","Model");
summary_canvas->Divide(2,1);


summary_canvas->cd(1);
 model_histo->Draw("COLZ");

summary_canvas->cd(2);
 loss_graph->Draw("AC");

TCanvas *results_canvas = new TCanvas("results_canvas","Model");
 results_canvas->cd();

 hTraining_results->Draw("COLZ");


TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Model");
kinematics_canvas->Divide(2,1);

kinematics_canvas->cd(1);
hPunched_kinematics->Draw("COLZ");

kinematics_canvas->cd(2);
hStopped_kinematics->Draw("COLZ");



TString name = "training_results_classification_knockout_";
 name = name + argv[12] + ".root";

TFile resultsFile(name,"RECREATE");

 summary_canvas->Write();
 kinematics_canvas->Write();
 results_canvas->Write();

theApp->Run();

return 0;



}
