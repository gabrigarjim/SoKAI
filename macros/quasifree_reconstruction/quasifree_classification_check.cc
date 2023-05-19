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
  google::InitGoogleLogging("Quasifree Reconstruction");

  TApplication* theApp = new TApplication("Classification_check", 0, 0);


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
  float fExcMax = 100;

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "../../SoKAI/macros/quasifree_reconstruction/files/U238_Quasifree_560AMeV_NN_chamber_train.root";
  TString crystalString = "../../SoKAI/macros/quasifree_reconstruction/files/angular_histograms.root";

  TH2F *hCorr_crystal_distribution;

  TFile *crystalFile;
  crystalFile = TFile::Open(crystalString);

  char name[100];
  Double_t randTheta_1,randPhi_1,randTheta_2,randPhi_2;

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

  Int_t rPunched[4];
  TBranch  *punchedBranch = eventTree->GetBranch("Punched");
  punchedBranch->SetAddress(rPunched);

  Float_t rPrimaryEnergy[2];
  TBranch  *primBranch = eventTree->GetBranch("ProtonEnergy");
  primBranch->SetAddress(rPrimaryEnergy);

  Int_t rMotherId[2];
  TBranch *idBranch = eventTree->GetBranch("MotherId");
  idBranch->SetAddress(rMotherId);

  Float_t rExcitationEnergy;
  TBranch  *excBranch = eventTree->GetBranch("ExcitationEnergy");
  excBranch->SetAddress(&rExcitationEnergy);




  Float_t  fExc_energy_nn, fExc_energy_mix;

  int nEvents = eventTree->GetEntries();

  TH2F **hCorr_classified_kinematics;
  hCorr_classified_kinematics = new TH2F*[8];

  TH2F **hCorr_classified_kinematics_wrong;
  hCorr_classified_kinematics_wrong = new TH2F*[8];

  TH2F **hCorr_classified_kinematics_good;
  hCorr_classified_kinematics_good = new TH2F*[8];




  for(int i = 0 ; i < 4 ; i++){

      sprintf(name, "hCorr_classified_kinematics_%i_0", i + 1);
      hCorr_classified_kinematics[2*i] = new TH2F(name,name,250,18,70,200,0,400);
      hCorr_classified_kinematics[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

      sprintf(name, "hCorr_classified_kinematics_%i_1", i + 1);
      hCorr_classified_kinematics[2*i+1] = new TH2F(name,name,250,18,70,200,0,400);
      hCorr_classified_kinematics[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

  }

for(int i = 0 ; i < 4 ; i++){

    sprintf(name, "hCorr_classified_kinematics_true_wrong_%i_0", i + 1);
    hCorr_classified_kinematics_wrong[2*i] = new TH2F(name,name,250,18,70,200,0,400);
    hCorr_classified_kinematics_wrong[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_wrong[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

    sprintf(name, "hCorr_classified_kinematics_true_wrong_%i_1", i + 1);
    hCorr_classified_kinematics_wrong[2*i+1] = new TH2F(name,name,250,18,70,200,0,400);
    hCorr_classified_kinematics_wrong[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_wrong[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

}

for(int i = 0 ; i < 4 ; i++){

    sprintf(name, "hCorr_classified_kinematics_true_good_%i_0", i + 1);
    hCorr_classified_kinematics_good[2*i] = new TH2F(name,name,250,18,70,200,0,400);
    hCorr_classified_kinematics_good[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_good[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

    sprintf(name, "hCorr_classified_kinematics_true_good_%i_1", i + 1);
    hCorr_classified_kinematics_good[2*i+1] = new TH2F(name,name,250,18,70,200,0,400);
    hCorr_classified_kinematics_good[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_good[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

}





  /* ------ Classification Model ------ */

  SKLayer   *layer_1_class = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_12_class = new SKWeights(8,8);
  SKWeights *gradients_12_class = new SKWeights(8,8);


  SKLayer   *layer_2_class = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_23_class = new SKWeights(8,8);
  SKWeights *gradients_23_class = new SKWeights(8,8);

  SKLayer   *layer_3_class = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_34_class = new SKWeights(8,4);
  SKWeights *gradients_34_class = new SKWeights(8,4);


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

  model_class->SetSummaryFile("test","1");

  model_class->Init();


  model_class->LoadWeights("model_weights_classification_knockout_1.txt");

  Float_t fGoodClassification=0.0, fGoodClassificationCut=0.0;

  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%1000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rAzimuthal[0]/fPolarMax);
    data_instance.push_back(rMotherCrystalEnergy[0]/fSingleCrystalEnergyMax);

    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);
    data_instance.push_back(rPolar[1]/fPolarMax);
    data_instance.push_back(rAzimuthal[1]/fPolarMax);
    data_instance.push_back(rMotherCrystalEnergy[1]/fSingleCrystalEnergyMax);


    label_instance.push_back(rPunched[0]);
    label_instance.push_back(rPunched[1]);
    label_instance.push_back(rPunched[2]);
    label_instance.push_back(rPunched[3]);

    input_labels.push_back(label_instance);


    sprintf(name, "distributionCrystalID_%i",rMotherId[0]);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi_1,randTheta_1);

    sprintf(name, "distributionCrystalID_%i",(int)rMotherId[1]);

    hCorr_crystal_distribution = (TH2F*)crystalFile->Get(name);
    hCorr_crystal_distribution->GetRandom2(randPhi_2,randTheta_2);


    data_sample.push_back(data_instance);

    output_vec = model_class->Propagate(0);

    int highest_index_training = distance(output_vec.begin(),max_element(output_vec.begin(), output_vec.end()));
    int highest_index_label = distance(input_labels.at(0).begin(),max_element(input_labels.at(0).begin(), input_labels.at(0).end()));

    /*---- The gross cut ---- */
    bool bPunchOne = (TMath::RadToDeg()*rPolar[0]) < 48.0 ;
    bool bPunchTwo = (TMath::RadToDeg()*rPolar[1]) < 48.0 ;

    if(bPunchOne && bPunchTwo) {
     if(highest_index_label == 3)
      fGoodClassificationCut += 2.0;

     if(highest_index_label == 1 || highest_index_label == 2)
      fGoodClassificationCut++;

    }

    if(!bPunchOne && !bPunchTwo) {
     if(highest_index_label == 0)
      fGoodClassificationCut += 2.0;

     if(highest_index_label == 1 || highest_index_label == 2)
      fGoodClassificationCut++;


    }

    if(!bPunchOne && bPunchTwo) {
     if(highest_index_label == 1)
      fGoodClassificationCut += 2.0;

     if(highest_index_label == 0 || highest_index_label == 3)
      fGoodClassificationCut++;


    }

    if(bPunchOne && !bPunchTwo) {
     if(highest_index_label == 2)
      fGoodClassificationCut += 2.0;

    if(highest_index_label == 0 || highest_index_label == 3)
     fGoodClassificationCut++;

    }


    hCorr_classified_kinematics[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));


    if(highest_index_label == highest_index_training){

     hCorr_classified_kinematics_good[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
     fGoodClassification += 2.0;
   }


    if(highest_index_label != highest_index_training){


      /*------ Class One ------ */
      if(highest_index_label == 0 && highest_index_training == 1){
       hCorr_classified_kinematics_good[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
       hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
       fGoodClassification++;
      }


     if(highest_index_label == 0 && highest_index_training == 2){
      hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
      hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
      fGoodClassification++;

     }

     if(highest_index_label == 0 && highest_index_training == 3){
      hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
      hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
     }



     /* ------ Class Two ------ */
    if(highest_index_label == 1 && highest_index_training == 0){
     hCorr_classified_kinematics_good[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
     fGoodClassification++;

    }


    if(highest_index_label == 1 && highest_index_training == 2){
     hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));

    }


    if(highest_index_label == 1 && highest_index_training == 3){
     hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
     fGoodClassification++;

    }


    /* ------ Class Three ------ */
   if(highest_index_label == 2 && highest_index_training == 0){
    hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
    fGoodClassification++;

   }


   if(highest_index_label == 2 && highest_index_training == 1){
    hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
   }


   if(highest_index_label == 2 && highest_index_training == 3){
    hCorr_classified_kinematics_good[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
    fGoodClassification++;

   }


   /* ----- Class Four ----- */

   if(highest_index_label == 3 && highest_index_training == 0){
    hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
   }


   if(highest_index_label == 3 && highest_index_training == 1){
    hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
    fGoodClassification++;

   }


   if(highest_index_label == 3 && highest_index_training == 2){
    hCorr_classified_kinematics_good[2*highest_index_training]->Fill(randTheta_1,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(randTheta_2,fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(4));
    fGoodClassification++;

  }


}

    data_instance.clear();
    label_instance.clear();

    model_class->Clear();

    data_sample.clear();
    input_labels.clear();
   }

/* --------- Plots and so on ....... ------*/
LOG(INFO)<<"Accuracy (pair): "<<100*(fGoodClassification)/(2.0*nEvents)<<" %";
LOG(INFO)<<"Accuracy Cut (pair): "<<100*(fGoodClassificationCut)/(2.0*nEvents)<<" %";



  TCanvas *kinematics_canvas = new TCanvas("kinematics_canvas","Model");
  kinematics_canvas->Divide(4,2);

  for(int h = 0 ; h < 4 ; h ++){

     kinematics_canvas->cd(h+1);

     hCorr_classified_kinematics[2*h]->Draw("COLZ");

     kinematics_canvas->cd(4 + h + 1);

     hCorr_classified_kinematics[2*h + 1]->Draw("COLZ");


 }


  TCanvas *kinematics_canvas_wrong = new TCanvas("kinematics_canvas_wrong","Model");
  kinematics_canvas_wrong->Divide(4,2);

  for(int h = 0 ; h < 4 ; h ++){

     kinematics_canvas_wrong->cd(h+1);

     hCorr_classified_kinematics_wrong[2*h]->Draw("COLZ");

     kinematics_canvas_wrong->cd(4 + h + 1);

     hCorr_classified_kinematics_wrong[2*h + 1]->Draw("COLZ");


 }

  TCanvas *kinematics_canvas_good = new TCanvas("kinematics_canvas_good","Model");
  kinematics_canvas_good->Divide(4,2);

  for(int h = 0 ; h < 4 ; h ++){

     kinematics_canvas_good->cd(h+1);

     hCorr_classified_kinematics_good[2*h]->Draw("COLZ");

     kinematics_canvas_good->cd(4 + h + 1);

     hCorr_classified_kinematics_good[2*h + 1]->Draw("COLZ");


 }


theApp->Run();

return 0;



}
