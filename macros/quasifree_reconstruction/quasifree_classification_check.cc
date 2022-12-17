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

  SKColorScheme();

   /* ------- Reading Root Data -------- */
  TString fileList = "/home/gabri/Analysis/s455/simulation/punch_through/writers/U238_Quasifree_560AMeV_NN_test.root";

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
  primBranch->SetAddress(&rPrimaryEnergy);



  Float_t  fExc_energy_nn, fExc_energy_mix;

  int nEvents = eventTree->GetEntries();

  TH1F *hExcEnergy_nn     = new TH1F("hExcEnergy_nn","Excitation Energy : NN ",400,-200,200);


  TH2F **hCorr_classified_kinematics;
  hCorr_classified_kinematics = new TH2F*[8];

  TH2F **hCorr_classified_kinematics_wrong;
  hCorr_classified_kinematics_wrong = new TH2F*[8];

  TH2F **hCorr_classified_kinematics_good;
  hCorr_classified_kinematics_good = new TH2F*[8];



  char name [100];

  for(int i = 0 ; i < 4 ; i++){

      sprintf(name, "hCorr_classified_kinematics_%i_0", i + 1);
      hCorr_classified_kinematics[2*i] = new TH2F(name,name,100,18,70,100,0,400);
      hCorr_classified_kinematics[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

      sprintf(name, "hCorr_classified_kinematics_%i_1", i + 1);
      hCorr_classified_kinematics[2*i+1] = new TH2F(name,name,100,18,70,100,0,400);
      hCorr_classified_kinematics[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
      hCorr_classified_kinematics[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

  }

for(int i = 0 ; i < 4 ; i++){

    sprintf(name, "hCorr_classified_kinematics_wrong_%i_0", i + 1);
    hCorr_classified_kinematics_wrong[2*i] = new TH2F(name,name,100,18,70,100,0,400);
    hCorr_classified_kinematics_wrong[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_wrong[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

    sprintf(name, "hCorr_classified_kinematics_wrong_%i_1", i + 1);
    hCorr_classified_kinematics_wrong[2*i+1] = new TH2F(name,name,100,18,70,100,0,400);
    hCorr_classified_kinematics_wrong[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_wrong[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

}

for(int i = 0 ; i < 4 ; i++){

    sprintf(name, "hCorr_classified_kinematics_good_%i_0", i + 1);
    hCorr_classified_kinematics_good[2*i] = new TH2F(name,name,100,18,70,100,0,400);
    hCorr_classified_kinematics_good[2*i]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_good[2*i]->GetYaxis()->SetTitle("Energy (MeV)");

    sprintf(name, "hCorr_classified_kinematics_good_%i_1", i + 1);
    hCorr_classified_kinematics_good[2*i+1] = new TH2F(name,name,100,18,70,100,0,400);
    hCorr_classified_kinematics_good[2*i+1]->GetXaxis()->SetTitle("Polar Angle (degrees)");
    hCorr_classified_kinematics_good[2*i+1]->GetYaxis()->SetTitle("Energy (MeV)");

}






  /* ------ Classification Model ------ */

  SKLayer   *layer_1_class = new SKLayer(8,"LeakyReLU");
  SKWeights *weights_12_class = new SKWeights(8,10);
  SKWeights *gradients_12_class = new SKWeights(8,10);


  SKLayer   *layer_2_class = new SKLayer(10,"Sigmoid");
  SKWeights *weights_23_class = new SKWeights(10,4);
  SKWeights *gradients_23_class = new SKWeights(10,4);


  SKLayer   *layer_3_class = new SKLayer(4,"LeakyReLU");

  weights_12_class->Init(seed);
  gradients_12_class->InitGradients();

  weights_12_class->Print();

  weights_23_class->Init(seed);

  gradients_23_class->InitGradients();

  SKModel *model_class = new SKModel("Classification");

  model_class->AddLayer(layer_1_class);
  model_class->AddWeights(weights_12_class);
  model_class->AddGradients(gradients_12_class);


  model_class->AddLayer(layer_2_class);
  model_class->AddWeights(weights_23_class);
  model_class->AddGradients(gradients_23_class);


  model_class->AddLayer(layer_3_class);

  model_class->SetInputSample(&data_sample);

  model_class->SetSummaryFile("test","53");

  model_class->Init();


  model_class->LoadWeights("model_weights_classification_53.txt");

  weights_12_class->Print();


  LOG(INFO)<<"Number of Samples : "<<nEvents<<endl;

  for (Int_t j = 0; j<nEvents; j++) {


    eventTree->GetEvent(j);

    if(!(j%10000))
     LOG(INFO)<<"Reading event "<<j<<" out of "<<nEvents<<" ("<<100.0*Float_t(j)/Float_t(nEvents)<<" % ) "<<endl;

    if(rClusterEnergy[0] < 30 || rClusterEnergy[1] < 30)
    continue;

    data_instance.push_back(rClusterEnergy[0]/fClusterEnergyMax);
    data_instance.push_back(rClusterEnergy[1]/fClusterEnergyMax);

    data_instance.push_back((rMotherCrystalEnergy[0])/fSingleCrystalEnergyMax);
    data_instance.push_back((rMotherCrystalEnergy[1])/fSingleCrystalEnergyMax);

    data_instance.push_back(rPolar[0]/fPolarMax);
    data_instance.push_back(rPolar[1]/fPolarMax);

    data_instance.push_back(rAzimuthal[0]/fAzimuthalMax);
    data_instance.push_back(rAzimuthal[1]/fAzimuthalMax);

    label_instance.push_back(rPunched[0]);
    label_instance.push_back(rPunched[1]);
    label_instance.push_back(rPunched[2]);
    label_instance.push_back(rPunched[3]);

    input_labels.push_back(label_instance);




    data_sample.push_back(data_instance);

    output_vec = model_class->Propagate(0);

    int highest_index_training = distance(output_vec.begin(),max_element(output_vec.begin(), output_vec.end()));
    int highest_index_label = distance(input_labels.at(0).begin(),max_element(input_labels.at(0).begin(), input_labels.at(0).end()));

    hCorr_classified_kinematics[2*highest_index_training]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(4),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
    hCorr_classified_kinematics[2*highest_index_training+1]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(5),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));



    hExcEnergy_nn->Fill(exc_energy(rPrimaryEnergy[0],rPrimaryEnergy[1],TMath::RadToDeg()*rPolar[0],TMath::RadToDeg()*rPolar[1],TMath::RadToDeg()*rAzimuthal[0],TMath::RadToDeg()*rAzimuthal[1]));


    if(highest_index_label == highest_index_training){

     hCorr_classified_kinematics_good[2*highest_index_training]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(4),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_good[2*highest_index_training+1]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(5),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));
   }


    if(highest_index_label != highest_index_training){

     hCorr_classified_kinematics_wrong[2*highest_index_training]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(4),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(0));
     hCorr_classified_kinematics_wrong[2*highest_index_training+1]->Fill(TMath::RadToDeg()*fPolarMax*data_sample.at(data_sample.size()-1).at(5),fClusterEnergyMax*data_sample.at(data_sample.size()-1).at(1));

   }





    data_instance.clear();
    label_instance.clear();

    model_class->Clear();

    data_sample.clear();
    input_labels.clear();
   }

/* --------- Plots and so on ....... ------*/


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

 TCanvas *excCanvas = new TCanvas("excCanvas","Exc Canvas");
 excCanvas->cd();
 hExcEnergy_nn->Draw();



theApp->Run();

return 0;



}
