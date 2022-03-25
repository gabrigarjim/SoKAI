void goFire() {

 TStyle *gStyle = new TStyle();

 gStyle->SetOptStat(0);
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

}
