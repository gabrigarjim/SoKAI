void SKColorScheme() {

gStyle->SetOptStat(0);
 Double_t One[5]    = { 0.00 , 0.50 , 1.00 , 1.00 , 1.00};
 Double_t Two[5]    = { 0.00 , 0.10 , 0.00 , 0.40 , 0.85};
 Double_t Three[5]  = { 0.00 , 0.28 , 0.00 , 0.05 , 0.15};
 Double_t Length[5] = { 0.00 , 0.25 , 0.70 , 0.85 , 1.00};
 Int_t nb=99;

 TColor::CreateGradientColorTable(5,Length,One,Two,Three,nb);
 gStyle->SetNumberContours(256);



}
