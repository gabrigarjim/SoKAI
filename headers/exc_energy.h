
#define UMA 931.494061
#define PROTONMASS 938.27208816
#define PROTACTINIUMMASS 237.05115 // In U.M.A
#define URANIUMMASS 238.0507882
#define BEAMENERGY 560

Float_t exc_energy(Float_t fEnergy1, Float_t fEnergy2, Float_t fTheta1, Float_t fTheta2, Float_t fPhi1, Float_t fPhi2)
{

      Float_t fP1,fP2,fPz1,fPz2,fPx1,fPx2,fPy1,fPy2;
      Float_t fInvariantMass;
      Float_t fExcitationEnergy;
      Float_t fTotalE1, fTotalE2;


           /*-------- For exc. energy -------- */
           fP1 = TMath::Sqrt((fEnergy1 + PROTONMASS)*(fEnergy1 + PROTONMASS) - PROTONMASS*PROTONMASS);
           fP2 = TMath::Sqrt((fEnergy2 + PROTONMASS)*(fEnergy2 + PROTONMASS) - PROTONMASS*PROTONMASS);

           fPz1 = fP1*TMath::Cos(TMath::DegToRad()*fTheta1);
           fPz2 = fP2*TMath::Cos(TMath::DegToRad()*fTheta2);

           if(fPhi1 < 0 )
            fPhi1 = 360 - TMath::Abs(fPhi1);

           if(fPhi2 < 0 )
            fPhi2 = 360 - TMath::Abs(fPhi2);

           fPx1 = fP1*TMath::Sin(TMath::DegToRad()*fTheta1)*TMath::Cos(TMath::DegToRad()*fPhi1);
           fPx2 = fP2*TMath::Sin(TMath::DegToRad()*fTheta2)*TMath::Cos(TMath::DegToRad()*fPhi2);

           fPy1 = fP1*TMath::Sin(TMath::DegToRad()*fTheta1)*TMath::Sin(TMath::DegToRad()*fPhi1);
           fPy2 = fP2*TMath::Sin(TMath::DegToRad()*fTheta2)*TMath::Sin(TMath::DegToRad()*fPhi2);

           fTotalE1 = fEnergy1+PROTONMASS;
           fTotalE2 = fEnergy2+PROTONMASS;

           fInvariantMass = TMath::Power(((BEAMENERGY*238+URANIUMMASS*UMA) + PROTONMASS - (fTotalE1+fTotalE2)),2) - ((fPx1+fPx2)*(fPx1+fPx2) + (fPy1+fPy2)*(fPy1+fPy2)) -
                            TMath::Power(TMath::Sqrt((BEAMENERGY*238+URANIUMMASS*UMA)*(BEAMENERGY*238+URANIUMMASS*UMA)-URANIUMMASS*UMA*URANIUMMASS*UMA)-(fPz1+fPz2),2);


           fInvariantMass = TMath::Sqrt(fInvariantMass);
           // cout<<"Invariant Mass : "<<fInvariantMass<<endl;

           fExcitationEnergy = fInvariantMass-PROTACTINIUMMASS*UMA;
           // cout<<"P1 : "<<fP1<<" fPx1 : "<<fPx1<<" fPy1 : "<<fPy1<<" fPz1 : "<<fPz1<<endl;
           // cout<<"P2 : "<<fP2<<" fPx2 : "<<fPx2<<" fPy2 : "<<fPy2<<" fPz2 : "<<fPz2<<endl;
           // cout<<"--------- End of Event -------- "<<endl;
           //
           // cout<<"Excitation Energy : "<<fExcitationEnergy<<endl;

           return fExcitationEnergy;
      }
