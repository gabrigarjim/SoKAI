#include "TF1.h"

//TGraph *geof,*gy,*gyp,*gdzl;

Double_t eofx_prog(Double_t *x, Double_t *par)
{
  Float_t xx=x[0];
  Double_t f=0;
  Double_t f1=0,g56=0,g50=0;

  if(xx<par[0]*0.5)
    {
      f1=par[1]*((par[0]-xx)/xx-1);
      g56=par[2]*exp(-0.5*pow((xx+par[3]-par[0])/par[4],2.));
      g50=pow(-1,par[0])*par[5]*exp(-0.5*pow((xx+par[6]-par[0])/par[7],2.));
    }
  else
    {
      f1=par[1]*pow(-1,par[0])*(xx/(par[0]-xx)-1);
      g56=pow(-1,par[0])*par[2]*exp(-0.5*pow((xx-par[3])/par[4],2.));
      g50=par[5]*exp(-0.5*pow((xx-par[6])/par[7],2.));
    }
  if(pow(-1,par[0])>0)
    f=f1+g56+g50+par[8];
  else
    f=f1+g56+g50;

  if(f>1)
    f=1;
  else if(f<-1)
    f=-1;
  
  return f;
}

TF1 *f_eofx=new TF1("f_eofx",eofx_prog,0,80,9);

void eofx_generator_ia(vector<double>& yields_v,vector<double>& deltas_v)
{  
  gRandom->SetSeed(0);
  TF1 *g1=new TF1("g1","[0]*exp(-0.5*pow((x-[1])/[2],2.))",20,80);
  TF1 *g2l=new TF1("g2l","[0]*exp(-0.5*pow((x-[1])/[2],2.))",20,80);
  TF1 *g3l=new TF1("g3l","[0]*exp(-0.5*pow((x-[1])/[2],2.))",20,80);
  TF1 *g2h=new TF1("g2h","[0]*exp(-0.5*pow((x-[1])/[2],2.))",20,80);
  TF1 *g3h=new TF1("g3h","[0]*exp(-0.5*pow((x-[1])/[2],2.))",20,80);
  
  //for(int itry=0;itry<1;itry++)
    {
      
      Float_t amp,mean,width,eopar1,eopar2,eopar3,eopar4,eopar5,eopar6,eopar7,eopar8;
      Float_t zv[80]={0};
      Float_t ypv[80]={0};
      Float_t yv[80]={0};
      Float_t dzlv[80]={0};
      Float_t dd[80]={0};
      Int_t Zfis;
      //Double_t eopars[9]={0};
      
      Zfis=(int)(gRandom->Rndm()*10.+85.);
      eopar1=gRandom->Rndm()*0.5;
      eopar2=gRandom->Rndm()*0.5;
      eopar3=54.+gRandom->Rndm()*4.;
      eopar4=2.+gRandom->Rndm()*4.;
      eopar5=gRandom->Rndm()*0.5;
      eopar6=48+gRandom->Rndm()*4.;
      eopar7=gRandom->Rndm();
      eopar8=gRandom->Rndm()*0.1;


      /*eopars[0]=Zfis;
      eopars[1]=eopar1;
      eopars[2]=eopar2;
      eopars[3]=eopar3;
      eopars[4]=eopar4;
      eopars[5]=eopar5;
      eopars[6]=eopar6;
      eopars[7]=eopar7;
      eopars[8]=eopar8;*/
      
      f_eofx->SetParameters(Zfis,eopar1,eopar2,eopar3,eopar4,eopar5,eopar6,eopar7,eopar8);
      
      TGraph *geof=new TGraph();
      
      //simulating our deltas:
      for(int kk=(int)(0.5*(Zfis+0.5));kk<80;kk++)
	{
	  dd[kk]=f_eofx->Eval(kk);
	  dd[Zfis-kk]=pow(-1,Zfis)*dd[kk];
	}
      
      for(int jj=Zfis-79;jj<80;jj++)
	geof->SetPoint(geof->GetN(),jj,dd[jj]);
      
      amp=gRandom->Rndm();
      mean=Zfis*0.5;
      width=(gRandom->Rndm())*1.+4.;
      g1->SetParameters(amp,mean,width);
      
      amp=gRandom->Rndm();
      mean=gRandom->Rndm()*4.+50;
      width=(gRandom->Rndm())*1.+2.;
      g2l->SetParameters(amp,Zfis-mean,width);
      g2h->SetParameters(amp,mean,width);
      
      amp=gRandom->Rndm();
      mean=gRandom->Rndm()*4.+54;
      width=(gRandom->Rndm())*1.+2.;
      g3l->SetParameters(amp,Zfis-mean,width);
      g3h->SetParameters(amp,mean,width);
      
      Float_t norm_ypv=0;
      Float_t norm_yv=0;
      
      for(int i=Zfis-79;i<80;i++)
	{
	  zv[i]=i;
	  ypv[i]=(g1->Eval(i)+g2l->Eval(i)+g3l->Eval(i)+g2h->Eval(i)+g3h->Eval(i));
	  yv[i]=ypv[i]*(1.+pow(-1,i)*geof->Eval(i));
	  //yv[i]=gRandom->Gaus(yv[i],0.01*yv[i]);
	  norm_ypv+=ypv[i];
	  norm_yv+=yv[i];
	  
	  dzlv[i]=geof->Eval(i);
	}

      //gy=new TGraph();
      //gyp=new TGraph();
      //gdzl=new TGraph(); 

      for(int i=Zfis-79;i<80;i++)
	{
	  yv[i]=yv[i]/norm_yv;
	  ypv[i]=ypv[i]/norm_ypv;
	  
	  //gy->SetPoint(gy->GetN(),zv[i],yv[i]);
	  //gyp->SetPoint(gyp->GetN(),zv[i],ypv[i]);
	  //gdzl->SetPoint(gdzl->GetN(),zv[i],dzlv[i]);
	}
      for(int i=(int)(0.5*(Zfis+0.5));i<(int)(0.5*(Zfis+0.5))+30;i++)
	{
	  yields_v.push_back(yv[i]);
	  deltas_v.push_back(dzlv[i]);
	}
    }
}
