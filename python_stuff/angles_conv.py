import numpy as np
import ROOT
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten

file = ROOT.TFile("U238_Fission_560AMeV_NN_Z91_train.root")
tree = file.evt


dataMatrix = []
dataEvent = []

dataLabel = []
labelMatrix = []

angleVector = []

fPolarMax = np.pi / 2.0
fAzimuthalMax = 2.0*np.pi
fClusterEnergyMax = 600
fSingleCrystalEnergyMax = 360
fNfMax = 210
fNsMax = 240
fAngularDeviationMax = 0.20
fPrimEnergyMax = 1200

nEvents = 100

counter = 0
for t in tree :

 dataEvent = []

 for i in range(2432) :
   dataEvent.append(t.CrystalArray[i] / fSingleCrystalEnergyMax)

 data = dataEvent

 dataMatrix.append(np.array(data))

 dataLabel = [t.ProtonPolar[0] / fPolarMax, t.ProtonPolar[1] / fPolarMax, t.ProtonAzimuthal[0] / fAzimuthalMax, t.ProtonAzimuthal[1] / fAzimuthalMax]

 label = dataLabel

 labelMatrix.append(label)

 angleEvent = [t.ClusterTheta[0] / fPolarMax, t.ClusterTheta[1] / fPolarMax, t.ClusterPhi[0] / fAzimuthalMax, t.ClusterPhi[1] / fAzimuthalMax]
 angle = angleEvent
 angleVector.append(angle)

 dataEvent = []
 counter = counter + 1
 print("Reading : ", 100 * counter / nEvents, end="\r")

 if counter > (nEvents -1):
  break


# Dividing data into training and test data

trainingMatrix = dataMatrix[1:int(len(dataMatrix)*0.7)]
trainingLabels = labelMatrix[1:int(len(labelMatrix)*0.7)]

testMatrix = dataMatrix[int(len(dataMatrix)*0.7):int(len(dataMatrix))]
testLabels = labelMatrix[int(len(labelMatrix)*0.7):int(len(dataMatrix))]

clusterAngles = angleVector[int(len(angleVector)*0.7):int(len(angleVector))]

# The model itself

model = Sequential()
model.add(Conv1D(2,4,2,activation='relu', padding='same', name='convLayer'))
model.add(Conv1D(2,4,(4), activation='relu'))
model.add(MaxPooling1D(4))

model.add(Flatten())
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation = 'sigmoid'))
model.add(Dense(4, activation = 'relu'))

model.build(input_shape = (int(len(dataMatrix)*0.7),2432,1))
model.summary()

model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='adam', metrics=[keras.losses.MeanAbsoluteError()])


history = model.fit(np.array(trainingMatrix).reshape(len(trainingMatrix),2432,1),np.array(trainingLabels).reshape(len(trainingLabels),4,1), epochs=1000, batch_size=8)


# Testing the model
y_pred = model.predict(np.array(testMatrix).reshape(len(testMatrix),2432,1))

hPolarResiduesCluster = ROOT.TH1F("PolarResiduesCluster"," Polar Residues Cluster ",300, -10,10)
hPolarResiduesNN = ROOT.TH1F("PolarResiduesNN"," Polar Residues NN ",300, -10,10)

hAzimuthalResiduesCluster = ROOT.TH1F("AzimuthalResiduesCluster"," Azimuthal Residues Cluster ",300, -10,10)
hAzimuthalResiduesNN = ROOT.TH1F("AzimuthalResiduesNN"," Azimuthal Residues NN ",300, -10,10)

for i in range (len(y_pred)):

 hPolarResiduesCluster.Fill((180/3.14159)*fPolarMax*(testLabels[i][0] - clusterAngles[i][0]))
 hPolarResiduesCluster.Fill((180/3.14159)*fPolarMax*(testLabels[i][1] - clusterAngles[i][1]))

 hAzimuthalResiduesCluster.Fill((180/3.14159)*fAzimuthalMax*(testLabels[i][2] - clusterAngles[i][2]))
 hAzimuthalResiduesCluster.Fill((180/3.14159)*fAzimuthalMax*(testLabels[i][3] - clusterAngles[i][3]))

 hPolarResiduesNN.Fill((180/3.14159)*fPolarMax*(testLabels[i][0] - y_pred[i][0]))
 hPolarResiduesNN.Fill((180/3.14159)*fPolarMax*(testLabels[i][1] - y_pred[i][1]))

 hAzimuthalResiduesNN.Fill((180/3.14159)*fAzimuthalMax*(testLabels[i][2] - y_pred[i][2]))
 hAzimuthalResiduesNN.Fill((180/3.14159)*fAzimuthalMax*(testLabels[i][3] - y_pred[i][3]))


hAzimuthalResiduesNN.SetLineColor(1)
hPolarResiduesNN.SetLineColor(1)

hAzimuthalResiduesCluster.SetLineColor(2)
hPolarResiduesCluster.SetLineColor(2)

if (hAzimuthalResiduesNN.GetMaximum() > hAzimuthalResiduesCluster.GetMaximum()):
 hAzimuthalResiduesNN.SetMaximum(1.2*hAzimuthalResiduesNN.GetMaximum())
 hAzimuthalResiduesCluster.SetMaximum(1.2*hAzimuthalResiduesNN.GetMaximum())

else:
 hAzimuthalResiduesNN.SetMaximum(1.2*hAzimuthalResiduesCluster.GetMaximum())
 hAzimuthalResiduesCluster.SetMaximum(1.2*hAzimuthalResiduesCluster.GetMaximum())

if (hPolarResiduesNN.GetMaximum() > hPolarResiduesCluster.GetMaximum()):
 hPolarResiduesNN.SetMaximum(1.2*hPolarResiduesNN.GetMaximum())
 hPolarResiduesCluster.SetMaximum(1.2*hPolarResiduesNN.GetMaximum())

else:
 hPolarResiduesNN.SetMaximum(1.2*hPolarResiduesCluster.GetMaximum())
 hPolarResiduesCluster.SetMaximum(1.2*hPolarResiduesCluster.GetMaximum())




canvas = ROOT.TCanvas("results","results")
canvas.Divide(2,1)

canvas.cd(1)
hPolarResiduesCluster.Draw()
hPolarResiduesNN.Draw("SAME")

canvas.cd(2)
hAzimuthalResiduesCluster.Draw()
hAzimuthalResiduesNN.Draw("SAME")
