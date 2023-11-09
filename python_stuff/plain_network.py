import numpy as np
import ROOT
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

file = ROOT.TFile("U238_Fission_560AMeV_NN_Z91_train.root")
tree = file.evt

dataMatrix = []
dataEvent = []

dataLabel = []
labelMatrix = []

fPolarMax = np.pi / 2.0
fAzimuthalMax = 2.0*np.pi
fClusterEnergyMax = 600
fSingleCrystalEnergyMax = 360
fNfMax = 210
fNsMax = 240
fAngularDeviationMax = 0.20
fPrimEnergyMax = 1200

nEvents = 100000


counter = 0
for t in tree :


 dataEvent = [t.ClusterEnergy[0] / fClusterEnergyMax, t.MotherCrystalEnergy[0] / fSingleCrystalEnergyMax, t.ClusterTheta[0] / fPolarMax, t.ClusterPhi[0] / fAzimuthalMax, t.MotherCrystalNf[0] / fNfMax , t.MotherCrystalNs[0] / fNsMax, t.AngularDeviation[0] / fAngularDeviationMax, t.ClusterEnergy[1] / fClusterEnergyMax , t.MotherCrystalEnergy[1] / fSingleCrystalEnergyMax, t.ClusterTheta[1] / fPolarMax, t.ClusterPhi[1] / fAzimuthalMax, t.MotherCrystalNf[1] / fNfMax , t.MotherCrystalNs[1] / fNsMax, t.AngularDeviation[1] / fAngularDeviationMax]

 data = dataEvent

 dataMatrix.append(data)

 dataLabel = [t.ProtonEnergy[0] / fPrimEnergyMax, t.ProtonEnergy[1] / fPrimEnergyMax]

 label = dataLabel

 labelMatrix.append(label)

 counter = counter + 1
 print("Reading : ", 100 * counter / nEvents, end="\r")

 if counter > (nEvents -1):
  break


trainingMatrix = dataMatrix[1:int(len(dataMatrix)*0.7)]
trainingLabels = labelMatrix[1:int(len(labelMatrix)*0.7)]

testMatrix = dataMatrix[int(len(dataMatrix)*0.7):int(len(dataMatrix))]
testLabels = labelMatrix[int(len(labelMatrix)*0.7):int(len(dataMatrix))]


model = Sequential()
model.add(Dropout(0.2, input_shape = (14,)))
model.add(Dense(14, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam', metrics=[keras.losses.MeanSquaredError()])

model.summary()

history = model.fit(trainingMatrix,trainingLabels, epochs=100, batch_size=8)

y_pred = model.predict(testMatrix)

h = ROOT.TH2F("Residues","Residues",500,-1000,1000,500,0,1000)

for i in range (len(y_pred)):
 h.Fill(fPrimEnergyMax*(testLabels[i][0] - y_pred[i][0]), fPrimEnergyMax*testLabels[i][0])
 h.Fill(fPrimEnergyMax*(testLabels[i][1] - y_pred[i][1]), fPrimEnergyMax*testLabels[i][1])

 print("Predicted : ",fPrimEnergyMax*y_pred[i][0], " and ", fPrimEnergyMax*y_pred[i][1])
 print("Real : ",fPrimEnergyMax*testLabels[i][0], " and ", fPrimEnergyMax*testLabels[i][1])
 print(" =========== End of Event ==========")

h.Draw("COLZ")
