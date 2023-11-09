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

fPolarMax = np.pi / 2.0
fAzimuthalMax = 2.0*np.pi
fClusterEnergyMax = 600
fSingleCrystalEnergyMax = 360
fNfMax = 210
fNsMax = 240
fAngularDeviationMax = 0.20
fPrimEnergyMax = 1200
nEvents = 1000

counter = 0
for t in tree :

 dataEvent = []
 for i in range(2432) :
  dataEvent.append( t.CrystalArray[i] / fSingleCrystalEnergyMax)

 data = dataEvent

 dataMatrix.append(np.array(data))

 dataLabel = [t.ProtonEnergy[0] / fPrimEnergyMax, t.ProtonEnergy[1] / fPrimEnergyMax]

 label = dataLabel

 labelMatrix.append(label)

 counter = counter + 1


 print("Reading file : ", 100 * counter / nEvents, end="\r")

 if counter > (nEvents -1):
  break

trainingMatrix = dataMatrix[1:int(len(dataMatrix)*0.7)]
trainingLabels = labelMatrix[1:int(len(labelMatrix)*0.7)]

testMatrix = dataMatrix[int(len(dataMatrix)*0.7):int(len(dataMatrix))]
testLabels = labelMatrix[int(len(labelMatrix)*0.7):int(len(dataMatrix))]

model = Sequential()
model.add(Conv1D(4,4,2,activation='relu', padding='same', name='convLayer'))
model.add(Conv1D(2,3,(4), activation='relu'))

model.add(Flatten())
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))

model.build(input_shape = (int(len(trainingMatrix)),2432,1))
model.summary()

model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam', metrics=[keras.losses.MeanSquaredError()])


history = model.fit(np.array(trainingMatrix).reshape(len(trainingMatrix),2432,1),np.array(trainingLabels).reshape(len(trainingLabels),2,1), epochs=500, batch_size=8)

y_pred = model.predict(np.array(testMatrix).reshape(len(testMatrix),2432,1))

h1 = ROOT.TH2F("Residues1","Residues 1",500,-1000,1000,500,0,1000)
h2 = ROOT.TH2F("Residues2","Residues 2 ",500,-1000,1000,500,0,1000)

canvas = ROOT.TCanvas("results","results")
canvas.Divide(2,1)


for i in range (len(y_pred)):
 h1.Fill(fPrimEnergyMax*(labelMatrix[i][0] - y_pred[i][0]), fPrimEnergyMax*labelMatrix[i][0])
 h2.Fill(fPrimEnergyMax*(labelMatrix[i][1] - y_pred[i][1]), fPrimEnergyMax*labelMatrix[i][1])
 print("Predicted Energy for First Proton : ",fPrimEnergyMax*y_pred[i][0])
 print("Predicted Energy for Second Proton : ",fPrimEnergyMax*y_pred[i][1])
 print("Reals : ",fPrimEnergyMax*labelMatrix[i][0]," ",fPrimEnergyMax*labelMatrix[i][1])
 print("\n")


canvas.cd(1)
h1.Draw("COLZ")

canvas.cd(2)
h2.Draw("COLZ")
