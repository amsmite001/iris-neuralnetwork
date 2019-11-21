[trainingSet, testTrainingInputs, testTrainingOutputs] = loadfile();

epochs=100;
eta_SGD=0.1;
SGD(trainingSet ,testTrainingInputs, testTrainingOutputs, epochs, eta_SGD);