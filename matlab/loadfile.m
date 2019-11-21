function [trainingSet, testTrainingInputs, testTrainingOutputs] = loadfile()
   global biases;
    global weights;
    %--------------FILE-INPUT-----------------
    %Load the data set into the array X
    filename='iris-data-numbers.txt';
    trash=load(filename);
    X=importdata(filename,',');
    %-----------------------------------------
    
    %----------NETWORK-LAYER-PREP-------------
    %Seperate the answers from the array
    numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4; %4
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
    
    %Clean the data. The last column is the answer column 
    %            so it needs to be removed and stored in ans
    X=X(randperm(size(X,1)),:);
    trainingData=X(1:150,:);
    
    %----------------TRAIN-DATA-----------------
    trainingInputs=trainingData(:,1:inputLayerSize);
    trainingOutputs=trainingData(:,5);
    for i=1:length(trainingInputs)
        trainingSet(i).x = trainingInputs(i,1:inputLayerSize);
        trainingSet(i).y = trainingOutputs(i);
    end
    %----------------TEST-DATA-----------------
    testTrainingInputs=X(:,1:inputLayerSize);
    testTrainingOutputs=X(:,5);
    
    %-----------WEIGHT/BIAS-------------------
    for i=1:2
        a=y(i);
        biases(i).bias=randn(a,1);
        weights(i).weight=randn(sizes(i+1),sizes(i));
    end
end

%--------------PYTHON-EQUIV-----------------
%def __init__(self, sizes):
%       self.num_layers = len(sizes)
%       self.sizes = sizes
%       self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
%       self.weights = [np.random.randn(y, x)
%             for x, y in zip(sizes[:-1], sizes[1:])]
%-----------------------------------------