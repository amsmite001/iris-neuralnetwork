function [n_test_score] = evaluate(testTrainingData,testTrainingOutputs)%,weights,biases)
    global biases;
    global weights;
n_test_score=0;
    indexA=0;
    
    for i=1:size(testTrainingData,1)
        outx=feedforward(testTrainingData(i,:));%,weights,biases);
        
        %fprintf("Outx is: %d \n", outx);
        %Taks: Create Max function
        %maxV=maxValue(outx);
        %return sum(int(x == y) for (x, y) in test_results) FIX
        if(find(outx == max(outx(:)))==testTrainingOutputs(i))
            n_test_score=n_test_score+1;
        %end
        end
        
   % outx
    end
end


%--------------PYTHON-EQUIV---------------------------
%def evaluate(self, test_data):
%Return the number of test inputs for which the neural
%      network outputs the correct result. Note that the neural
%      network's output is assumed to be the index of whichever
%      neuron in the final layer has the highest activation.

%       test_results = [(np.argmax(self.feedforward(x)), y)
%                        for (x, y) in test_data]
%        return sum(int(x == y) for (x, y) in test_results)
%-------------------------------------------------------------
