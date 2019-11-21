function [eta_passed] = update_mini_batch(trainingSet, eta_passed)
    global biases;
        global weights;
    %----------NETWORK-LAYER-PREP-------------
    %Seperate the answers from the array
    numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4;
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
    
    ra=randperm(size(trainingSet,1));
    %-----------------------------------------
  
    
    %--------------BASE-W/B-------------------
%   nabla_b = [np.zeros(b.shape) for b in self.biases]
%   nabla_w = [np.zeros(w.shape) for w in self.weights]
    for i=1:numel(y)
        a=y(i);
        nabla_b(i).b=zeros(a,1); %ax1 array for biases
        nabla_w(i).w=zeros(sizes(i+1),sizes(i));
    end
    %-----------------------------------------
    
    %-----------BASE-W/B------------------
  %for x, y in mini_batch:
    for i=1:size(trainingSet,1)
        x=trainingSet(ra(i)).x;
        y=trainingSet(ra(i)).y;
       %-----------------------------------------

       
       %-----------APPLY BACKPROP----------------
       %delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        [delta_nabla_b, delta_nabla_w]=backprop(x,y);%,weights,biases);
        
       %nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
       %nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for j=1:numel(y)
            nabla_b(j).b=nabla_b(j).b+delta_nabla_b(j).b;
            nabla_w(j).b=nabla_w(j).w+delta_nabla_w(j).w; 
        end
       %-----------------------------------------
    end

   %--------------APPLY-CHANGES-------------------
    for i=1:numel(y)
      %self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
       weights(i).weight=weights(i).weight - nabla_w(i).w*(eta_passed/numel(trainingSet));
      %self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
       biases(i).bias=biases(i).bias-(eta_passed/numel(trainingSet)).*nabla_b(i).b;
    end
    %-----------------------------------------
end

%--------------PYTHON-EQUIV---------------------------
%def update_mini_batch(self, mini_batch, eta):
%Update the network's weights and biases by applying
%    gradient descent using backpropagation to a single mini batch.
%    The "mini_batch" is a list of tuples "(x, y)", and "eta"
%    is the learning rate
%    nabla_b = [np.zeros(b.shape) for b in self.biases]
%    nabla_w = [np.zeros(w.shape) for w in self.weights]
%        for x, y in mini_batch:
%            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
%            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
%            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
%        self.weights = [w-(eta/len(mini_batch))*nw 
%                        for w, nw in zip(self.weights, nabla_w)]
%        self.biases = [b-(eta/len(mini_batch))*nb 
%                for b, nb in zip(self.biases, nabla_b)]
%----------------------------------------------------------------