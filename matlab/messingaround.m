A =[1 2 3; 4 5 6]
%%
B=A(2,2:3)

[X, trainingInputs, testTrainingInputs, testTrainingOutputs, sizes, weights,biases] = loadfile()
%%

  numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4; %4
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
    
    ra=randperm(numel(trainingInputs));
    %-----------------------------------------
%%
    %numTrainingInputs=numel(trainingInputs);
   % eta=eta_passed;
    
    %--------------BASE-W/B-------------------
%   nabla_b = [np.zeros(b.shape) for b in self.biases]
%   nabla_w = [np.zeros(w.shape) for w in self.weights]
    for i=1:numel(y)
        a=y(i);
        nabla_b(i).b=zeros(a,1); %ax1 array for biases
        nabla_w(i).w=zeros(sizes(i+1),sizes(i));
    end
    %-----------------------------------------
    %%
    %-----------BASE-W/B------------------
  %for x, y in mini_batch:
    for i=1:numel(trainingInputs)
        x=trainingInputs(ra(i)).x;
        y=trainingInputs(ra(i)).y;
       %-----------------------------------------

     
       %-----------APPLY BACKPROP----------------
       %delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        [delta_nabla_b, delta_nabla_w]=backprop(x,y,weights,biases);
        
       %nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
       %nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for j=1:numel(y)
            nabla_b(j).b=nabla_b(j).b+delta_nabla_b(j).b;
            nabla_w(j).b=nabla_w(j).w+delta_nabla_w(j).w; 
        end
       %-----------------------------------------
    end
%%
   %--------------APPLY-CHANGES-------------------
    for i=1:numel(y)
      %self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
       weights(i).weight=weights(i).weight - nabla_w(i).w*(eta_passed/numel(trainingInputs));
      %self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
       biases(i).bias=biases(i).bias-(eta_passed/numel(trainingInputs)).*nabla_b(i).b;
    end
    %-----------------------------------------
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% backprop

iy=1
  numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4; %4
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
    %-----------------------------------------
 %%   
    %--------------BASE-W/B-------------------
%   nabla_b = [np.zeros(b.shape) for b in self.biases]
%   nabla_w = [np.zeros(w.shape) for w in self.weights]
    for i=1:numel(y)
        a=y(i);
        nabla_b(i).b=zeros(a,1); %ax1 array for biases
        nabla_w(i).w=zeros(sizes(i+1),sizes(i));
    end
    %-----------------------------------------
    %%
%Feedforward-----------------------------------
    activation = transpose(x);
    activations(1).activ = [x];  
    zs=[];
    %%
    numel(y)
    hey = weights(1).weight*activation
    
    
    %%
  %for b, w in zip(self.biases, self.weights):
    for i=1:numel(y)
       % z = np.dot(w, activation)+b
        z=weights(i).weight*activation + biases(i).bias;
       %zs.append(z)
        zs(i).zs=z;
       %activation = sigmoid(z)
        activation=sigmoid(z);
       %activations.append(activation)
        activations(i+1).activ=activation;
    end
%------------------------------------------------
  %%
%Backward pass----------------------------------
  %delta = self.cost_derivative(activations[-1], y) * \ sigmoid_prime(zs[-1])
    delta = costDeriv(activations(numel(y)+1).activ, iy).*sigmoidprime(zs(numel(y)).zs);
   %nabla_b[-1] = delta
    nabla_b(numel(y)).b=delta;
%%   %nabla_w[-1] = np.dot(delta, activations[-2].transpose()) ERROR Matrix sizes don't match?
    nabla_w(numel(y)).w=delta.*transpose(activations(numel(y)).activ);
    %%
   %for l in xrange(2, self.num_layers):
    for l=1:numel(y)-1
       %z = zs[-l]
        z=zs(numel(y)-l).zs;
       %sp = sigmoid_prime(z)
        sp=sigmoidprime(z);
       %delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        delta=(transpose(weights(numel(y)-l+1).weight)*delta).*sp;
        
       %nabla_b[-l] = delta
        nabla_b(numel(y)).b=delta;
       %nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
       %matlab: nabla_w(numel(y)-l)
        nabla_w(numel(y)).w=delta.*transpose(activations(numel(y)-1).activ);
    end
%------------------------------------------------

%--------------PYTHON-EQUIV---------------------------
%def backprop(self, x, y):
%Return a tuple ``(nabla_b, nabla_w)`` representing the
%  gradient for the cost function C_x.  ``nabla_b`` and
%  ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
%   To ``self.biases`` and ``self.weights``.

%   nabla_b = [np.zeros(b.shape) for b in self.biases]
%   nabla_w = [np.zeros(w.shape) for w in self.weights]

% Feedforward---------------------------------
%    activation = x
%    activations = [x] # list to store all the activations, layer by layer
%    zs = [] # list to store all the z vectors, layer by layer
%    for b, w in zip(self.biases, self.weights):
%        z = np.dot(w, activation)+b
%        zs.append(z)
%        activation = sigmoid(z)
%        activations.append(activation)

%Backward pass----------------------------------
%        delta = self.cost_derivative(activations[-1], y) * \ sigmoid_prime(zs[-1])
%        nabla_b[-1] = delta
%        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
%Note that the variable l in the loop below is used a little
% differently to the notation in Chapter 2 of the book.  Here,
% l = 1 means the last layer of neurons, l = 2 is the
% second-last layer, and so on.  It's a renumbering of the
% scheme in the book, used here to take advantage of the fact
%  that Python can use negative indices in lists.
%    for l in xrange(2, self.num_layers):
%         z = zs[-l]
%         sp = sigmoid_prime(z)
%         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
%         nabla_b[-l] = delta
%         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
%     return (nabla_b, nabla_w)   
%-------------------------------------------

%%


testTrainingInputs(i,:)
%%
    n_test_score=0;
    
    for i=1:size(testTrainingInputs,1)
        outx=feedforward(testTrainingInputs(i),weights,biases);
        fprintf('HERE outx is %d',outx);
      %return sum(int(x == y) for (x, y) in test_results) FIX
        if(outx==testTrainingOutputs(i))
            n_test_score=n_test_score+1;
        end
    end
    
    
    %% sigmoid
    
 a = testTrainingInputs(7,:)
 transpose(a)
 weights(1).weight*transpose(a)
 %%
    y=[5,3];

        z=weights(1).weight*transpose(a)+biases(1).bias;
        %%
        a=sigmoid(z);
        
        %%
        a
        %%
              z=weights(2).weight*transpose(a)+biases(2).bias;
              %%
        a=sigmoid(z);

%%
i=5
%%
global weights;
global biases;
outx=feedforward(testTrainingInputs(i,:))%,weights,biases)
maxValue(outx)
i=i+2;

%%
[X, trainingSet, testTrainingInputs, testTrainingOutputs, sizes, weights,biases] = loadfile()

%%
%%%%%%%%%%%%%%%%%%%%%%%%%% update mini batch

% [weights,biases, delta_nabla_b, delta_nabla_w] = update_mini_batch(trainingInputs, eta_passed, weights, biases)

    %----------NETWORK-LAYER-PREP-------------
    %Seperate the answers from the array
    numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4; %4
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
    
    trainingInputs=testTrainingInputs;
    ra=randperm(size(trainingInputs,1));
    %-----------------------------------------
   %% 
    %numTrainingInputs=numel(trainingInputs);
    eta_passed = .3;
    eta=eta_passed;
    
    %--------------BASE-W/B-------------------
%   nabla_b = [np.zeros(b.shape) for b in self.biases]
%   nabla_w = [np.zeros(w.shape) for w in self.weights]
    for i=1:numel(y)
        a=y(i);
        nabla_b(i).b=zeros(a,1); %ax1 array for biases
        nabla_w(i).w=zeros(sizes(i+1),sizes(i));
    end
    %-----------------------------------------
    %%
    trainingInputs(ra(25),:)
    %%
    
    %-----------BASE-W/B------------------
  %for x, y in mini_batch:
    for i=1:size(trainingInputs,1)
        x=trainingInputs(ra(i),:);
        y=trainingOutputs(ra(i));
       %-----------------------------------------

       
       %-----------APPLY BACKPROP----------------
       %delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        [delta_nabla_b, delta_nabla_w]=backprop(x,y,weights,biases);
        
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
       weights(i).weight=weights(i).weight - nabla_w(i).w*(eta_passed/numel(trainingInputs));
      %self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
       biases(i).bias=biases(i).bias-(eta_passed/numel(trainingInputs)).*nabla_b(i).b;
    end
    %-----------------------------------------


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

