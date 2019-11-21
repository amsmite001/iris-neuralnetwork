function [nabla_b, nabla_w] = backprop(x,inputy)%,weights,biases)
global biases;
        global weights;
    %----------NETWORK-LAYER-PREP-------------
    %Seperate the answers from the array
    numLayers=3; %Input, Hiddenx1, Output
    inputLayerSize=4; %4
    outputLayerSize=3; 
    hiddenLayerSize=5;
    sizes=[inputLayerSize,hiddenLayerSize,outputLayerSize];
    y=[hiddenLayerSize, outputLayerSize];
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
    
%Feedforward-----------------------------------
    activation = transpose(x);
    activations(1).activ = [x];  
    zs=[];
    
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
  
%Backward pass----------------------------------
  %delta = self.cost_derivative(activations[-1], y) * \ sigmoid_prime(zs[-1])
    delta = costDeriv(activations(numel(y)+1).activ, inputy).*sigmoidprime(zs(numel(y)).zs);
   %nabla_b[-1] = delta
    nabla_b(numel(y)).b=delta;
   %nabla_w[-1] = np.dot(delta, activations[-2].transpose()) ERROR Matrix sizes don't match?
    nabla_w(numel(y)).w=delta.*transpose(activations(numel(y)).activ);
    
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
        nabla_w(numel(y)).w=delta.*transpose(activations(numel(y)).activ);
    end
%------------------------------------------------
end

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