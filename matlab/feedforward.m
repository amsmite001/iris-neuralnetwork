function [a] = feedforward(a)%, weights, biases)
    global biases;
    global weights;
    a = transpose(a);
    for i=1:2
        a=sigmoid((weights(i).weight*a)+biases(i).bias);
    end
end

%--------------PYTHON-EQUIV-----------------
%def feedforward(self, a):
%Return the output of the network if ``a`` is input.
%   for b, w in zip(self.biases, self.weights):
%      a = sigmoid(np.dot(w, a)+b)
%    return a
%-------------------------------------------