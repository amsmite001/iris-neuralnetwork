%S` function for backpropagation
function sp = sigmoidprime(z)
    sp=sigmoid(z).*(1-sigmoid(z));
end

%------------Python Equiv-----------------------
%def sigmoid_prime(z):
%    Derivative of the sigmoid function.
%   return sigmoid(z)*(1-sigmoid(z))
%-------------END-----------------------------------
