%S` function for backpropagation
function [costs] = costDeriv(outputActivations, y)
    costs= (outputActivations - y); %.-??
end

%------------Python Equiv-----------------------
%def cost_derivative(self, output_activations, y):
%Return the vector of partial derivatives \partial C_x partial a for the output activations."""       
%   return (output_activations-y)
%-------------END-----------------------------------
