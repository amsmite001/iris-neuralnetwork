function z = sigmoid(z)
    z=1./(1.0+exp(-1.*z));
end

%------------Python Equiv-----------------------
%def sigmoid(z):
%    return 1.0/(1.0+np.exp(-z))
%-------------END-----------------------------------
