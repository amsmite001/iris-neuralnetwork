# Tasks and Known Issues
PROBLEM:
    Currently I'm only handling 1 input at a time instead of 4. How do I fix this?
        * The original problem comes from a 1 input, 3 hidden layer, 1 output problem. I have 3 inputs, 5 hidden and 3 outputs.

Need to randomize data, currently the training data is all 3s
Theres an OOB error with evaluate. Needs to be examined

# Function List

[U]   Initialize (Loadfile) Task: Create Training subset, does the training inputs need to include the answer??
[X]   Feedforward
[ ]   SGD
[X]   Update_mini_batch
[X]   Backpropagation
[??]   Evaluate (What is y?)
[D]   CostDeriv {Might have an error with the operation)
[D]   Sigmoid
[D]   Sigmoid Prime
[ ]   Run