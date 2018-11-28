function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add bias unit to input layer.
X = [ones(size(X, 1), 1) X];

% Calculate layer1
a2 = arrayfun(@(x) sigmoid(x), (Theta1 * X')');

% Add bias unit to layer1
a2 = [ones(size(a2, 1), 1) a2];

% Calculate layer2
a3 = arrayfun(@(x) sigmoid(x), (Theta2 * a2')');


% Recode y labels to vectors containing only values 0 or 1.
Y = zeros(size(y), num_labels);

for i = 1:m
    y_label = y(i);
    Y(i, y_label) = 1;
end

% Calcuate cost
j = 1/m * (-Y * log(a3)' - (1 - Y) * log(1 - a3)');
% j is a m x m matrix, and the diagonal components are costs of i-th data. 
% The others components are junk. Therefore, remove these junk components by 
% multiplying a m x m diagonal matrix.
j = j .* eye(m);

% Calculaate total cost by summing j.
J = sum(sum(j));


% Add regulation term to the cost.
Theta1_noBias = Theta1(:, 2:end);
Theta2_noBias = Theta2(:, 2:end);
regulation = lambda / (2 * m) * (sum(sum(Theta1_noBias.^2)) + sum(sum(Theta2_noBias.^2)));

J = J + regulation;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end