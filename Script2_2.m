%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

% Plot the data with + indicating (y = 1) examples and o indicating (y = 0) examples.
plotData(X, y);
 
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')

%  Setting options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

% Initializing the fitting parameters
[m, n] = size(X);
% Add intercept term to X
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Add some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%  Predict probability for a student with score 45 on exam 1  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

figure; %creates new figure window
hold on;

% Find Indices of Positive and Negative Examples
pos = find(y == 1); %find returns the indices of elements having 1
neg = find(y == 0); %find returns the indices of elements having 0
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

hold off;

end

function g = sigmoid(z)
    
%SIGMOID Computes sigmoid function. (z can be a matrix, vector or scalar)
g = zeros(size(z));
g = 1./(1+exp(-z));

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Computes cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
n = size(theta);
h = sigmoid(X*theta);
J = 1/m*(-y'*log(h)-(1-y)'*log(1-h));
sum = 0;

for j = 2:n
    sum = sum + theta(j)*theta(j);
end

J = J + lambda/2/m*sum;

sum = 0;
for i=1:m
    sum = sum + (sigmoid(X(i,:)*theta)-y(i))*X(i, 1);
end

grad(1) = 1/m*sum;

for j=2:n
    sum = 0;
    for i=1:m
        sum = sum + (sigmoid(X(i,:)*theta)-y(i))*X(i, j);
    end
    grad(j) = 1/m*sum+lambda/m*theta(j);
end

end

function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculating the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = 1/m*(-y'*log(h)-(1-y)'*log(1-h));
grad = 1/m*X'*(h-y);

end

function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

p = sigmoid(X*theta);

for i = 1:m
    if p(i) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
    
end

end