% =========================================================================
% AI Mini-Project 3: Question 1, Part 1
% Fuzzy System for Function Approximation
% =========================================================================

clc;
clear;
close all;

%% --- Configuration Parameters ---
% Centralized place to manage the key parameters of the fuzzy system.
numMFs = 17;             % Number of membership functions for both input and output
inputSigma = 0.08;       % Standard deviation (width) for input Gaussian MFs
outputSigma = 0.03;      % Standard deviation (width) for output Gaussian MFs
domain = [-1, 1];        % Input domain for the variable x
resolution = 0.01;       % Step size for plotting and evaluation

%% --- Step 1: Define and Analyze the Target Function ---
% Define the mathematical function to be approximated.
targetFunc = @(x) exp(-pi*x.^2) + x .* sin(pi*x);

% Generate data points to visualize and analyze the function's range.
inputVector = domain(1):resolution:domain(2);
targetOutput = targetFunc(inputVector);

% Plot the original function to observe its behavior.
figure('Name', 'Target Function h(x)');
plot(inputVector, targetOutput, 'k', 'LineWidth', 2);
title('Analysis of Target Function: h(x)');
xlabel('Input (x)');
ylabel('Output h(x)');
grid on;
legend('h(x) = e^{-\pi x^2} + x \cdot sin(\pi x)');

% Determine the output range programmatically for setting up the output MFs.
minOutput = min(targetOutput);
maxOutput = max(targetOutput);
% Define a robust output range with a small buffer.
outputDomain = [floor(minOutput * 10)/10 - 0.1, ceil(maxOutput * 10)/10 + 0.1];
fprintf('Function h(x) has a range of [%.2f, %.2f] on the domain [%d, %d].\n', minOutput, maxOutput, domain(1), domain(2));
fprintf('The chosen FIS output range is [%.2f, %.2f].\n', outputDomain(1), outputDomain(2));

%% --- Step 2: Design the Fuzzy Inference System (FIS) ---
% Initialize a Mamdani-type FIS.
functionApproximatorFIS = mamfis('Name', 'FunctionApproximator');

% Define the properties of the FIS operators.
functionApproximatorFIS.AndMethod = 'min';
functionApproximatorFIS.OrMethod = 'max';
functionApproximatorFIS.ImplicationMethod = 'min';
functionApproximatorFIS.AggregationMethod = 'max';
functionApproximatorFIS.DefuzzificationMethod = 'centroid';

%% --- Step 3: Define Input and Output Membership Functions ---
% Add the input variable 'x'.
functionApproximatorFIS = addInput(functionApproximatorFIS, domain, 'Name', 'x');
inputCenters = linspace(domain(1), domain(2), numMFs);
for i = 1:numMFs
    functionApproximatorFIS = addMF(functionApproximatorFIS, 'x', 'gaussmf', [inputSigma, inputCenters(i)], 'Name', ['in_mf' num2str(i)]);
end

% Add the output variable 'h_approx'.
functionApproximatorFIS = addOutput(functionApproximatorFIS, outputDomain, 'Name', 'h_approx');
outputCenters = linspace(outputDomain(1), outputDomain(2), numMFs);
for i = 1:numMFs
    functionApproximatorFIS = addMF(functionApproximatorFIS, 'h_approx', 'gaussmf', [outputSigma, outputCenters(i)], 'Name', ['out_mf' num2str(i)]);
end

% Visualize the defined membership functions.
figure('Name', 'Membership Functions');
subplot(2,1,1);
plotmf(functionApproximatorFIS, 'input', 1);
title('Input Membership Functions for x');
subplot(2,1,2);
plotmf(functionApproximatorFIS, 'output', 1);
title('Output Membership Functions for h_{approx}');

%% --- Step 4: Generate Fuzzy Rules ---
% Create rules by mapping the peak of each input MF to the nearest output MF.
% This version creates a rule matrix, which is a different way to achieve the same goal.
ruleList = zeros(numMFs, 4); % Format: [in_mf_idx, out_mf_idx, weight, logic]
for i = 1:numMFs
    % Find the true function value at the center of the current input MF.
    centerPoint = inputCenters(i);
    trueValueAtCenter = targetFunc(centerPoint);
    
    % Find the index of the output MF whose center is closest to this true value.
    [~, closestOutputIndex] = min(abs(outputCenters - trueValueAtCenter));
    
    % Define the rule: IF x is in_mf_i THEN h_approx is out_mf_j
    ruleList(i, :) = [i, closestOutputIndex, 1, 1]; % [InputMF, OutputMF, Weight, AND-operator]
end

% Add the entire set of rules to the FIS.
functionApproximatorFIS = addRule(functionApproximatorFIS, ruleList);
disp(['Successfully generated and added ' num2str(size(ruleList, 1)) ' rules.']);

%% --- Step 5: Evaluate the FIS and Analyze the Error ---
% Generate the approximated output from the fuzzy system.
approximatedOutput = evalfis(functionApproximatorFIS, inputVector');

% Plot the results for a visual comparison.
figure('Name', 'Fuzzy Approximation vs. Actual Function');
plot(inputVector, targetOutput, 'b', 'LineWidth', 2.5);
hold on;
plot(inputVector, approximatedOutput, 'r--', 'LineWidth', 1.5);
title('Comparison: Actual Function vs. Fuzzy Approximation');
xlabel('Input (x)');
ylabel('Output');
legend('Actual h(x)', 'Fuzzy Approximation');
grid on;
hold off;

% Calculate and display the maximum approximation error.
approximationError = abs(targetOutput - approximatedOutput');
maxError = max(approximationError);
fprintf('\nMaximum Approximation Error: %.4f\n', maxError);
if maxError < 0.1
    fprintf('The error is within the desired tolerance of ε = 0.1.\n');
else
    fprintf('The error exceeds the desired tolerance of ε = 0.1.\n');
end