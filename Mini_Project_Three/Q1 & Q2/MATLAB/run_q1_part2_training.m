% =========================================================================
% AI Mini-Project 3: Question 1, Part 2
% Main script for training a Mamdani FIS with Gradient Descent
% =========================================================================

clc;
clear;
close all;

%% --- Configuration & Setup ---
% Define the target function
targetFunc = @(x) exp(-pi*x.^2) + x .* sin(pi*x);

% Training and FIS parameters stored in a configuration struct
config = struct();
config.numTrainingData = 300;
config.numInputMFs = 18;
config.numOutputMFs = 18;
config.mfType = 'gaussmf';
config.defuzzMethods = {'centroid', 'bisector', 'mom'}; % Methods to test
config.domain = [-1, 1];

% Gradient Descent Hyperparameters
config.epochs = 50;
config.lr_mean = 0.05;    % Learning rate for MF means
config.lr_sigma = 0.01; % Learning rate for MF sigmas
config.earlyStopTol = 1e-5; % Tolerance for early stopping

% --- Generate Training Data ---
trainingInputs = linspace(config.domain(1), config.domain(2), config.numTrainingData)';
trainingTargets = targetFunc(trainingInputs);
fprintf('Generated %d training data points.\n', config.numTrainingData);

% --- Main Experiment Loop ---
results = struct();
figure('Name', 'MSE Learning Curves'); % Figure for all learning curves

for i = 1:length(config.defuzzMethods)
    method = config.defuzzMethods{i};
    fprintf('\n--- Training with Defuzzification Method: %s ---\n', upper(method));

    % 1. Create the initial Fuzzy Inference System
    initialFIS = createInitialMamdaniFIS(config, method, targetFunc);
    fprintf('Initial FIS created with %d rules.\n', length(initialFIS.Rules));

    % 2. Train the FIS using numerical gradient descent
    [trainedFIS, mseHistory] = trainFIS_GradientDescent(initialFIS, trainingInputs, trainingTargets, config);
    
    % 3. Store and display results for this method
    results.(method).fis = trainedFIS;
    results.(method).mseHistory = mseHistory;
    
    % Plot learning curve
    plot(1:length(mseHistory), mseHistory, 'LineWidth', 2);
    hold on;

    % 4. Evaluate the final trained model
    evalDomain = -1:0.01:1;
    actualOutput = targetFunc(evalDomain);
    fuzzyOutput = evalfis(trainedFIS, evalDomain');
    
    finalMaxError = max(abs(actualOutput - fuzzyOutput'));
    results.(method).maxError = finalMaxError;
    
    fprintf('Training complete for method "%s".\n', method);
    fprintf('Final MSE: %.6f | Final Max Error: %.4f\n', mseHistory(end), finalMaxError);
end

% --- Final Comparison ---
hold off;
title('MSE vs. Epochs for Different Defuzzification Methods');
xlabel('Epoch');
ylabel('Mean Squared Error (MSE)');
legend(config.defuzzMethods);
grid on;
saveas(gcf, 'q2_learning_curves.png'); % Save for report

% Find and announce the best method
bestMethod = '';
minError = inf;
for i = 1:length(config.defuzzMethods)
    method = config.defuzzMethods{i};
    if results.(method).maxError < minError
        minError = results.(method).maxError;
        bestMethod = method;
    end
end
fprintf('\n=================================================');
fprintf('\nBest performing method is "%s" with a max error of %.4f.\n', upper(bestMethod), minError);
fprintf('=================================================\n');

% Plot the best result
figure('Name', 'Best Approximation Result');
plot(evalDomain, actualOutput, 'b', 'LineWidth', 2.5);
hold on;
plot(evalDomain, evalfis(results.(bestMethod).fis, evalDomain'), 'r--', 'LineWidth', 1.5);
title(sprintf('Best Approximation by "%s" Method', upper(bestMethod)));
xlabel('x');
ylabel('h(x)');
legend('Actual Function', 'Fuzzy Approximation');
grid on;
saveas(gcf, 'q2_best_approximation.png');