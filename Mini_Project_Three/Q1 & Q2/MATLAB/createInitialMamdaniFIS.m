function fis = createInitialMamdaniFIS(config, defuzzMethod, targetFunc)
    % Creates and initializes a Mamdani FIS based on the configuration.
    
    fis = mamfis('Name', ['FIS_GD_' defuzzMethod]);
    fis.DefuzzificationMethod = defuzzMethod;

    % Add input variable and MFs
    fis = addInput(fis, config.domain, 'Name', 'x');
    inputCenters = linspace(config.domain(1), config.domain(2), config.numInputMFs);
    inputSigmas = ones(1, config.numInputMFs) * (diff(config.domain) / (config.numInputMFs - 1));
    for i = 1:config.numInputMFs
        fis = addMF(fis, 'x', config.mfType, [inputSigmas(i), inputCenters(i)], 'Name', ['in' num2str(i)]);
    end

    % Add output variable and MFs
    outputRange = [min(targetFunc(config.domain(1):0.01:config.domain(2))), max(targetFunc(config.domain(1):0.01:config.domain(2)))];
    fis = addOutput(fis, outputRange, 'Name', 'h_approx');
    outputCenters = linspace(outputRange(1), outputRange(2), config.numOutputMFs);
    outputSigmas = ones(1, config.numOutputMFs) * (diff(outputRange) / (config.numOutputMFs - 1));
    for i = 1:config.numOutputMFs
        fis = addMF(fis, 'h_approx', config.mfType, [outputSigmas(i), outputCenters(i)], 'Name', ['out' num2str(i)]);
    end
    
    % Generate and add rules by mapping input MF centers to the closest output MF
    ruleList = zeros(config.numInputMFs, 4);
    for i = 1:config.numInputMFs
        trueValueAtCenter = targetFunc(inputCenters(i));
        [~, closestOutputIndex] = min(abs(outputCenters - trueValueAtCenter));
        ruleList(i, :) = [i, closestOutputIndex, 1, 1];
    end
    fis = addRule(fis, ruleList);
end