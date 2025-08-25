function [fis, mse_history] = trainFIS_GradientDescent(fis, trainingInputs, trainingTargets, config)
    % Trains the FIS using numerical gradient descent.
    
    mse_history = zeros(config.epochs, 1);
    delta = 1e-4; % Small change for numerical gradient calculation

    for epoch = 1:config.epochs
        % Calculate current error
        current_predictions = evalfis(fis, trainingInputs);
        mse_history(epoch) = mean((trainingTargets - current_predictions).^2);
        fprintf('Epoch %d/%d, MSE: %.6f\n', epoch, config.epochs, mse_history(epoch));

        % Early stopping condition
        if epoch > 1 && abs(mse_history(epoch) - mse_history(epoch-1)) < config.earlyStopTol
            fprintf('Convergence tolerance reached. Stopping training.\n');
            mse_history = mse_history(1:epoch); % Trim unused history
            break;
        end

        original_fis = fis; % Store original state for this epoch

        % --- Update Input MF Parameters ---
        for i = 1:config.numInputMFs
            % Update Mean
            p_mean = original_fis.Inputs(1).MembershipFunctions(i).Parameters(2);
            temp_fis = original_fis;
            temp_fis.Inputs(1).MembershipFunctions(i).Parameters(2) = p_mean + delta;
            mse_plus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            temp_fis.Inputs(1).MembershipFunctions(i).Parameters(2) = p_mean - delta;
            mse_minus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            grad_mean = (mse_plus - mse_minus) / (2 * delta);
            new_mean = p_mean - config.lr_mean * grad_mean;
            fis.Inputs(1).MembershipFunctions(i).Parameters(2) = max(config.domain(1), min(config.domain(2), new_mean));

            % Update Sigma
            p_sigma = original_fis.Inputs(1).MembershipFunctions(i).Parameters(1);
            temp_fis = original_fis;
            temp_fis.Inputs(1).MembershipFunctions(i).Parameters(1) = p_sigma + delta;
            mse_plus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            temp_fis.Inputs(1).MembershipFunctions(i).Parameters(1) = p_sigma - delta;
            mse_minus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            grad_sigma = (mse_plus - mse_minus) / (2 * delta);
            new_sigma = p_sigma - config.lr_sigma * grad_sigma;
            fis.Inputs(1).MembershipFunctions(i).Parameters(1) = max(1e-5, new_sigma); % Ensure sigma is positive
        end

        % --- Update Output MF Parameters ---
        output_range = fis.Outputs(1).Range;
        for i = 1:config.numOutputMFs
            % Update Mean
            p_mean = original_fis.Outputs(1).MembershipFunctions(i).Parameters(2);
            temp_fis = original_fis;
            temp_fis.Outputs(1).MembershipFunctions(i).Parameters(2) = p_mean + delta;
            mse_plus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            temp_fis.Outputs(1).MembershipFunctions(i).Parameters(2) = p_mean - delta;
            mse_minus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            grad_mean = (mse_plus - mse_minus) / (2 * delta);
            new_mean = p_mean - config.lr_mean * grad_mean;
            fis.Outputs(1).MembershipFunctions(i).Parameters(2) = max(output_range(1), min(output_range(2), new_mean));

            % Update Sigma
            p_sigma = original_fis.Outputs(1).MembershipFunctions(i).Parameters(1);
            temp_fis = original_fis;
            temp_fis.Outputs(1).MembershipFunctions(i).Parameters(1) = p_sigma + delta;
            mse_plus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            temp_fis.Outputs(1).MembershipFunctions(i).Parameters(1) = p_sigma - delta;
            mse_minus = mean((trainingTargets - evalfis(temp_fis, trainingInputs)).^2);
            grad_sigma = (mse_plus - mse_minus) / (2 * delta);
            new_sigma = p_sigma - config.lr_sigma * grad_sigma;
            fis.Outputs(1).MembershipFunctions(i).Parameters(1) = max(1e-5, new_sigma);
        end
    end
end