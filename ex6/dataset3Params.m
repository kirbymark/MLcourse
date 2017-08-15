function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possibleCValues = [0.01 0.03 0.1 0.3 1 3 10 30];
possibleSigmaValues = [0.01 0.03 0.1 0.3 1 3 10 30];
% start with ver large error value
errorBenchmark=Inf;  
fprintf('Starting error is = %f', errorBenchmark)

for C = possibleCValues
  for sigma = possibleSigmaValues
    fprintf('.');
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    err   = mean(double(svmPredict(model, Xval) ~= yval));
    % if the new error is less than old keep track of the values used	
    if( err <= errorBenchmark )
      CFinal     = C;
      sigmaFinal = sigma;
	  % and set the new min error
      errorBenchmark   = err;
      fprintf('Smaller Error found with C, sigma = %f, %f - the error now is = %f', CFinal, sigmaFinal, errorBenchmark)
    end
  end
end
C     = CFinal;
sigma = sigmaFinal;

fprintf('Smallest Error found with C, sigma = %f, %f - the error now is = %f', CFinal, sigmaFinal, errorBenchmark)



% =========================================================================

end
