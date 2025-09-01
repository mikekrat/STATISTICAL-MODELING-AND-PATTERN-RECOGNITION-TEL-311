function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	[NumSamples, NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels)
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    %For each class i
	%Find the necessary statistics
    P = zeros;
    Sw = zeros;
    mu = zeros(NumClasses, NumFeatures);
    for i = 1 : NumClasses
        prev_i = i-1;
        
        %Calculate the Class Prior Probability
        P(i) = sum(Labels == prev_i) / size(NumSamples, 1);
        
        %Calculate the Class Mean 
        mu(i,:) = mean( Samples(Labels == prev_i, :) );
        
        %Calculate the Within Class Scatter Matrix
        Sw = Sw + P(i) .* cov( Samples(Labels == prev_i, :) );
    end
    
    %Calculate the Global Mean
	m0 = sum(mu) ./ NumClasses;

    %Calculate the Between Class Scatter Matrix
	Sb = zeros; 
    for i = 1 : NumClasses
       Sb = Sb + P(i) .* ( (mu(i) - m0)' * (mu(i) - m0) ); 
    end
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = Sw \ Sb;
    
    %Perform Eigendecomposition
    [V, D] = eig(EigMat);
    [~, indexes] = sort(diag(D), 1, 'descend');
    V_Sorted = V(:, indexes);
    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	A = V_Sorted(:, 1 : NewDim);  % Return the LDA projection vectors
    
    
