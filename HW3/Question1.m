%Question 1

%All four sample sizes
for i = 1:4
    markerList = '+*ox';
    %Tolerance for EM stopping criterion
    delta = 1e-4;
    %Regularization parameter for covariance estimates
    regWeight = 1e-10; 
    %K-Fold Cross Validation
    K = 10; 
    %Number of samples
    N = [10,100,1000,10000]; 

    % Generate samples from a 4-component GMM
    alpha_true = [0.4,0.3,0.2,0.1];
    mu_true = [-20 -4 10 12;-20 3 10 -8];
    Sigma_true(:,:,1) = [4 1;1 20];
    Sigma_true(:,:,2) = [8 1;1 2];
    Sigma_true(:,:,3) = [2 1;1 16];
    Sigma_true(:,:,4) = [16 1;1 4];
    x = randGMM(N(i),alpha_true,mu_true,Sigma_true);
    %To determine dimensionality of samples and number of GMM components
    [d,MM] = size(mu_true); 
    
    figure(9)
    hold on
    plot(x(1,:),x(2,:),markerList(i))
    hold off
    if i == 4
        legend('N = 10', 'N = 100', 'N = 1000', 'N = 10000');
    end
    % Divide the data set into 10 approximately-equal-sized partitions
    dummy = ceil(linspace(0,N(i),K+1));
    for k = 1:K
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end

    % Allocate space
    loglikelihoodtrain = zeros(K,6); loglikelihoodvalidate = zeros(K,6); 
    Averagelltrain = zeros(1,6); Averagellvalidate = zeros(1,6);
    BICtrain = zeros(1,6); BICvalidate = zeros(1,6);
    % Try all 6 mixture options
    for M = 1:6

        % 10-fold cross validation
        for k = 1:K

            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            x1Validate = x(1,indValidate); % Using folk k as validation set
            x2Validate = x(2,indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N(i)];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):N(i)];
            end
            x1Train = x(1,indTrain); % using all other folds as training set
            x2Train = x(2,indTrain);
            xTrain = [x1Train; x2Train];
            xValidate = [x1Validate; x2Validate];
            Ntrain = length(indTrain); Nvalidate = length(indValidate);
            % Train model parameters (EM)
            %Initialize the GMM to randomly selected samples
            alpha = ones(1,M)/M;
            shuffledIndices = randperm(Ntrain);
            %Pick M random samples as initial mean estimates
            mu = xTrain(:,shuffledIndices(1:M)); 
            %Assign each sample to the nearest mean
            [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); 
            %Use sample covariances of initial assignments as initial covariance estimates
            for m = 1:M 
                Sigma(:,:,m) = cov(xTrain(:,assignedCentroidLabels==m)') + regWeight*eye(d,d);
            end
            t = 0;
            % Not converged at the beginning
            Converged = 0; 
            count = 1;
            while ~Converged
                count = count+1;
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                clear temp
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
                muNew = xTrain*w';
                for l = 1:M
                    v = xTrain-repmat(muNew(:,l),1,Ntrain);
                    u = repmat(w(l,:),d,1).*v;
                    %Adding a small regularization term
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); 
                end
                Dalpha = sum(abs(alphaNew-alpha));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                %Check if converged
                Converged = ((Dalpha+Dmu+DSigma)<delta); 
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                t = t+1;
            end
            % Validation
            loglikelihoodtrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)))/(length(indTrain));
            loglikelihoodvalidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)))/(N(i)-length(indTrain));
        end
        Averagelltrain(1,M) = mean(loglikelihoodtrain(:,M)); % average training MSE over folds
        BICtrain(1,M) = -2*Averagelltrain(1,M)+M*log(N(i));
        Averagellvalidate(1,M) = mean(loglikelihoodvalidate(:,M)); % average validation MSE over folds
        if isinf(Averagellvalidate(1,M))
            Averagellvalidate(1,M) = (min(Averagellvalidate(isfinite(Averagellvalidate))));
        end
        BICvalidate(1,M) = -2*Averagellvalidate(1,M)+M*log(N(i));
    end

figure(i), clf,
plot(Averagelltrain,'.b'); hold on; plot(Averagellvalidate,'rx');
xlabel('GMM Number'); ylabel(strcat('Log likelihoos estimate with ',num2str(K),'-fold cross-validation'));
legend('Training Log Likelihood','Validation Log Likelihood');
grid on


figure(i+4), clf,
plot(Averagelltrain,'.b'); hold on; plot(Averagellvalidate,'rx');
xlabel('GMM Number'); ylabel(strcat('BIC estimate with ',num2str(K),'-fold cross-validation'));
legend('Training BIC','Validation BIC');
grid on
%Save graph
%saveas(gcf,'Q1',num2str(i),'.png')

clear
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(Sigma\(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
