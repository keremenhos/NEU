%% Question 2
rng(444)
clear, clc, close all
%% Data Generation
% Parameters
N=1000; n = 2; K=10;
mu(:,1) = [0;0];
Sigma(:,:,1) = [1 0;0 1]; 
p = [0.35,0.65];
% Label generation
label = rand(1,N) >= p(1); l = 2*(label-0.5);
% Number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N); 
% Radius and angle uniform distribution
rad(:,label==1) = rand(1,length(x(:,label==1)))+2;
ang(:,label==1) = rand(1,length(x(:,label==1)))*2*pi-pi;
[x(1,:),x(2,:)] = pol2cart(ang,rad);
% Generate samples
for lbl = 0:0
    x(:,label==lbl) = randGaussian(Nc(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
end
figure(1)
plot(x(1,label==0),x(2,label==0),'ob'), hold on,
plot(x(1,label==1),x(2,label==1),'+r'), axis equal,
legend('Class -1','Class +1'), 
title('Training Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), hold off
% Generate independent test samples
labelIndep = rand(1,N) >= p(1); lIndep = 2*(labelIndep-0.5);
% Number of samples from each class
NcIndep = [length(find(labelIndep==0)),length(find(labelIndep==1))]; 
xIndep = zeros(n,N); 
% Radius and angle uniform distribution
radv(:,labelIndep==1) = rand(1,length(xIndep(:,labelIndep==1)))+2;
angv(:,labelIndep==1) = rand(1,length(xIndep(:,labelIndep==1)))*2*pi-pi;
[xIndep(1,:),xIndep(2,:)] = pol2cart(angv,radv);
% Generate samples
for lbl = 0:0
    xIndep(:,labelIndep==lbl) = randGaussian(NcIndep(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
end
figure(2)
plot(xIndep(1,labelIndep==0),xIndep(2,labelIndep==0),'ob'), hold on,
plot(xIndep(1,labelIndep==1),xIndep(2,labelIndep==1),'+r'), axis equal,
legend('Class -1','Class +1'), 
title('Validation Data with Real Population/Class Labels'),
xlabel('x_1'), ylabel('x_2'), hold off

%% Linear Kernel
%Train a Linear kernel SVM with cross-validation
%to select hyperparameters that minimize probability 
%of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; 
end
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        % Using fold k as the validation set
        xValidate = x(:,indValidate); 
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        %Using all other folds as the training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        %Labels of validation data using the trained SVM
        dValidate = SVMk.predict(xValidate')'; 
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
figure(3)
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), 
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
% Labels of training data using the trained SVM
d = SVMBest.predict(x')'; 
% Find training samples that are incorrectly classified by the trained SVM
indINCORRECT = find(l.*d == -1); 
% Find training samples that are correctly classified by the trained SVM
indCORRECT = find(l.*d == 1); 
ind00 = find(d==-1 & l==-1); 
ind10 = find(d==1 & l==-1); 
ind01 = find(d==-1 & l==1);
ind11 = find(d==1 & l==1);
figure(4); plot(x(1,ind00),x(2,ind00),'og'); hold on; plot(x(1,ind10),x(2,ind10),'or'); 
plot(x(1,ind01),x(2,ind01),'+r'); plot(x(1,ind11),x(2,ind11),'+g'); axis equal,
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
legend('Incorrect Class -1','Correct Class +1'), 
title('Training Data with Decisions'),
hold off

% Independent data
dindep = SVMBest.predict(xIndep')'; 
indINCORRECTv = find(lIndep.*dindep == -1); 
indCORRECTv = find(lIndep.*dindep == 1);
pTrainingErrorv = length(indINCORRECTv)/N,
ind00 = find(dindep==-1 & lIndep==-1); 
ind10 = find(dindep==1 & lIndep==-1); 
ind01 = find(dindep==-1 & lIndep==1);
ind11 = find(dindep==1 & lIndep==1);
figure(5); plot(xIndep(1,ind00),xIndep(2,ind00),'og'); hold on; plot(xIndep(1,ind10),xIndep(2,ind10),'or');
plot(xIndep(1,ind01),xIndep(2,ind01),'+r'); plot(xIndep(1,ind11),xIndep(2,ind11),'+g'); axis equal,
legend('Incorrect Class -1','Correct Class +1'), 
title('Validation Data with Decisions'),
hold off

%% Gaussian Kernel
%Train a Gaussian kernel SVM with cross-validation
%to select hyperparameters that minimize probability 
%of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            %Using fold k as validation set
            xValidate = x(:,indValidate); 
            lValidate = l(indValidate);  
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
            end
            %Using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            %Labels of validation data using the trained SVM
            dValidate = SVMk.predict(xValidate')'; 
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end

figure(6),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
% Labels of training data using the trained SVM
d = SVMBest.predict(x')'; 
% Find training samples that are incorrectly classified by the trained SVM
indINCORRECT = find(l.*d == -1); 
% Find training samples that are correctly classified by the trained SVM
indCORRECT = find(l.*d == 1); 
ind00 = find(d==-1 & l==-1); 
ind10 = find(d==1 & l==-1); 
ind01 = find(d==-1 & l==1);
ind11 = find(d==1 & l==1);

figure(7); plot(x(1,ind00),x(2,ind00),'og'); hold on; plot(x(1,ind10),x(2,ind10),'or');
plot(x(1,ind01),x(2,ind01),'+r'); plot(x(1,ind11),x(2,ind11),'+g'); axis equal;
pTrainingError = length(indINCORRECT)/N, %Empirical estimate of training error probability 
% Grid search (unnecessary for linear kernels)
Nx = 10100/2; Ny = 9900/2; xGrid = linspace(-4,4,Nx); yGrid = linspace(-4,4,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(7), contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Correct Class -1','Incorrect Class -1','Incorrect Class +1','Correct Class +1','Equilevel contours of the discriminant function' ), 
title('Training Data with Decisions'); hold off

% Independent data
dindep = SVMBest.predict(xIndep')'; 
indINCORRECTv = find(lIndep.*dindep == -1); 
indCORRECTv = find(lIndep.*dindep == 1);
pTrainingErrorv = length(indINCORRECTv)/N,
ind00 = find(dindep==-1 & lIndep==-1); 
ind10 = find(dindep==1 & lIndep==-1); 
ind01 = find(dindep==-1 & lIndep==1);
ind11 = find(dindep==1 & lIndep==1);
figure(8); plot(xIndep(1,ind00),xIndep(2,ind00),'og'); hold on; plot(xIndep(1,ind10),xIndep(2,ind10),'or');
plot(xIndep(1,ind01),xIndep(2,ind01),'+r'); plot(xIndep(1,ind11),xIndep(2,ind11),'+g'); axis equal;
%Grid search (unnecessary for linear kernels)
Nx = 10100/2; Ny = 9900/2; xGrid = linspace(-4,4,Nx); yGrid = linspace(-4,4,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(8),contour(xGrid,yGrid,zGrid,10); xlabel('x1'), ylabel('x2'), axis equal,
legend('Correct Class -1','Incorrect Class -1','Incorrect Class +1','Correct Class +1','Equilevel contours of the discriminant function' ), 
title('Validation Data with Decisions'); hold off

%% Functions
function x = randGaussian(N,mu,Sigma)
%Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
