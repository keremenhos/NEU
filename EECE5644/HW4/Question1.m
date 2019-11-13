%% Question 1
rng(444)
clear, clc, close all
%% K-Means
% Importing images
Iplane = imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg');
Ibird = imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg');
% Convert to grayscale (values between 0-1)
plane = zeros(size(Iplane));
bird = zeros(size(Ibird));
for i=1:3
    plane(:,:,i) = mat2gray(Iplane(:,:,i));
    bird(:,:,i) = mat2gray(Ibird(:,:,i));
end
% Normalize horizontal and vertical axes
x = (0:1:size(plane,1)-1)/(size(plane,1)-1);
y = (0:1:size(plane,2)-1)/(size(plane,2)-1);
% Create mesh grid with normalized axes
[X, Y] = meshgrid(x, y);
% Load mesh grid on matrices
plane(:,:,4) = X';
plane(:,:,5) = Y';
bird(:,:,4) = X';
bird(:,:,5) = Y';
% Total number of pixels
totPixel = 481*321;
% Reshape into vector
plane = reshape(plane, [totPixel,5]);
bird = reshape(bird, [totPixel,5]);
% k-means function
[indP2,~] = kmeans(plane,2);
[indP3,~] = kmeans(plane,3);
[indP4,~] = kmeans(plane,4);
[indP5,~] = kmeans(plane,5);
[indB2,~] = kmeans(bird,2);
[indB3,~] = kmeans(bird,3);
[indB4,~] = kmeans(bird,4);
[indB5,~] = kmeans(bird,5);
% Turn back to image form
indP2 = reshape(indP2, [321,481]);
indP3 = reshape(indP3, [321,481]);
indP4 = reshape(indP4, [321,481]);
indP5 = reshape(indP5, [321,481]);
indB2 = reshape(indB2, [321,481]);
indB3 = reshape(indB3, [321,481]);
indB4 = reshape(indB4, [321,481]);
indB5 = reshape(indB5, [321,481]);
% Visualization
figure(1); imagesc(indP2); figure(2); imagesc(indP3); figure(3); imagesc(indP4); figure(4); imagesc(indP5)
figure(5); imagesc(indB2); figure(6); imagesc(indB3); figure(7); imagesc(indB4); figure(8); imagesc(indB5)

%% GMM + MAP
% First Image
% K = 2
gmm = fitgmdist(plane,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i)
        decision(i) = 1;
    elseif g2(i) < g1(i)
        decision(i) = 2;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(9); imagesc(decisioN)

% K = 3
gmm = fitgmdist(plane,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i)<g2(i) && g1(i)<g3(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) && g2(i)<g3(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) && g3(i)<g2(i)
        decision(i) = 3;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(10); imagesc(decisioN)

% K = 4
gmm = fitgmdist(plane,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i) && g1(i) < g3(i) && g1(i) < g4(i)
        decision(i) = 1;
    elseif g2(i) < g1(i) && g2(i) < g3(i) && g2(i) < g4(i)
        decision(i) = 2;
    elseif g3(i) < g1(i) && g3(i) < g2(i) && g3(i) < g4(i)
        decision(i) = 3;
    elseif g4(i) < g1(i) && g4(i) < g2(i) && g4(i) < g3(i)
        decision(i) = 4;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(11); imagesc(decisioN)

% K = 5
gmm = fitgmdist(plane,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(plane',m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(plane',m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(plane',m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(plane',m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(plane',m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(plane',m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(plane',m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(plane',m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(plane',m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i) && g1(i) < g3(i) && g1(i) < g4(i) && g1(i) < g5(i)
        decision(i) = 1;
    elseif g2(i) < g1(i) && g2(i) < g3(i) && g2(i) < g4(i) && g2(i) < g5(i)
        decision(i) = 2;
    elseif g3(i) < g1(i) && g3(i) < g2(i) && g3(i) < g4(i) && g3(i) < g5(i)
        decision(i) = 3;
    elseif g4(i) < g1(i) && g4(i) < g2(i) && g4(i) < g3(i) && g4(i) < g5(i)
        decision(i) = 4;
    elseif g5(i) < g1(i) && g5(i) < g2(i) && g5(i) < g3(i) && g5(i) < g4(i)
        decision(i) = 5;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(12); imagesc(decisioN)

% Second Image
% K = 2
gmm = fitgmdist(bird,2);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1;1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2);
g2 = lambda(2,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i)
        decision(i) = 1;
    elseif g2(i) < g1(i)
        decision(i) = 2;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(13); imagesc(decisioN)

% K = 3
gmm = fitgmdist(bird,3);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1;1 0 1;1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3);
g2 = lambda(2,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3);
g3 = lambda(3,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i) && g1(i) < g3(i)
        decision(i) = 1;
    elseif g2(i) < g1(i) && g2(i) < g3(i)
        decision(i) = 2;
    elseif g3(i) < g1(i) && g3(i) < g2(i)
        decision(i) = 3;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(14); imagesc(decisioN)

% K = 4
gmm = fitgmdist(bird,4);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i) && g1(i) < g3(i) && g1(i) < g4(i)
        decision(i) = 1;
    elseif g2(i) < g1(i) && g2(i) < g3(i) && g2(i) < g4(i)
        decision(i) = 2;
    elseif g3(i) < g1(i) && g3(i) < g2(i) && g3(i) < g4(i)
        decision(i) = 3;
    elseif g4(i) < g1(i) && g4(i) < g2(i) && g4(i) < g3(i)
        decision(i) = 4;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(15); imagesc(decisioN)

% K = 5
gmm = fitgmdist(bird,5);
Sigma = gmm.Sigma;
m = gmm.mu';
p = gmm.ComponentProportion;
% Loss values (0-1 for this error minimization)
lambda = [0 1 1 1 1;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1;1 1 1 1 0]; 
% Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4) + lambda(1,5)*evalGaussian(bird',m(:,5),Sigma(:,:,5))*p(5);
g2 = lambda(2,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4) + lambda(2,5)*evalGaussian(bird',m(:,5),Sigma(:,:,5))*p(5);
g3 = lambda(3,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4) + lambda(3,5)*evalGaussian(bird',m(:,5),Sigma(:,:,5))*p(5);
g4 = lambda(4,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4) + lambda(4,5)*evalGaussian(bird',m(:,5),Sigma(:,:,5))*p(5);
g5 = lambda(5,1)*evalGaussian(bird',m(:,1),Sigma(:,:,1))*p(1) + lambda(5,2)*evalGaussian(bird',m(:,2),Sigma(:,:,2))*p(2) + lambda(5,3)*evalGaussian(bird',m(:,3),Sigma(:,:,3))*p(3) + lambda(5,4)*evalGaussian(bird',m(:,4),Sigma(:,:,4))*p(4) + lambda(5,5)*evalGaussian(bird',m(:,5),Sigma(:,:,5))*p(5);
decision = zeros(1,totPixel); 
for i = 1:totPixel
    if g1(i) < g2(i) && g1(i) < g3(i) && g1(i) < g4(i) && g1(i) < g5(i)
        decision(i) = 1;
    elseif g2(i) < g1(i) && g2(i) < g3(i) && g2(i) < g4(i) && g2(i) < g5(i)
        decision(i) = 2;
    elseif g3(i) < g1(i) && g3(i) < g2(i) && g3(i) < g4(i) && g3(i) < g5(i)
        decision(i) = 3;
    elseif g4(i) < g1(i) && g4(i) < g2(i) && g4(i) < g3(i) && g4(i) < g5(i)
        decision(i) = 4;
    elseif g5(i) < g1(i) && g5(i) < g2(i) && g5(i) < g3(i) && g5(i) < g4(i)
        decision(i) = 5;
    end
end
% Visualization
decisioN = reshape(decision, [321,481]);
figure(16); imagesc(decisioN)

function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(Sigma\(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end
