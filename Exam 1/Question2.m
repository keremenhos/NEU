clear; close all
% Generate true object location
radius = rand;
angle = 2*pi*rand;
obj_tr = [radius.*cos(angle); radius.*sin(angle)];
% Generate reference locations
rad_ref = 1;
angle_ref = [2*pi pi 2*pi/3 pi/2];
% Variables
K = 4;
xC = linspace(-2,2,1001);
[X, Y] = meshgrid(xC);
val = linspace(0,400,20);
sigma_x = 0.25 + 0.1*randn;
sigma_y = 0.25 + 0.1*randn;

% MAP classifier for each cases of K
for k = 1:K
    % Generate referance points
    obj_ref = [rad_ref.*cos(0:angle_ref(k):1.99*pi); rad_ref.*sin(0:angle_ref(k):1.99*pi)];
    obj_ref(abs(obj_ref)<10^-14 ) = 0; % If number is too small equal to 0
    sigma = repmat(sqrt(0.09), 1, k);
    % True distance
    d = sqrt(sum((repmat(obj_tr,1,k)-obj_ref).^2,1));
    % Measurement noise
    v = normrnd(0,0.3);
    % Range measurement
    ri = d + v;
    % First term of MAP
    fT = (X./sigma_x).^2 + (Y./sigma_y).^2;
    sT = 0;
    % Second term of MAP
    for i = 1:k
        di = sqrt((X-obj_ref(1,i)).^2 + (Y-obj_ref(2,i)).^2);
        sT = sT + ((ri(i)-di)/sigma(i)).^2;
    end
    % Resulting MAP
    MAP = fT + sT;
    
    % Contour plot with true object and reference points
    figure(1)
    subplot(2,2,k)
    plot(obj_tr(1),obj_tr(2),'+'); 
    hold on
    grid on
    plot(obj_ref(1,:),obj_ref(2,:),'o'); 
    axis([-2 2 -2 2])
    contour(X,Y,MAP,val)
    hold off
    axis equal
    legend('True Object','Reference Points','MAP Grid') 
    title(['MAP Estimator for K = ',num2str(k)])
    xlabel('X')
    ylabel('Y')
end
