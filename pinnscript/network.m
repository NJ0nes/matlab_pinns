% Training Data =============

% Boundary Conditions

numBoundaryConditionPoints = [25 25];

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));

% Initial Conditions

numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];

% Collocation Points

numInternalCollocationPoints = 10000;

pointSet = sobolset(2);
points = net(pointSet,numInternalCollocationPoints);

dataX = 2*points(:,1)-1;
dataT = points(:,2);

ds = arrayDatastore([dataX dataT]);

% Define Deep Learning Model ==============

numLayers = 9;
numNeurons = 20;

parameters = struct;

sz = [numNeurons 2];
parameters.fc1.Weights = initializeHe(sz,2);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);

% Training ==================

numEpochs = 3000;
miniBatchSize = 1000;

executionEnvironment = "auto";

initialLearnRate = 0.01;
decayRate = 0.005;

mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);

X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0);

averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelLoss);

figure
C = colororder;
lineLoss = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

start = tic;

iteration = 0;

for epoch = 1:numEpochs
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        XT = next(mbq);
        X = XT(1,:);
        T = XT(2,:);

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(accfun,parameters,X,T,X0,T0,U0);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end

tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);

figure

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);

    % Make predictions.
    XTest = dlarray(XTest,"CB");
    TTest = dlarray(TTest,"CB");
    UPred = model(parameters,XTest,TTest);

    % Calculate true values.
    UTest = solveBurgers(extractdata(XTest),t,0.01/pi);

    % Calculate error.
    err = norm(extractdata(UPred) - UTest) / norm(UTest);

    % Plot predictions.
    subplot(2,2,i)
    plot(XTest,extractdata(UPred),"-",LineWidth=2);
    ylim([-1.1, 1.1])

    % Plot true values.
    hold on
    plot(XTest, UTest, "--",LineWidth=2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

subplot(2,2,2)
legend("Predicted","True")

function U = solveBurgers(X,t,nu)

% Define functions.
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));

% Initialize solutions.
U = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);

    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
    if abs(x) ~= 1
        fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
        uxt = -integral(fun,-inf,inf);
        fun = @(eta) f(x-eta) .* g(eta);
        U(i) = uxt / integral(fun,-inf,inf);
    end
end

end

function [loss,gradients] = modelLoss(parameters,X,T,X0,T0,U0)

% Make predictions with the initial conditions.
U = model(parameters,X,T);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.
U0Pred = model(parameters,X0,T0);
lossU = mse(U0Pred, U0);

% Combine losses.
loss = lossF + lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

function U = model(parameters,X,T)

XT = [X;T];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
U = fullyconnect(XT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    U = tanh(U);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    U = fullyconnect(U, weights, bias);
end

end