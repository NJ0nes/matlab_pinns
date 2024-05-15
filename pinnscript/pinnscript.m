parpool(24)

parfor jobID = 1:length(experiment_list)
    
    p = experiment_list(jobID);
    % Training Data =============
    
    % Boundary Conditions
    
    x0BC1 = -1*ones(1,p.numBoundaryConditionPoints);
    x0BC2 = ones(1,p.numBoundaryConditionPoints);
    
    t0BC1 = linspace(0,1,p.numBoundaryConditionPoints);
    t0BC2 = linspace(0,1,p.numBoundaryConditionPoints);
    
    u0BC1 = zeros(1,p.numBoundaryConditionPoints);
    u0BC2 = zeros(1,p.numBoundaryConditionPoints);
    
    % Initial Conditions
    
    x0IC = linspace(-1,1,p.numInitialConditionPoints);
    t0IC = zeros(1,p.numInitialConditionPoints);
    u0IC = -sin(pi*x0IC);
    
    X0 = [x0IC x0BC1 x0BC2];
    T0 = [t0IC t0BC1 t0BC2];
    U0 = [u0IC u0BC1 u0BC2];
    
    % Collocation Points
    points = p.collocationFn(p.numInternalCollocationPoints);
    
    % Points are generated from [0, 1]^2 so we need to rescale
    dataX = 2*points(:,1)-1;
    dataT = points(:,2);
    
    ds = arrayDatastore([dataX dataT]);
    
    % Define Deep Learning Model ==============
    
    parameters = struct;
    
    sz = [p.numNeurons 2];
    parameters.fc1.Weights = initializeHe(sz,2);
    parameters.fc1.Bias = initializeZeros([p.numNeurons 1]);
    parameters.fc1.Learnable = dlarray([1.0]);

    for layerNumber=2:p.numLayers-1
        name = "fc"+layerNumber;
    
        sz = [p.numNeurons p.numNeurons];
        numIn = p.numNeurons;
        parameters.(name).Weights = initializeHe(sz,numIn);
        parameters.(name).Bias = initializeZeros([p.numNeurons 1]);
        parameters.(name).Learnable = dlarray([1.0]);
    end
    
    sz = [1 p.numNeurons];
    numIn = p.numNeurons;
    parameters.("fc" + p.numLayers).Weights = initializeHe(sz,numIn);
    parameters.("fc" + p.numLayers).Bias = initializeZeros([1 1]);
    parameters.("fc" + p.numLayers).Learnable = dlarray([1.0]);
    
    % Training ==================
    
    executionEnvironment = "auto";
    
    mbq = minibatchqueue(ds, ...
        MiniBatchSize=p.miniBatchSize, ...
        MiniBatchFormat="BC", ...
        OutputEnvironment=executionEnvironment);
    
    X0 = dlarray(X0,"CB");
    T0 = dlarray(T0,"CB");
    U0 = dlarray(U0);
    
    averageGrad = [];
    averageSqGrad = [];
    
    accfun = dlaccelerate(@modelLoss);
    
    loss_history = zeros(p.numEpochs, 1);
    start = tic;
    
    iteration = 0;
    loss = 0.0;
    
    for epoch = 1:p.numEpochs
        reset(mbq);
    
        while hasdata(mbq)
            iteration = iteration + 1;
    
            XT = next(mbq);
            X = XT(1,:);
            T = XT(2,:);
    
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients] = dlfeval(accfun,parameters,X,T,X0,T0,U0, p);
    
            % Update learning rate.
            learningRate = p.initialLearnRate / (1+p.decayRate*iteration);
    
            % Update the network parameters using the adamupdate function.
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,learningRate);
        end
    
        loss = double(gather(extractdata(loss)));
        loss_history(epoch) = loss;
    end
    
    runTime = toc(start);
    
    tTest = [0.25 0.5 0.75 1];
    numPredictions = 1001;
    XTest = linspace(-1,1,numPredictions);
    
    l2errors = zeros(4, 1);
    maxerrors = zeros(4, 1);
    for i=1:numel(tTest)
        t = tTest(i);
        TTest = t*ones(1,numPredictions);
    
        % Make predictions.
        XTest = dlarray(XTest,"CB");
        TTest = dlarray(TTest,"CB");
        UPred = model(parameters,XTest,TTest, p);
    
        % Calculate true values.
        UTest = solveBurgers(extractdata(XTest),t, p.viscosity);
    
        % Calculate error.
        l2errors(i) = norm(extractdata(UPred) - UTest) / norm(UTest);
        maxerrors(i) = norm(extractdata(UPred) - UTest, inf);
    end
    
    % ====== Output a line of CSV
    param_part = string([jobID, p.label, p.viscosity, p.numEpochs, p.miniBatchSize, ...
        p.weightF, p.weightU, p.numBoundaryConditionPoints, ...
        p.numInitialConditionPoints, p.numInternalCollocationPoints, ...
        p.numLayers, p.numNeurons, p.initialLearnRate, p.decayRate]);
    
    names_part = [p.activationName, p.collocationName];
    
    result_part = [runTime, accfun.HitRate, l2errors', maxerrors', loss_history']';
    result_fmt = "%.8f";
    result_strings = compose(result_fmt, result_part);
    
    % The CSV file has room for 3000 epochs, but we might not run
    % for that long, so we need to pad the length
    missing_epochs = repelem("", 3000 - p.numEpochs);
    
    out_string = join([param_part, names_part, result_strings', ...
        missing_epochs], ",");
    
    outfile = fopen("outputfolder/job" + string(jobID) + ".csv", "w");
    fprintf(outfile, out_string);
    fclose(outfile);
    
    % We also want to save a copy of the parameters for further analysis
    parsave("outputfolder/job" + string(jobID) + "parameters.mat", parameters);

end


% ====== Auxillary Functions
function parsave(fname, parameters)
    save(fname, "parameters");
end

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

function [loss,gradients] = modelLoss(parameters,X,T,X0,T0,U0, p)
% Make predictions with the initial conditions.
U = model(parameters,X,T, p);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - p.viscosity.*Uxx;
zeroTarget = zeros(size(f), "like", f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.
U0Pred = model(parameters,X0,T0, p);
lossU = mse(U0Pred, U0);

% Combine losses.
loss = p.weightF * lossF + p.weightU * lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

function U = model(parameters,X,T,p)

XT = [X;T];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
U = fullyconnect(XT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    U = p.activationFn(U, parameters.(name).Learnable);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    U = fullyconnect(U, weights, bias);
end

end
