clear

% ====== CONFIGURATION
dt = 0.01; % Time step
dx = 0.01; % Space step
final_time = 1.0;
viscosity = 0.02 / pi;

% ====== FINITE DIFFERENCE METHOD
tic
points = -1:dx:1;
times = 0:dt:final_time;

% Approximate solution at all times and points
u = zeros(length(points), length(times));
u(:, 1) = initial_cond(points);

for t = 1:(length(times) - 1)
    % Boundary is already set, so only update interior points
    for x = 2:length(points) - 1
        u(x, t+1) = u(x, t) ...
            + viscosity * dt / dx^2 * (u(x-1, t) + u(x+1, t) - 2*u(x, t)) ...
            - dt * u(x, t) * (u(x+1, t) - u(x-1, t)) / (2 * dx);
    end
end

toc
fprintf("Finished computing u");

% Create an animation showing the system evolving in time
M(length(times)) = struct('cdata',[],'colormap',[]);
h = figure;
h.Visible = "off";
ax = gca;
ax.NextPlot = 'replacechildren';
ax.XLimMode = "manual";
ax.YLimMode = "manual";
ax.XLim = [-1, 1];
ax.YLim = [-2, 2];

for j = 1:length(times)
    truevalues = solveBurgers(points, times(j), viscosity);
    error = norm(u(:,j)' - truevalues) / norm(truevalues);
    plot(points, u(:,j), "-")
    xlabel("x")
    ylabel("u")
    title(strcat("u(x, t = ", string(times(j)), "), Error = ", string(error)));
    drawnow
    M(j) = getframe(gcf);
end

v = VideoWriter("burger_x01_t01_v02_error_newic.avi");
v.Quality = 95;
v.FrameRate = ceil(length(times) / 12);
open(v)
writeVideo(v, M);
close(v)

function y = initial_cond(x)
    y = -sin(pi * x);
end

% === Analytic Solution
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