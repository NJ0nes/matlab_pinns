function u = actSigmoid(v)
    u = 2 * sigmoid(v) - 1; % Rescale so that we have symmetry
end