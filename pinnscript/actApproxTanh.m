% J. M. ain't a mathematician
% (https://math.stackexchange.com/users/498/j-m-aint-a-mathematician),
% Rapid approximation of $\tanh(x)$, URL (version: 2012-02-09):
% https://math.stackexchange.com/q/107666

function u = actApproxTanh(v)
    u = (v.*(10 + v.*v).*(60 + v.*v));
    u = u ./ (600 + v.*v.*(270 + v.*v.*(11 + v.*v.*(1.0 / 24.0))));
end