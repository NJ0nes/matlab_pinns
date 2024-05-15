% Hayou, S., Doucet, A., & Rousseau, J. (2018). "On the selection of initialization
%   and activation functions for deep neural networks." arXiv preprint arXiv:1805.08266.

function parameter = initializeEOCb(sz,numIn,className)

arguments
    sz
    numIn
    className = 'single'
end

parameter = 0.2 * randn(sz,className);
parameter = dlarray(parameter);

end
