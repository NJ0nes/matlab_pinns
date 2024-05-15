function u = actSwish(v, l)
    u = v .* sigmoid(l * v);
end
