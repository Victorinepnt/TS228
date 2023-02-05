function decision = weakClassif(x, theta, p)
% WEAKCLASSIF Return the decision of the weak classifier
if p == 1
    if x < theta
        decision = 1;
    else
        decision = -1;
    end
elseif p == -1
    if x > theta
        decision = 1;
    else
        decision = -1;
    end
end
end