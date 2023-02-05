function [alpha, thetaval, dim, pval, error] = best_h(X,Y, wx)
%BEST_H Return the best weak classifier (decision stump)
n = size(X,1);
m = size(X,2);
p = [-1 1]';

% Creating points between samples as decision stump, for each dimension
for i=1:m
    DimSorted = sort(X(:,i));
    theta(:,i) = [DimSorted(1) - 1; DimSorted(2:end) - diff(DimSorted) / 2; DimSorted(size(DimSorted,1)) + 1];
end


for i=1:size(p,1) % Going though all created decision stumps
    for j=1:size(theta,2)
        for k=1:size(theta,1)
            Xh = [];
            for l=1:n % Each decision stump classify each sample
                Xh = [Xh; weakClassif(X(l,j),theta(k,j),p(i))];
            end
            % Get the error of the decision stump
            errors(k,j,i) = dot(wx',Xh ~= cell2mat(Y));
        end
    end
end

% Find smallest error, this is the weak classifier to return
error = min(reshape(errors,[],1));

% Returning 
% - alpha,
% - position of decision stump, 
% - dimension of sample the stump analyses,
% - value of p, 
% - error

[k,j,i] = ind2sub(size(errors),find(errors == error, 1));

pval = p(i);
thetaval = theta(k,j);
dim = j;
alpha = 0.5*log((1-error)/error);