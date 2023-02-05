clear,
close all,
clc,

load fisheriris.mat;

X = meas;
Y = species';
n = size(X,1);
m = size(X,2);

Nit = 5; % nombre d'itérations
weak_classifiers = []; % contient les paramètres des classifieurs faibles
errors = []; % vecteur contenant les erreurs du classifieur fort à chaque itération
Wx = 1/n * ones(n,1); % poids des exemples

for i=1:Nit % i itérations
    [alpha, theta, dim, p, error] = best_h(X,Y,Wx); %recupère meilleur classifieur faible
    weak_classifiers = [weak_classifiers; alpha, theta, dim, p, error];
    
    for j=1:n % MAJ poids des exemples
    Wx(j) = (Wx(j) * exp(-alpha*Y(j)*weakClassif(X(j, dim),theta,p))) / (2*sqrt(error*(1-error)));
    end
    
    % calcule erreur du classifieur fort
    error = 0;
    for j = 1:n
    error = error + (strongClassif(X(j,:),weak_classifiers) ~= Y(j)) / n;
    end
    errors = [errors; error];
end