clear, close all, clc
% Chargement des données Iris


load fisheriris;

% iris_data=iris_dataset;
% iris_data=iris_data(:,1:100);
% iris_species= [zeros(51,1);ones(49,1)]';

% Séparation en ensemble d'entraînement et de test
rng(1); % pour reproductibilité
indices = crossvalind('Kfold', species, 5);
test = (indices == 1); 
train = ~test;

% Entraînement du classificateur Adaboost
Mdl = fitensemble(meas(train,:), species(train), 'AdaBoostM2', 100, 'Tree');

% Prédiction sur l'ensemble de test
y_pred = predict(Mdl, meas(test,:));

% Calcul de la précision
accuracy = sum(strcmp(y_pred, species(test))) / sum(test)
