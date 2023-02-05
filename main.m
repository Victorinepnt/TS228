clear,
close all,
clc;

%Récupération de la base de données
iris_data=iris_dataset;
iris_data=iris_data(:,1:100);
iris_species= [zeros(51,1);ones(49,1)]';


%Initialisation
d = 28; %d = 784
N = 28; %N = 20000
N_test = size(iris_data,1); %N_test = 10000
c=10;               %Définir qui 
T=250;              %Définir qui
N = size(iris_data,1); %Nombre d'images
t=[-1 1];           %Cible
w=zeros(10,N);      %Poids
w(:,1)=1/N;         %Initialisation du premier poids
J = zeros(d,102);   %Initialisation de la somme des poids des erreurs
alpha = ones(c,2,T);  %On initialise alpha
err=zeros(c,T);
y=zeros(c,T);       %Prédiction binaire

g = zeros(c,N); %g(x): 10x20000
g_test = zeros(c,N_test); %g(x): 10x10000
err_train_binary = zeros(c,T); err_train = zeros(1,T);
err_test_binary = zeros(c,T); err_test = zeros(1,T);
largestWeight_id = zeros(c,T); %largest weight indexes: 10x250
gammas = zeros(c,N,5); %margin: 10x20000x250
gamma_count = 0;
wt = zeros(c,T); %initial step size: 10x250



test=iris_dataset;


%Algorithme

for it = 1:T
  
    for k = 1:c
        gk = g(k,:);
        gk_test = g_test(k,:);
        wl = it;
        jBest = alpha(k,1,wl);
        uBest = alpha(k,2,wl);
        iris_data = X(jBest,:); %1x20000
        xj_test = X_test(jBest,:); %1x10000    Chercher ce par quoi on doi
         
        %update the learned function
        if uBest <= 51
            n = uBest - 1;
            t = n/50;
            gk = gk + wt(k,wl)*u(xj,t);
            gk_test = gk_test + wt(k,wl)*u(xj_test,t);
        else
            n = uBest - 52;
            t = n/50;
            gk = gk + wt(k,wl)*(-u(xj,t));
            gk_test = gk_test + wt(k,wl)*(-u(xj_test,t));
        end
        g(k,:) = gk; %1x20000
        g_test(k,:) = gk_test; %1x10000
    
        %compute the weights
        yk = Y(k,:);
        wk = exp(-yk.*gk); %1x20000
        w(k,:) = wk;
        
        %compute the negative gradient
        for j = 1:d
            xj = X(j,:);
            for n = 0:50
                t = n/50;

                ux = u(xj,t); %1x20000
                ydiffu_id = yk~=ux; %find i | yi!=u(xi)
                ydiffutwin_id = yk~=-ux;
                
                J(j,n+1) = sum(wk(ydiffu_id));
                J(j,n+52) = sum(wk(ydiffutwin_id));
            end
        end
        minW = min(W(:));
        [jBest,uBest] = find(W==minW,1);
        alpha(k,:,it+1) = [jBest,uBest];
        
        %compute the step size
        ep = W(jBest,uBest)/sum(wk);
        wt(k,it+1) = 1/2*log((1-ep)/ep);
    end
    
end