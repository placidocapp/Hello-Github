clear all;
close all;
clc

%Rede neural para entender os 9 números
%%  %PARAMETROS DA REDE NEURAL
numCamada1 = 34;
numCamada2 = 34;
lambda = 1;

%%  %Recebe os dados
load('data.mat')
X_train = X_train';
X_test = X_test';

%Acrescenta o termo cte
X_train = [ones(size(X_train,1),1) X_train];
X_test = [ones(size(X_test,1),1) X_test];

%Corrige Y para saída binária
y = zeros(size(Y_train,1), 10);
for i=1:size(Y_train,1)
    y(i,Y_train(i,1)+1) = 1;    %0 corresponde a 1 e assim por diante
end
x = X_train;

%%  %Otimiza função de custo

try
    load('OptThetas_34_camadas')
catch

    %PARAMETROS
    maxIter = 4000;
    alpha = 2;
    beta = 0.6;
    eps = 10^-6;
    maxRestart = 1;
    mul = 1;

    %GRADIENTE
    %Inicializa melhores valores
    melhorCusto = 10^10;
    melhorTheta1 = zeros(numCamada1, size(X_train,2));
    melhorTheta2 = zeros(numCamada2,numCamada1+1);
    melhorTheta3 = zeros(10, numCamada2+1);
    for restart = 1:maxRestart
        %Inicializa theta aleatoriamente
        theta1 = sqrt(2/size(X_train,2))*mul*randn(numCamada1, size(X_train,2));
        theta2 = sqrt(2/( numCamada1+1) )*mul*randn(numCamada2,numCamada1+1);
        theta3 = sqrt(2/( numCamada2+1) )*mul*randn(10, numCamada2+1);

        %inicializa momentos
        Vtheta1_ant = 0;
        Vtheta2_ant = 0;
        Vtheta3_ant = 0;
        
        custo = zeros(maxIter+1, 1);  %apenas para plotar no final
        custo(1) = 10^10;
        for i = 2:maxIter
            [custo(i), dtheta1, dtheta2, dtheta3] = ...
                    costFunction(theta1, theta2, theta3, x, y, lambda);
            %Acrescenta o momento
            Vtheta1 = (beta)*Vtheta1_ant + (1-beta)*dtheta1;
            Vtheta2 = (beta)*Vtheta2_ant + (1-beta)*dtheta2;
            Vtheta3 = (beta)*Vtheta3_ant + (1-beta)*dtheta3;
            
            Vtheta1 = Vtheta1/(1-beta^2);
            Vtheta2 = Vtheta2/(1-beta^2);
            Vtheta3 = Vtheta3/(1-beta^2);
            
            %avança o passo
            theta1 = theta1 - alpha*Vtheta1;
            theta2 = theta2 - alpha*Vtheta2;
            theta3 = theta3 - alpha*Vtheta3;
    
            %Imprime na tela
            disp(custo(i));
            disp(i);

            %Caso esteja errado o gradiente
%             if custo(i) > custo(i-1) || custo(i) == custo(i-1)
%                 break;
%             end
            
            %Atualiza o momento
            Vtheta1_ant = Vtheta1;
            Vtheta2_ant = Vtheta2;
            Vtheta3_ant = Vtheta3;

            %Atualiza o melhor gradiente
            if custo(i) < melhorCusto
                melhorCusto = custo(i);
                melhorTheta1 = theta1;
                melhorTheta2 = theta2;
                melhorTheta3 = theta3;
            end
        end
        disp('Reinicio')
        disp(restart)
    end
    
    theta1 = melhorTheta1;
    theta2 = melhorTheta2;
    theta3 = melhorTheta3;

    %salva valores ótimos
    save('OptThetas_34_camadas', 'theta1', 'theta2', 'theta3');
end

%%  %Confere gradiente
% [custo(i), dtheta1, dtheta2, dtheta3] = ...
%                     costFunction(theta1, theta2, theta3, x, y, lambda);
% [Dtheta1, Dtheta2] = ...
%     Gradiente(theta1, theta2, theta3, x, y, dtheta1, dtheta2, dtheta3, lambda);

%%  %Acurácia no grupo de teste

%Inicializa 
y = Y_train;

%Foward Propagation
z1 = x*theta1';
a1 = sigmoid(z1);
a1 = [ones(size(a1,1),1) a1];   %Acrescenta linha de 1's
z2 = a1*theta2';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];   %Acrescenta linha de 1's
z3 = a2*theta3';
a3 = sigmoid(z3);

%Previsão
y_pred = zeros(size(y));
for i=1:size(a3,1)
    [MAX, max_index] = max(a3(i,:));
    y_pred(i) = max_index-1;
end
accuracy_train = sum( (y_pred == y) )/size(y,1)

%%  %Predict

%Inicializa
x = X_test;
y = Y_test;

%Foward Propagation
z1 = x*theta1';
a1 = sigmoid(z1);
a1 = [ones(size(a1,1),1) a1];   %Acrescenta linha de 1's
z2 = a1*theta2';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];   %Acrescenta linha de 1's
z3 = a2*theta3';
a3 = sigmoid(z3);

%Prediz
y_pred = zeros(size(y));
for i=1:size(a3,1)
    [MAX, max_index] = max(a3(i,:));
    y_pred(i) = max_index-1;
end

accuracy_test = sum( (y_pred == y) )/size(y,1)

%%  %Para ver detalhadamente

pause
for i=1:size(y_pred,1)
    disp(y_pred(i))
    imshow(reshape(x(i,2:end),28,28),[0,1]);
    pause
end







