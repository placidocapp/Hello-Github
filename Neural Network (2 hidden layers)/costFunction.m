function [ custo, dtheta1, dtheta2, dtheta3 ] = costFunction( theta1, theta2,theta3, x, y, lambda )
%A função calcula o custo e o gradiente dado theta, x e y
%A função utiliza custo logarítmo e um sigmoid para calcular o custo e
%gradiente, a função recebe theta, x, y de 1 a 9 para cada imagem x e nn
%que contém um vetor com o número de nos em cada camada 
% (e.g. nn = [10 10]).

%%  %inicialização
m = size(x,1);  %número de amostras

%%  %Foward Propagation

z1 = x*theta1';
a1 = sigmoid(z1);
a1 = [ones(size(a1,1),1) a1];   %Acrescenta linha de 1's
z2 = a1*theta2';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];   %Acrescenta linha de 1's
z3 = a2*theta3';
a3 = sigmoid(z3);

%%  %Custo da rede neural

custo = (1/m)*sum(sum( -y.*log(a3) - (1-y).*log(1-a3) ) ) + ...
    ( lambda/(2*m) )*sum(sum(theta1(:,2:end).^2)) + ...
    ( lambda/(2*m) )*sum(sum(theta2(:,2:end).^2)) + ...
    ( lambda/(2*m) )*sum(sum(theta3(:,2:end).^2));

%%  %Back   Propagation

dz3 = a3 - y;               %60000x10
dtheta3 = (1/m)*dz3'*a2;    %10x11 = 1*(60000x10)'*(60000x11)
a = sigmoid(z2);
dz2 = dz3*theta3(:,2:end).*a.*(1-a);           %
dtheta2 = (1/m)*dz2'*a1;    %
a = sigmoid(z1);
dz1 = dz2*theta2(:,2:end).*a.*(1-a);           %(60000x11) = (60000x10)*(10x11)
dtheta1 = (1/m)*dz1'*x;%(10x785) = 1*(60000x10)'*(60000x785)

%acrescentando a regularização
dtheta1 = dtheta1 + (lambda/m)*[zeros(size(theta1,1),1) theta1(:,2:end)];
dtheta2 = dtheta2 + (lambda/m)*[zeros(size(theta2,1),1) theta2(:,2:end)];
dtheta3 = dtheta3 + (lambda/m)*[zeros(size(theta3,1),1) theta3(:,2:end)];

end

