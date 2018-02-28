function [ custo ] = custo( theta1, theta2, theta3, x, y , lambda)
%UNTITLED21 Summary of this function goes here
%   Detailed explanation goes here
 %%  %inicialização
    m = size(x,1);  %número de amostras

    %Foward Propagation
    z1 = x*theta1';
    a1 = sigmoid(z1);
    a1 = [ones(size(a1,1),1) a1];   %Acrescenta linha de 1's
    z2 = a1*theta2';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1),1) a2];   %Acrescenta linha de 1's
    z3 = a2*theta3';
    a3 = sigmoid(z3);
    
    %Custo da rede neural
    custo = (1/m)*sum(sum( -y.*log(a3) - (1-y).*log(1-a3) ) ) + ...
    ( lambda/(2*m) )*sum(sum(theta1(:,2:end).^2)) + ...
    ( lambda/(2*m) )*sum(sum(theta2(:,2:end).^2)) + ...
    ( lambda/(2*m) )*sum(sum(theta3(:,2:end).^2));

end

