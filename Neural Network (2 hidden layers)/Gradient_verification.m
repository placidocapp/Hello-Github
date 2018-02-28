function [ Dtheta1, Dtheta2 ] = Gradiente( theta1, theta2, theta3, x, y,... 
        dtheta1, dtheta2, dtheta3, lambda )
%Calcula o gradiente por aproximação numérica para conferir o resultado do
%back propagation
%   Detailed explanation goes here
num_testes = 4;
eps = 10^-4;
Dtheta1 = zeros(size(theta1(:,1:num_testes)));
Dtheta2 = zeros(size(theta2(:,1:num_testes)));
Dtheta3 = zeros(size(theta3(:,1:num_testes)));


% for i = 1:size(theta1, 1)
%     for j = 1:num_testes
%         %com o custo em mãos variamos eps e calculamos a derivada numérica
%         aux = custo(theta1, theta2, theta3, x, y, lambda);
%         theta1(i,j) = theta1(i,j) + eps;    %soma eps 
%         aux = custo(theta1, theta2, theta3, x, y, lambda) - aux;
%         Dtheta1(i,j) = aux/eps;             %retorna antes do eps
%         theta1(i,j) = theta1(i,j) - eps;
%     
%         disp(norm(Dtheta1(i,j) - dtheta1(i,j),2)/...
%                     (norm(Dtheta1(i,j),2) + norm(dtheta1(i,j),2)));
%     end
% end

% for i = 1:size(theta2, 1)
%     for j = 1:num_testes
%         %com o custo em mãos variamos eps e calculamos a derivada numérica
%         aux = custo(theta1, theta2, theta3, x, y, lambda);
%         theta2(i,j) = theta2(i,j) + eps;    %soma eps 
%         aux = custo(theta1, theta2, theta3, x, y, lambda) - aux;
%         Dtheta2(i,j) = aux/eps;             %retorna antes do eps
%         theta2(i,j) = theta2(i,j) - eps;
%        disp(norm(Dtheta2(i,j) - dtheta2(i,j),2)/...
%                     (norm(Dtheta2(i,j),2) + norm(dtheta2(i,j),2)));
%     end
% end

for i = 1:size(theta3, 1)
    for j = 1:num_testes
        %com o custo em mãos variamos eps e calculamos a derivada numérica
        aux = custo(theta1, theta2, theta3, x, y, lambda);
        theta3(i,j) = theta3(i,j) + eps;    %soma eps 
        aux = custo(theta1, theta2, theta3, x, y, lambda) - aux;
        Dtheta3(i,j) = aux/eps;             %retorna antes do eps
        theta3(i,j) = theta3(i,j) - eps;
       disp(norm(Dtheta3(i,j) - dtheta3(i,j),2)/...
                    (norm(Dtheta3(i,j),2) + norm(dtheta3(i,j),2)));
    end
end

% %%  %Compara gradientes
% 
% aux1 = dtheta1(:,1:num_testes) - Dtheta1;
% aux2 = dtheta2(:,1:num_testes) - Dtheta2;
% 
% disp(max(abs(aux1)))
% disp(max(abs(aux2)))

end

