function [ y ] = sigmoid( x )
%Calcula o sigmoid de qualquer matriz numero a numero
%   Detailed explanation goes here

y = zeros(size(x));

for i = 1:size(x,1)
    for j = 1:size(x,2)
        y(i,j) = 1/(1+exp(-x(i,j)));
    end
end

end

