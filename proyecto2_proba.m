%  Proyecto 2 - Temas selectos de Probabilidad
% 1er Examen Parccial: Segmentación Bayesiana
% --------------------------------------- %
clc; close all;

% PRIMERA PARTE: Obtencion de Parámetros
% --------------------------------------- %

% 1.- Definiendo 5 ventanas que identifiquen 5 regiones típicas sobre Paris.bmp
I = imread('Paris.bmp');
sz = size(I); % Tamaño de la imagen en pixeles
x = 0:255;

%histogram(I)

ventana1 = [214 427 227-214 466-427]; %[x1, y1, ancho, alto] zona de agua
ventana2 = [270 390 284-270 396-390]; %[x1, y1, ancho, alto] zona sombra
ventana3 = [19 151 36-19 162-151]; %[x1, y1, ancho, alto] zona vegetación
ventana4 = [463 247 471-459 253-247]; %[x1, y1, alto, alto] zona edificios
ventana5 = [262 194 329-262 252-194]; %[x1, y1, ancho, alto] zona calles

% Extract the regions defined by the windows
region1 = imcrop(I, ventana1);
region2 = imcrop(I, ventana2);
region3 = imcrop(I, ventana3);
region4 = imcrop(I, ventana4);
region5 = imcrop(I, ventana5);

% 2.- Cálculo de las medias de cada ventana
mediaV1 = mean(region1(:));
mediaV2 = mean(region2(:));
mediaV3 = mean(region3(:));
mediaV4 = mean(region4(:));
mediaV5 = mean(region5(:));

meanValues = [mediaV2, mediaV1, mediaV3, mediaV5, mediaV4];

%------------SEGUNDA PARTE: Presegmentación----------------
% 3.- Empleando aproximación de Bayes:

% 4.- Aproximando las probabilidades condicionales según funciones Anexas
s1 = meanValues(2) - meanValues(1);
s2 = meanValues(3) - meanValues(2);
s3 = meanValues(4) - meanValues(3);
s4 = meanValues(5) - meanValues(4);

function y = primer_ventana(x,mi,s)
    %Tramo 1: 0 <= x < mi+s/4  -> y = 1
    ch1 = x >= 0 & x <  mi+s/4;
    y(ch1) = 1;

    % Tramo 2: mi+s/4 <= x <= mi+s/2+s/4 -> y = -2/s * x + b
    ch2 = x >= mi+s/4 & x <= mi+s/2+s/4;
    b = 1.5 + 2*mi./s; %Obtenido de ecuaciones igualadas en valores conocidos de y (0 1)
    y(ch2) = -2 .* x(ch2) ./s + b;

    %Tramo 3: x > mi+s/4+s/2 -> y = 0
    ch3 = x > mi+s/4+s/2;
    y(ch3) = 0;
end

function y = ventana_n(x,mi,si, sii)
    syms b_pre b_pre1  % <-- declara los símbolos
    
    %Tramo 1: x < mi-si/4-si/2  -> y = 0
    ch1 = x < mi-si/4-si/2;
    y(ch1) = 0;
    
    %Tramo 2: mi-si/4-si/2 <= x < mi-si/4 -> y = 2/si * x + b
    ch2 = x >= mi-si/4-si/2 & x < mi-si/4;

    eq1 = 0 == 2 * (mi-si/4-si/2) /si + b_pre;
    b = solve(eq1,b_pre);

    y(ch2) = 2 .* x(ch2) ./si + b;

    %Tramo 3: mi-si/4 <= x < mi+sii/4  -> y = 1
    ch3 = x >= mi-si/4 & x < mi+sii/4;
    y(ch3) = 1;

    % Tramo 4: mi+sii/4 <= x <= mi+sii/2+sii/4 -> y = -2/sii * x +b
    ch4 = x >= mi+sii/4 & x <= mi+sii/2+sii/4;
    
    eq2 = 0 == - 2 * (mi+sii/2+sii/4) /sii + b_pre1;
    b1 = solve(eq2,b_pre1);

    y(ch4) = - 2 .* x(ch4) ./sii + b1;

    %Tramo 5: x > mi+sii/4+sii/2 -> y = 0
    ch5 = x > mi+sii/2+sii/4;
    y(ch5) = 0;

end

function y = ultima_ventana(x,mi,s)
    syms b_pre2
    %Tramo 1: x < mi-s/4-s/2  -> y = 0
    ch1 = x <  mi-s/4-s/2;
    y(ch1) = 0;

    % Tramo 2: mi-s/4-s/2 <= x <= mi-s/4 -> y = 2/s * x + b
    ch2 = x >= mi-s/4-s/2 & x <= mi-s/4;
    
    eq3 = 0 == 2 * (mi-s/2-s/4) /s + b_pre2;
    b2 = solve(eq3,b_pre2);
    
    y(ch2) = 2 .* x(ch2) ./s + b2;

    %Tramo 3: x > mi-s/4 -> y = 1
    ch3 = x > mi-s/4;
    y(ch3) = 1;
end


pX_W1 = primer_ventana(x, meanValues(1), s1);
pX_W2 = ventana_n(x, meanValues(2), s1, s2);
pX_W3 = ventana_n(x, meanValues(3), s2, s3);
pX_W4 = ventana_n(x, meanValues(4), s3, s4);
pX_W5 = ultima_ventana(x, meanValues(5), s4);


% Graficando las funciones obtenidas, e indicando la media calculada en
% cada función
%figure; plot(x, pX_W1, 'LineWidth', 2); grid on;
x%label('x'); ylabel('P(X|W1)'); title('primer\_tramo');

%figure; plot(x, pX_W2, 'LineWidth', 2); grid on;
%xlabel('x'); ylabel('P(X|W2)'); title('segundo\_tramo');

%figure; plot(x, pX_W3, 'LineWidth', 2); grid on;
%xlabel('x'); ylabel('P(X|W3)'); title('tercer\_tramo');

%figure; plot(x, pX_W4, 'LineWidth', 2); grid on;
%xlabel('x'); ylabel('P(X|W4)'); title('cuarto\_tramo');

%figure; plot(x, pX_W5, 'LineWidth', 2); grid on;
%xlabel('x'); ylabel('P(X|W5)'); title('quinto\_tramo');

figure; grid on; hold on;
plot(x, pX_W1, 'r', 'LineWidth', 2);
plot(x, pX_W2, 'b', 'LineWidth', 2);
plot(x, pX_W3, 'g', 'LineWidth', 2);
plot(x, pX_W4, 'y', 'LineWidth', 2);
plot(x, pX_W5, 'k', 'LineWidth', 2);
xline(meanValues(1), 'c', 'LineWidth', 1);
xline(meanValues(2), 'c', 'LineWidth', 1);
xline(meanValues(3), 'c', 'LineWidth', 1);
xline(meanValues(4), 'c', 'LineWidth', 1);
xline(meanValues(5), 'c', 'LineWidth', 1);
hold off;


%Dado que son equiprobables para el primer ejemplo:
PW1 = 1/5;
PW2 = 1/5;
PW3 = 1/5;
PW4 = 1/5;
PW5 = 1/5;

% 5.- Análisis pixel por pixel de imagen, segmentando de acuerdo con valores
%máximos de probabilidad a posteriori:
for x = 1:sz(1)
    % Process each pixel in the image for segmentation
    for y = 1:sz(2)
        pixelValue = I(x, y, :);
        % calculando pWi_X para cada pixel
        pW1_X = pX_W1(pixelValue)*PW1;
        pW2_X = pX_W2(pixelValue)*PW2;
        pW3_X = pX_W3(pixelValue)*PW3;
        pW4_X = pX_W4(pixelValue)*PW4;
        pW5_X = pX_W5(pixelValue)*PW5;

        [~, maxIdx] = max([pW1_X, pW2_X, pW3_X, pW4_X, pW5_X]); %~ ignora (descarta) el valor máximo; solo te quedas con maxIdx
        segmentedImage(x, y) = maxIdx; % Asigna el número de la clase a la que pertenece basado en la maxima probabilidad
    end
end



figure; imshow(I);


k = 1;  % clase a inspeccionar
figure; imshow(segmentedImage == k);
title(sprintf('Máscara de la clase %d', k));

k = 2;  % clase a inspeccionar
figure; imshow(segmentedImage == k);
title(sprintf('Máscara de la clase %d', k));

k = 3;  % clase a inspeccionar
figure; imshow(segmentedImage == k);
title(sprintf('Máscara de la clase %d', k));

k = 4;  % clase a inspeccionar
figure; imshow(segmentedImage == k);
title(sprintf('Máscara de la clase %d', k));

k = 5;  % clase a inspeccionar
figure; imshow(segmentedImage == k);
title(sprintf('Máscara de la clase %d', k));

