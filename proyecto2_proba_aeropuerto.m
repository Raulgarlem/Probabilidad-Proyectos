clc; close all;

% --- Cargar imagen ---
I = imread('Aeropuerto.jpg');  % Imagen en escala de grises
x = 0:255;                    % Rango de niveles de intensidad

histogram(I)

% --- Definir 2 ventanas representativas ---
%ventana1 = [1780 938 30 30];  % Región clase 1
ventana1 = [348 403 30 30];
ventana2 = [421 173 30 30];   % Región clase 2
ventana3 = [86 53 30 30];
ventana4 = [36 23 30 30];

region1 = imcrop(I, ventana1);
region2 = imcrop(I, ventana2);
region3 = imcrop(I, ventana3);
region4 = imcrop(I, ventana4);

%figure; imshow(region1);
%figure; imshow(region2);
%figure; imshow(region3);
%figure; imshow(region4);

% --- Calcular medias y parámetro de dispersión ---
media1 = mean(region1(:));
%media1 = 72;
media2 = mean(region2(:));
%media2 = 105;
media3 = mean(region3(:));
%media3 =207;
media4 = mean(region4(:));
%media4 = 231;
s1 = abs(media2 - media1);
s2 = abs(media3 - media2);
s3 = abs(media4 - media3);

% --- Funciones condicionales ---
function y = clase1(x,mi,s1)
    %Tramo 1: 0 <= x < mi+s/4  -> y = 1
    ch1 = x >= 0 & x <  mi+s1/4;
    y(ch1) = 1;

    % Tramo 2: mi+s/4 <= x <= mi+s/2+s/4 -> y = -2/s * x + b
    ch2 = x >= mi+s1/4 & x <= mi+s1/2+s1/4;
    b = 1.5 + 2*mi./s1; %Obtenido de ecuaciones igualadas en valores conocidos de y (0 1)
    y(ch2) = -2 .* x(ch2) ./s1 + b;

    %Tramo 3: x > mi+s/4+s/2 -> y = 0
    ch3 = x > mi+s1/4+s1/2;
    y(ch3) = 0;
end

function y = clase_n(x,mi,si, sii)
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

function y = clasefin(x,mi,s4)
    syms b_pre2
    %Tramo 1: x < mi-s/4-s/2  -> y = 0
    ch1 = x <  mi-s4/4-s4/2;
    y(ch1) = 0;

    % Tramo 2: mi-s/4-s/2 <= x <= mi-s/4 -> y = 2/s * x + b
    ch2 = x >= mi-s4/4-s4/2 & x <= mi-s4/4;
    
    eq3 = 0 == 2 * (mi-s4/2-s4/4) /s4 + b_pre2;
    b2 = solve(eq3,b_pre2);
    
    y(ch2) = 2 .* x(ch2) ./s4 + b2;

    %Tramo 3: x > mi-s/4 -> y = 1
    ch3 = x > mi-s4/4;
    y(ch3) = 1;
end


% --- Evaluar funciones condicionales ---
pX_W1 = clase1(x, media1, s1);
pX_W2 = clase_n(x, media2, s1, s2);
pX_W3 = clase_n(x, media3, s2, s3);
pX_W4 = clasefin(x, media4, s3);

% --- Visualizar funciones ---
figure; grid on; hold on;
plot(x, pX_W1, 'r', 'LineWidth', 2);
plot(x, pX_W2, 'b', 'LineWidth', 2);
plot(x, pX_W3, 'g', 'LineWidth', 2);
plot(x, pX_W4, 'y', 'LineWidth', 2);
xline(media1, 'r--');
xline(media2, 'b--');
xline(media3, 'g--');
xline(media4, 'y--');
legend('P(X|W1)', 'P(X|W2)', 'P(X|W3)', 'P(X|W4)');
title('Funciones condicionales para segmentación');
hold off;

% 5.- Análisis pixel por pixel de imagen, segmentando de acuerdo con valores
%máximos de probabilidad a posteriori:
function segmentedImage = SegmentarImagen(Img, pX_W1, pX_W2, pX_W3, pX_W4, PW1, PW2, PW3, PW4)
    imgsz = size(Img);

    for x = 1:imgsz(1)
        % Process each pixel in the image for segmentation
        for y = 1:imgsz(2)
            pixelValue = Img(x, y, :);
            % calculando pWi_X para cada pixel
            pW1_X = pX_W1(pixelValue)*PW1;
            pW2_X = pX_W2(pixelValue)*PW2;
            pW3_X = pX_W3(pixelValue)*PW3;
            pW4_X = pX_W4(pixelValue)*PW4;

            [~, maxIdx] = max([pW1_X, pW2_X, pW3_X, pW4_X]); %~ ignora (descarta) el valor máximo; solo te quedas con maxIdx
            segmentedImage(x, y) = maxIdx; % Asigna el número de la clase a la que pertenece basado en la maxima probabilidad
        end
    end
end



% --- Función para actualizar probabilidades ---
function new_PW = calculateNewPW_bin(seg)
    for n = 1:4
        classImage = seg == n;
        new_PW(n) = mean(classImage(:));
    end
    %new_PW(1) = mean(seg(:) == 1);  % P(W1)
    %new_PW(2) = mean(seg(:) == 0);  % P(W2)
end

% --- Función para calcular error iterativo por clase ---
function [error_W1, error_W2] = calcularCambioPorClase(segAnterior, segActual)
    total_W1 = sum(segAnterior(:) == 1);
    total_W2 = sum(segAnterior(:) == 0);

    error_W1 = 100 * sum((segAnterior(:) == 1) & (segActual(:) ~= 1)) / max(total_W1, 1);
    error_W2 = 100 * sum((segAnterior(:) == 0) & (segActual(:) ~= 0)) / max(total_W2, 1);
end

% --- Iteración 0: Equiprobable ---
PW0 = [1/4, 1/4, 1/4, 1/4];
segm0 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW0(1), PW0(2), PW0(3), PW0(4));

% --- Iteraciones 1 a 5 ---
PW1 = calculateNewPW_bin(segm0);
segm1 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW1(1), PW1(2), PW1(3), PW1(4));

PW2 = calculateNewPW_bin(segm1);
segm2 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW2(1), PW2(2), PW2(3), PW2(4));

PW3 = calculateNewPW_bin(segm2);
segm3 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW3(1), PW3(2), PW3(3), PW3(4));

PW4 = calculateNewPW_bin(segm3);
segm4 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW4(1), PW4(2), PW4(3), PW4(4));

PW5 = calculateNewPW_bin(segm4);
segm5 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, PW5(1), PW5(2), PW5(3), PW5(4));

% --- Calcular errores por clase entre iteraciones ---
error_W1 = zeros(6,1);
error_W2 = zeros(6,1);
error_W1(1) = 0;
error_W2(1) = 0;

[error_W1(2), error_W2(2)] = calcularCambioPorClase(segm0, segm1);
[error_W1(3), error_W2(3)] = calcularCambioPorClase(segm1, segm2);
[error_W1(4), error_W2(4)] = calcularCambioPorClase(segm2, segm3);
[error_W1(5), error_W2(5)] = calcularCambioPorClase(segm3, segm4);
[error_W1(6), error_W2(6)] = calcularCambioPorClase(segm4, segm5);

% --- Tabla de evolución de probabilidades y errores por clase ---
PWi_table = [PW0; PW1; PW2; PW3; PW4; PW5];
iteracion = (0:5)';
T = table(iteracion, error_W1, error_W2, ...
    PWi_table(:,1), PWi_table(:,2), ...
    'VariableNames', {'Iteracion', 'Error_W1', 'Error_W2', 'P_W1', 'P_W2'});

disp(T);

% Visualizando iteraciones:
levels1 = uint8([media1 media2 media3 media4]);     % 5 niveles de gris
segs0 = zeros(size(segm0),'uint8');
segs1 = zeros(size(segm1),'uint8');
segs2 = zeros(size(segm2),'uint8');
segs3 = zeros(size(segm3),'uint8');
segs4 = zeros(size(segm4),'uint8');
segs5 = zeros(size(segm5),'uint8');

for r = 1:4
    segs0(segm0 == r) = levels1(r);
    segs1(segm1 == r) = levels1(r);
    segs2(segm2 == r) = levels1(r);
    segs3(segm3 == r) = levels1(r);
    segs4(segm4 == r) = levels1(r);
    segs5(segm5 == r) = levels1(r);
end

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
nexttile; imshow(segs0); title('Presegmentación');
nexttile; imshow(segs1); title('Primera segmentación');
nexttile; imshow(segs2); title('Segunda segmentación');
nexttile; imshow(segs3); title('Tercera segmentación');
nexttile; imshow(segs4); title('Cuarta segmentación');
nexttile; imshow(segs5); title('Quinta segmentación');


%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segm0==1); title('Presegmentación');
%nexttile; imshow(segm0==2); title('Primera segmentación');
%nexttile; imshow(segm0==3); title('Segunda segmentación');
%nexttile; imshow(segm0==4); title('Tercera segmentación');
