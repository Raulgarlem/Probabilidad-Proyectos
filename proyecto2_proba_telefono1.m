clc; close all;

% --- Cargar imagen ---
I = imread('Telefono.bmp');  % Imagen en escala de grises
x = 0:255;                    % Rango de niveles de intensidad

%histogram(I)

% --- Definir 2 ventanas representativas ---
%ventana1 = [1780 938 30 30];  % Región clase 1
ventana1 = [119 196 30 30];
ventana2 = [251 64 30 30];   % Región clase 2

region1 = imcrop(I, ventana1);
region2 = imcrop(I, ventana2);

%figure; imshow(region1);
%figure; imshow(region2);

% --- Calcular medias y parámetro de dispersión ---
media1 = mean(region1(:));
%media1 = 80;
media2 = mean(region2(:));
%media2 = 195;
s = abs(media2 - media1);

% --- Funciones condicionales ---
function y = clase1(x, mi, s)
    ch1 = x < mi + s/4;
    y(ch1) = 1;

    ch2 = x >= mi + s/4 & x <= mi + s/2 + s/4;
    b = 1.5 + 2*mi/s;
    y(ch2) = -2 * x(ch2)/s + b;
    
    ch3 = x > mi + s/2 + s/4;
    y(ch3) = 0;
end

function y = clase2(x, mi, s)
    syms b_pre
    ch1 = x < mi - s/4 - s/2;
    y(ch1) = 0;
    
    ch2 = x >= mi - s/4 - s/2 & x < mi - s/4;
    eq = 0 == 2 * (mi - s/4 - s/2)/s + b_pre;
    b = double(solve(eq, b_pre));
    y(ch2) = 2 * x(ch2)/s + b;
    
    ch3 = x >= mi - s/4;
    y(ch3) = 1;
end

% --- Evaluar funciones condicionales ---
pX_W1 = clase1(x, media1, s);
pX_W2 = clase2(x, media2, s);

% --- Visualizar funciones ---
figure; grid on; hold on;
plot(x, pX_W1, 'r', 'LineWidth', 2);
plot(x, pX_W2, 'b', 'LineWidth', 2);
xline(media1, 'r--');
xline(media2, 'b--');
legend('P(X|W1)', 'P(X|W2)');
title('Funciones condicionales para segmentación binaria');
hold off;

% --- Función de segmentación binaria ---
function segmentedImage = SegmentarBinaria(Img, pX_W1, pX_W2, PW1, PW2)
    imgsz = size(Img);
    segmentedImage = zeros(imgsz(1), imgsz(2));
    for x = 1:imgsz(1)
        for y = 1:imgsz(2)
            pixelValue = Img(x, y);
            pW1_X = pX_W1(pixelValue+1) * PW1;
            pW2_X = pX_W2(pixelValue+1) * PW2;

            [~, maxIdx1] = max([pW1_X, pW2_X]); %~ ignora (descarta) el valor máximo; solo te quedas con maxIdx
            segmentedImage(x, y) = maxIdx1; % Asigna el número de la clase a la que pertenece basado en la maxima probabilidad
        end
    end
end

% --- Función para actualizar probabilidades ---
function new_PW = calculateNewPW_bin(seg)
    for n = 1:2
        classImage = seg == n;
        new_PW(n) = mean(classImage(:));
    end
    %new_PW(1) = mean(seg(:) == 1);  % P(W1)
    %new_PW(2) = mean(seg(:) == 0);  % P(W2)
end

% --- Función para calcular error iterativo por clase ---
function delta = calcularErrorAbsolutoRelativo(prev, actual)
    delta = abs((actual - prev) ./ max(prev, eps)) * 100;
end

% --- Iteración 0: Equiprobable ---
PW0 = [0.5, 0.5];
segm0 = SegmentarBinaria(I, pX_W1, pX_W2, PW0(1), PW0(2));

% --- Iteraciones 1 a 5 ---
PW1 = calculateNewPW_bin(segm0);
segm1 = SegmentarBinaria(I, pX_W1, pX_W2, PW1(1), PW1(2));

PW2 = calculateNewPW_bin(segm1);
segm2 = SegmentarBinaria(I, pX_W1, pX_W2, PW2(1), PW2(2));

PW3 = calculateNewPW_bin(segm2);
segm3 = SegmentarBinaria(I, pX_W1, pX_W2, PW3(1), PW3(2));

PW4 = calculateNewPW_bin(segm3);
segm4 = SegmentarBinaria(I, pX_W1, pX_W2, PW4(1), PW4(2));

PW5 = calculateNewPW_bin(segm4);
segm5 = SegmentarBinaria(I, pX_W1, pX_W2, PW5(1), PW5(2));

% --- Calcular errores por clase entre iteraciones (segmentación) ---
error_W1 = zeros(6,1);
error_W2 = zeros(6,1);
error_W1(1) = 0;
error_W2(1) = 0;

delta = calcularErrorAbsolutoRelativo(PW0, PW1);
error_W1(2) = delta(1);
error_W2(2) = delta(2);

delta = calcularErrorAbsolutoRelativo(PW1, PW2);
error_W1(3) = delta(1);
error_W2(3) = delta(2);

delta = calcularErrorAbsolutoRelativo(PW2, PW3);
error_W1(4) = delta(1);
error_W2(4) = delta(2);

delta = calcularErrorAbsolutoRelativo(PW3, PW4);
error_W1(5) = delta(1);
error_W2(5) = delta(2);

delta = calcularErrorAbsolutoRelativo(PW4, PW5);
error_W1(6) = delta(1);
error_W2(6) = delta(2);


% --- Tabla de evolución de probabilidades y errores por clase ---
PWi_table = [PW0; PW1; PW2; PW3; PW4; PW5];
iteracion = (0:5)';
T = table(iteracion, error_W1, error_W2, ...
    PWi_table(:,1), PWi_table(:,2), ...
    'VariableNames', {'Iteracion', 'ErrorRel_W1', 'ErrorRel_W2', 'P_W1', 'P_W2'});

disp('Tabla de evolución de probabilidades y errores por clase:');
disp(T);

% --- Visualizar iteraciones ---
figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(segm0); title('Iteración 0');
nexttile; imshow(segm1); title('Iteración 1');
nexttile; imshow(segm2); title('Iteración 2');
nexttile; imshow(segm3); title('Iteración 3');
nexttile; imshow(segm4); title('Iteración 4');
nexttile; imshow(segm5); title('Iteración 5');

levels1 = uint8([media1 media2]);
segs0 = zeros(size(segm0),'uint8');
segs1 = zeros(size(segm1),'uint8');
segs2 = zeros(size(segm2),'uint8');
segs3 = zeros(size(segm3),'uint8');
segs4 = zeros(size(segm4),'uint8');
segs5 = zeros(size(segm5),'uint8');

for r = 1:2
    segs0(segm0 == r) = levels1(r);
    segs1(segm1 == r) = levels1(r);
    segs2(segm2 == r) = levels1(r);
    segs3(segm3 == r) = levels1(r);
    segs4(segm4 == r) = levels1(r);
    segs5(segm5 == r) = levels1(r);
end

figure; imshow(I);

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(segs0); title('Presegmentación');
nexttile; imshow(segs1); title('Primera segmentación');
nexttile; imshow(segs2); title('Segunda segmentación');
nexttile; imshow(segs3); title('Tercera segmentación');
nexttile; imshow(segs4); title('Cuarta segmentación');
nexttile; imshow(segs5); title('Quinta segmentación');

ventana = [150 250 250 350];
figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(segs0(ventana(1):ventana(2), ventana(3):ventana(4))); title('Ventana Presegmentación');
nexttile; imshow(segs1(ventana(1):ventana(2), ventana(3):ventana(4))); title('Primera segmentación');
nexttile; imshow(segs2(ventana(1):ventana(2), ventana(3):ventana(4))); title('Segunda segmentación');
nexttile; imshow(segs3(ventana(1):ventana(2), ventana(3):ventana(4))); title('Tercera segmentación');
nexttile; imshow(segs4(ventana(1):ventana(2), ventana(3):ventana(4))); title('Cuarta segmentación');
nexttile; imshow(segs5(ventana(1):ventana(2), ventana(3):ventana(4))); title('Quinta segmentación');

figure('Name','Histogramas por Iteración');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile; h0 = histcounts(segs0, 0:256); bar(0:255, h0, 'r'); title('Presegmentación'); xlim([0 255]); ylim([0 1.1*max(h0)]);
nexttile; h1 = histcounts(segs1, 0:256); bar(0:255, h1, 'r'); title('Iteración 1'); xlim([0 255]); ylim([0 1.1*max(h1)]);
nexttile; h2 = histcounts(segs2, 0:256); bar(0:255, h2, 'r'); title('Iteración 2'); xlim([0 255]); ylim([0 1.1*max(h2)]);
nexttile; h3 = histcounts(segs3, 0:256); bar(0:255, h3, 'r'); title('Iteración 3'); xlim([0 255]); ylim([0 1.1*max(h3)]);
nexttile; h4 = histcounts(segs4, 0:256); bar(0:255, h4, 'r'); title('Iteración 4'); xlim([0 255]); ylim([0 1.1*max(h4)]);
nexttile; h5 = histcounts(segs5, 0:256); bar(0:255, h5, 'r'); title('Iteración 5'); xlim([0 255]); ylim([0 1.1*max(h5)]);



