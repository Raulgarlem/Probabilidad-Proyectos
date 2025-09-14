%  Proyecto 2 - Temas selectos de Probabilidad
% 1er Examen Parccial: Segmentación Bayesiana
% --------------------------------------- %
clc; close all;

% PRIMERA PARTE: Obtencion de Parámetros
% --------------------------------------- %

% 1.- Definiendo 5 ventanas que identifiquen 5 regiones típicas sobre Paris.bmp
I = imread('Paris.bmp');
J = imread('Moneda_bn.bmp');
sz = size(I); % Tamaño de la imagen en pixeles
x = 0:255;

%histogram(I)
histogram(J)

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

P0Wi = [PW1, PW2, PW3, PW4, PW5]

% 5.- Análisis pixel por pixel de imagen, segmentando de acuerdo con valores
%máximos de probabilidad a posteriori:
function segmentedImage = SegmentarImagen(Img, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, PW1, PW2, PW3, PW4, PW5)
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
            pW5_X = pX_W5(pixelValue)*PW5;

            [~, maxIdx] = max([pW1_X, pW2_X, pW3_X, pW4_X, pW5_X]); %~ ignora (descarta) el valor máximo; solo te quedas con maxIdx
            segmentedImage(x, y) = maxIdx; % Asigna el número de la clase a la que pertenece basado en la maxima probabilidad
        end
    end
end

segmentedImageResult = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, PW1, PW2, PW3, PW4, PW5);


%TERCERA PARTE:Segmentación Iterativa
% -------------------------------------------

%6.- Emplea el resultado de la presegmentación para actualizar
%probabilidades P(Wi)

%Calculando nuevos valores PWi
function new_PW = calculateNewPW(preSegmentation)
    for n = 1:5
        classImage = preSegmentation == n;
        new_PW(n) = mean(classImage(:));
    end
end

P1Wi = calculateNewPW(segmentedImageResult)

%Repite el proceso de segmentación (Punto 5), las funciones condicionales
%(Punto 4) no cambian
segmentedResult1 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P1Wi(1), P1Wi(2), P1Wi(3), P1Wi(4), P1Wi(5));

% 7.- Repite 4 veces más el proceso de segmentación (punto 5). Determina el porcentaje de pixeles que cambian de clase entre los resultados de (presegmentación, 1ª segmentación), (1ª segmentación, 2ª segmentación),…(4ª segmentación, 5ª segmentación). Presenta las probabilidades P(Wi) y los porcentajes de cambio en una tabla. 
P2Wi = calculateNewPW(segmentedResult1)
segmentedResult2 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P2Wi(1), P2Wi(2), P2Wi(3), P2Wi(4), P2Wi(5));

P3Wi = calculateNewPW(segmentedResult2)
segmentedResult3 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P3Wi(1), P3Wi(2), P3Wi(3), P3Wi(4), P3Wi(5));

P4Wi = calculateNewPW(segmentedResult3)
segmentedResult4 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P4Wi(1), P4Wi(2), P4Wi(3), P4Wi(4), P4Wi(5));

P5Wi = calculateNewPW(segmentedResult4)
segmentedResult5 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P5Wi(1), P5Wi(2), P5Wi(3), P5Wi(4), P5Wi(5));




levels = uint8([meanValues(1) meanValues(2) meanValues(3) meanValues(4) meanValues(5)]);     % 5 niveles de gris
preseg = zeros(size(segmentedImageResult),'uint8');
seg1 = zeros(size(segmentedResult1),'uint8');
seg2 = zeros(size(segmentedResult2),'uint8');
seg3 = zeros(size(segmentedResult3),'uint8');
seg4 = zeros(size(segmentedResult4),'uint8');
seg5 = zeros(size(segmentedResult5),'uint8');

for r = 1:5
    preseg(segmentedImageResult == r) = levels(r);
    seg1(segmentedResult1 == r) = levels(r);
    seg2(segmentedResult2 == r) = levels(r);
    seg3(segmentedResult3 == r) = levels(r);
    seg4(segmentedResult4 == r) = levels(r);
    seg5(segmentedResult5 == r) = levels(r);
end

figure; imshow(I);

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
nexttile; imshow(preseg); title('Presegmentación');
nexttile; imshow(seg1); title('Primera segmentación');
nexttile; imshow(seg2); title('Segunda segmentación');
nexttile; imshow(seg3); title('Tercera segmentación');
nexttile; imshow(seg4); title('Cuarta segmentación');
nexttile; imshow(seg5); title('Quinta segmentación');


for k=1:5
    diffSegment1 = xor(segmentedImageResult == k, segmentedResult1 == k); %Comparación entre (presegmentación - 1a segmentación)
    pctCambio1(k) = 100 * nnz(diffSegment1) / numel(diffSegment1); %Calculando porcentaje de cambio

    diffSegment2 = xor(segmentedResult1 == k, segmentedResult2 == k); %Comparación entre (presegmentación - 1a segmentación)
    pctCambio2(k) = 100 * nnz(diffSegment2) / numel(diffSegment2); %Calculando porcentaje de cambio

    diffSegment3 = xor(segmentedResult2 == k, segmentedResult3 == k); %Comparación entre (presegmentación - 1a segmentación)
    pctCambio3(k) = 100 * nnz(diffSegment3) / numel(diffSegment3); %Calculando porcentaje de cambio

    diffSegment4 = xor(segmentedResult3 == k, segmentedResult4 == k); %Comparación entre (presegmentación - 1a segmentación)
    pctCambio4(k) = 100 * nnz(diffSegment4) / numel(diffSegment4); %Calculando porcentaje de cambio

    diffSegment5 = xor(segmentedResult4 == k, segmentedResult5 == k); %Comparación entre (presegmentación - 1a segmentación)
    pctCambio5(k) = 100 * nnz(diffSegment5) / numel(diffSegment5); %Calculando porcentaje de cambio
end


%GENERANDO TABLA
% Asegura que cada PxWi sea un vector fila de 1x5
toRow = @(v) reshape(v,1,[]);

M = [
    toRow(P0Wi);
    toRow(P1Wi);
    toRow(P2Wi);
    toRow(P3Wi);
    toRow(P4Wi);
    toRow(P5Wi)
];

ClaseNames = {'Clase1','Clase2','Clase3','Clase4','Clase5'};
RowNames  = {'Presegmentación','Segmentación 1','Segmentación 2','Segmentación 3','Segmentación 4','Segmentación 5'};

T = array2table(M, 'VariableNames', ClaseNames, 'RowNames', RowNames);
disp(T);                         % Muestra en consola

% (Opcional) Mostrar en ventana tipo hoja
figure('Position',[200 200 560 220]);
uitable('Data',T{:,:}, 'ColumnName',T.Properties.VariableNames, ...
        'RowName',T.Properties.RowNames, 'Units','normalized', 'Position',[0 0 1 1]);



% 1) Construir la matriz de datos (5 filas x 11 columnas)
M = [ ...
    P0Wi(:), ...
    P1Wi(:), pctCambio1(:), ...
    P2Wi(:), pctCambio2(:), ...
    P3Wi(:), pctCambio3(:), ...
    P4Wi(:), pctCambio4(:), ...
    P5Wi(:), pctCambio5(:) ...
];   % Cada "(:)" asegura columna (5x1)

% 2) Nombres de filas (clases)
RowNames = arrayfun(@(k) sprintf('Clase%d',k), 1:5, 'UniformOutput', false);

% 3) Titulares (con espacios, para mostrar en UI)
ColPretty = {'Presegmentación','Segmentación 1','Delta Ps-S1 (en %)', ...
             'Segmentación 2','Delta S1-S2 (en %)', ...
             'Segmentación 3','Delta S2-S3 (en %)', ...
             'Segmentación 4','Delta S3-S4 (en %)', ...
             'Segmentación 5','Delta S4-S5 (en %)'};

% 4) Nombres de variables válidos para la tabla (sin espacios)
VarNames = matlab.lang.makeValidName(ColPretty, 'ReplacementStyle','delete');

% 5) Crear la tabla (filas = clases, columnas = titulares)
T = array2table(M, 'RowNames', RowNames, 'VariableNames', VarNames);

% 6) Ver en consola
disp(T);

% 7) (Opcional) Mostrar en una tabla tipo hoja con los títulos "bonitos"
figure('Position',[200 200 900 220]);
uitable('Data', T{:,:}, ...
        'ColumnName', ColPretty, ...
        'RowName',   RowNames, ...
        'Units','normalized', 'Position',[0 0 1 1]);

%
% --- Comparación de una ventana específica ---
ventana = [100 200 100 200];  % [fila_ini fila_fin col_ini col_fin]
figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(preseg(ventana(1):ventana(2), ventana(3):ventana(4))); title('Ventana Presegmentación');
nexttile; imshow(seg1(ventana(1):ventana(2), ventana(3):ventana(4))); title('Primera segmentación');
nexttile; imshow(seg2(ventana(1):ventana(2), ventana(3):ventana(4))); title('Segunda segmentación');
nexttile; imshow(seg3(ventana(1):ventana(2), ventana(3):ventana(4))); title('Tercera segmentación');
nexttile; imshow(seg4(ventana(1):ventana(2), ventana(3):ventana(4))); title('Cuarta segmentación');
nexttile; imshow(seg5(ventana(1):ventana(2), ventana(3):ventana(4))); title('Quinta segmentación');


%k=2;
%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segmentedImageResult == k); title('Presegmentación');
%nexttile; imshow(segmentedResult1 == k); title('Primera segmentación clase %d', k);
%nexttile; imshow(segmentedResult2 == k); title('Segunda segmentación clase %d', k');
%nexttile; imshow(segmentedResult3 == k); title('Tercera segmentación clase %d', k');
%nexttile; imshow(segmentedResult4 == k); title('Cuarta segmentación clase %d', k');
%nexttile; imshow(segmentedResult5 == k); title('Quinta segmentación clase %d', k');

%k = 2;  % clase a inspeccionar
%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segmentedImageResult == k); title('Presegmentación');
%nexttile; imshow(segmentedResult1 == k); title('Primera segmentación clase %d', k);
%nexttile; imshow(segmentedResult2 == k); title('Segunda segmentación clase %d', k');
%nexttile; imshow(segmentedResult3 == k); title('Tercera segmentación clase %d', k');
%nexttile; imshow(segmentedResult4 == k); title('Cuarta segmentación clase %d', k');
%nexttile; imshow(segmentedResult5 == k); title('Quinta segmentación clase %d', k');

%k = 3;  % clase a inspeccionar
%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segmentedImageResult == k); title('Presegmentación');
%nexttile; imshow(segmentedResult1 == k); title('Primera segmentación clase %d', k);
%nexttile; imshow(segmentedResult2 == k); title('Segunda segmentación clase %d', k');
%nexttile; imshow(segmentedResult3 == k); title('Tercera segmentación clase %d', k');
%nexttile; imshow(segmentedResult4 == k); title('Cuarta segmentación clase %d', k');
%nexttile; imshow(segmentedResult5 == k); title('Quinta segmentación clase %d', k');

%k = 4;  % clase a inspeccionar
%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segmentedImageResult == k); title('Presegmentación');
%nexttile; imshow(segmentedResult1 == k); title('Primera segmentación clase %d', k);
%nexttile; imshow(segmentedResult2 == k); title('Segunda segmentación clase %d', k');
%nexttile; imshow(segmentedResult3 == k); title('Tercera segmentación clase %d', k');
%nexttile; imshow(segmentedResult4 == k); title('Cuarta segmentación clase %d', k');
%nexttile; imshow(segmentedResult5 == k); title('Quinta segmentación clase %d', k');

%k = 5;  % clase a inspeccionar
%figure;
%tiledlayout(2,3,'TileSpacing','compact','Padding','compact');  % 1 fila, 5 columnas
%nexttile; imshow(segmentedImageResult == k); title('Presegmentación');
%nexttile; imshow(segmentedResult1 == k); title('Primera segmentación clase %d', k);
%nexttile; imshow(segmentedResult2 == k); title('Segunda segmentación clase %d', k');
%nexttile; imshow(segmentedResult3 == k); title('Tercera segmentación clase %d', k');
%nexttile; imshow(segmentedResult4 == k); title('Cuarta segmentación clase %d', k');
%nexttile; imshow(segmentedResult5 == k); title('Quinta segmentación clase %d', k');
