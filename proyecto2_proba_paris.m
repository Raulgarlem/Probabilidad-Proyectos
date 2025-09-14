clc; close all;

% --- PRIMERA PARTE: Obtención de Parámetros ---
I = imread('Paris.bmp');
sz = size(I);
x = 0:255;

% Definición de ventanas
ventana1 = [214 427 227-214 466-427];
ventana2 = [270 390 284-270 396-390];
ventana3 = [19 151 36-19 162-151];
ventana4 = [463 247 471-459 253-247];
ventana5 = [262 194 329-262 252-194];

% Extracción de regiones
region1 = imcrop(I, ventana1);
region2 = imcrop(I, ventana2);
region3 = imcrop(I, ventana3);
region4 = imcrop(I, ventana4);
region5 = imcrop(I, ventana5);

% Cálculo de medias
mediaV1 = mean(region1(:));
mediaV2 = mean(region2(:));
mediaV3 = mean(region3(:));
mediaV4 = mean(region4(:));
mediaV5 = mean(region5(:));

meanValues = [mediaV2, mediaV1, mediaV3, mediaV5, mediaV4];

% --- SEGUNDA PARTE: Presegmentación ---

s1 = meanValues(2) - meanValues(1);
s2 = meanValues(3) - meanValues(2);
s3 = meanValues(4) - meanValues(3);
s4 = meanValues(5) - meanValues(4);

% Funciones condicionales
function y = primer_ventana(x,mi,s)
    ch1 = x >= 0 & x <  mi+s/4;
    y(ch1) = 1;
    ch2 = x >= mi+s/4 & x <= mi+s/2+s/4;
    b = 1.5 + 2*mi./s;
    y(ch2) = -2 .* x(ch2) ./s + b;
    ch3 = x > mi+s/4+s/2;
    y(ch3) = 0;
end

function y = ventana_n(x,mi,si,sii)
    syms b_pre b_pre1
    ch1 = x < mi-si/4-si/2;
    y(ch1) = 0;
    ch2 = x >= mi-si/4-si/2 & x < mi-si/4;
    eq1 = 0 == 2 * (mi-si/4-si/2) /si + b_pre;
    b = double(solve(eq1,b_pre));
    y(ch2) = 2 .* x(ch2) ./si + b;
    ch3 = x >= mi-si/4 & x < mi+sii/4;
    y(ch3) = 1;
    ch4 = x >= mi+sii/4 & x <= mi+sii/2+sii/4;
    eq2 = 0 == -2 * (mi+sii/2+sii/4) /sii + b_pre1;
    b1 = double(solve(eq2,b_pre1));
    y(ch4) = -2 .* x(ch4) ./sii + b1;
    ch5 = x > mi+sii/2+sii/4;
    y(ch5) = 0;
end

function y = ultima_ventana(x,mi,s)
    syms b_pre2
    ch1 = x < mi-s/4-s/2;
    y(ch1) = 0;
    ch2 = x >= mi-s/4-s/2 & x <= mi-s/4;
    eq3 = 0 == 2 * (mi-s/2-s/4) /s + b_pre2;
    b2 = double(solve(eq3,b_pre2));
    y(ch2) = 2 .* x(ch2) ./s + b2;
    ch3 = x > mi-s/4;
    y(ch3) = 1;
end

% Evaluación de funciones condicionales
pX_W1 = primer_ventana(x, meanValues(1), s1);
pX_W2 = ventana_n(x, meanValues(2), s1, s2);
pX_W3 = ventana_n(x, meanValues(3), s2, s3);
pX_W4 = ventana_n(x, meanValues(4), s3, s4);
pX_W5 = ultima_ventana(x, meanValues(5), s4);

% Visualización de funciones
figure; grid on; hold on;
plot(x, pX_W1, 'r', 'LineWidth', 2);
plot(x, pX_W2, 'b', 'LineWidth', 2);
plot(x, pX_W3, 'g', 'LineWidth', 2);
plot(x, pX_W4, 'y', 'LineWidth', 2);
plot(x, pX_W5, 'k', 'LineWidth', 2);
xline(meanValues, 'c', 'LineWidth', 1);
hold off;

% --- TERCERA PARTE: Segmentación Iterativa ---

% Función de segmentación
function segmentedImage = SegmentarImagen(Img, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, PW1, PW2, PW3, PW4, PW5)
    imgsz = size(Img);
    for x = 1:imgsz(1)
        for y = 1:imgsz(2)
            pixelValue = Img(x, y);
            pW1_X = pX_W1(pixelValue+1)*PW1;
            pW2_X = pX_W2(pixelValue+1)*PW2;
            pW3_X = pX_W3(pixelValue+1)*PW3;
            pW4_X = pX_W4(pixelValue+1)*PW4;
            pW5_X = pX_W5(pixelValue+1)*PW5;
            [~, maxIdx] = max([pW1_X, pW2_X, pW3_X, pW4_X, pW5_X]);
            segmentedImage(x, y) = maxIdx;
        end
    end
end

% Función para calcular nuevas probabilidades
function new_PW = calculateNewPW(seg)
    for n = 1:5
        new_PW(n) = mean(seg(:) == n);
    end
end

% Probabilidades iniciales
PW1 = 1/5; PW2 = 1/5; PW3 = 1/5; PW4 = 1/5; PW5 = 1/5;
P0Wi = [PW1, PW2, PW3, PW4, PW5];

% Segmentación iterativa
segmentedImageResult = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, PW1, PW2, PW3, PW4, PW5);
PWi = calculateNewPW(segmentedImageResult);

segmentedResult1 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, PWi(1), PWi(2), PWi(3), PWi(4), PWi(5));
P1Wi = calculateNewPW(segmentedResult1);

segmentedResult2 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P1Wi(1), P1Wi(2), P1Wi(3), P1Wi(4), P1Wi(5));
P2Wi = calculateNewPW(segmentedResult2);

segmentedResult3 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P2Wi(1), P2Wi(2), P2Wi(3), P2Wi(4), P2Wi(5));
P3Wi = calculateNewPW(segmentedResult3);

segmentedResult4 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P3Wi(1), P3Wi(2), P3Wi(3), P3Wi(4), P3Wi(5));
P4Wi = calculateNewPW(segmentedResult4);

segmentedResult5 = SegmentarImagen(I, pX_W1, pX_W2, pX_W3, pX_W4, pX_W5, P4Wi(1), P4Wi(2), P4Wi(3), P4Wi(4), P4Wi(5));

% --- Análisis de cambios y tabla final ---
function delta = errorAbsolutoRelativo(prev, actual)
    delta = abs((actual - prev) ./ max(prev, eps)) * 100;
end

function cambio = calcularCambioSegmentacion(segAnterior, segActual)
    cambio = 100 * mean(segAnterior(:) ~= segActual(:));
end

% Tabla de probabilidades por iteración
PWi_table = [P0Wi; PWi; P1Wi; P2Wi; P3Wi; P4Wi];

% Errores absolutos relativos por clase
deltaPWi = zeros(6,5);
deltaPWi(1,:) = 0;
for i = 1:5
    deltaPWi(i+1,:) = errorAbsolutoRelativo(PWi_table(i,:), PWi_table(i+1,:));
end

% Cambios entre segmentaciones
cambiosSegmentacion = zeros(6,1);
cambiosSegmentacion(1) = 0;
cambiosSegmentacion(2) = calcularCambioSegmentacion(segmentedImageResult, segmentedResult1);
cambiosSegmentacion(3) = calcularCambioSegmentacion(segmentedResult1, segmentedResult2);
cambiosSegmentacion(4) = calcularCambioSegmentacion(segmentedResult2, segmentedResult3);
cambiosSegmentacion(5) = calcularCambioSegmentacion(segmentedResult3, segmentedResult4);
cambiosSegmentacion(6) = calcularCambioSegmentacion(segmentedResult4, segmentedResult5);

% Crear tabla
iteracion = (0:5)';
T = table(iteracion, cambiosSegmentacion, ...
    PWi_table(:,1), deltaPWi(:,1), ...
    PWi_table(:,2), deltaPWi(:,2), ...
    PWi_table(:,3), deltaPWi(:,3), ...
    PWi_table(:,4), deltaPWi(:,4), ...
    PWi_table(:,5), deltaPWi(:,5), ...
    'VariableNames', {'Iteracion', 'CambioSegmentacion', ...
    'P_W1', 'ErrorRel_W1', 'P_W2', 'ErrorRel_W2', ...
    'P_W3', 'ErrorRel_W3', 'P_W4', 'ErrorRel_W4', 'P_W5', 'ErrorRel_W5'});

disp('Tabla de evolución de probabilidades y errores por clase:');
disp(T);


% --- Visualización de segmentaciones en escala de grises ---
levels = uint8([meanValues(1) meanValues(2) meanValues(3) meanValues(4) meanValues(5)]);
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

% Mostrar todas las segmentaciones
figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(preseg); title('Presegmentación');
nexttile; imshow(seg1); title('Primera segmentación');
nexttile; imshow(seg2); title('Segunda segmentación');
nexttile; imshow(seg3); title('Tercera segmentación');
nexttile; imshow(seg4); title('Cuarta segmentación');
nexttile; imshow(seg5); title('Quinta segmentación');

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

figure('Name','Histogramas por Iteración');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile; h0 = histcounts(preseg, 0:256); bar(0:255, h0, 'r'); title('Presegmentación'); xlim([0 255]); ylim([0 1.1*max(h0)]);
nexttile; h1 = histcounts(seg1, 0:256); bar(0:255, h1, 'r'); title('Iteración 1'); xlim([0 255]); ylim([0 1.1*max(h1)]);
nexttile; h2 = histcounts(seg2, 0:256); bar(0:255, h2, 'r'); title('Iteración 2'); xlim([0 255]); ylim([0 1.1*max(h2)]);
nexttile; h3 = histcounts(seg3, 0:256); bar(0:255, h3, 'r'); title('Iteración 3'); xlim([0 255]); ylim([0 1.1*max(h3)]);
nexttile; h4 = histcounts(seg4, 0:256); bar(0:255, h4, 'r'); title('Iteración 4'); xlim([0 255]); ylim([0 1.1*max(h4)]);
nexttile; h5 = histcounts(seg5, 0:256); bar(0:255, h5, 'r'); title('Iteración 5'); xlim([0 255]); ylim([0 1.1*max(h5)]);


%Segmentando por umbrales
function [Iseg, th] = segmenta_con_umbrales(I, modo, k_o_vect)
% Segmenta por umbrales y muestra (1) original, (2) segmentada y (3) histograma con umbrales.
% I           : imagen RGB o en gris (uint8/uint16/double)
% modo        : 'multithresh'
% k_o_vect    : si 'multithresh' -> k (número de niveles, p.ej., 2,3,4)

% Salida:
% Iseg : imagen etiquetada (1..Nclases)
% th   : umbrales en unidad de la imagen de trabajo (double 0..1 si se normaliza)

    % --- 0) Asegurar gris en [0..1] para umbralado robusto ---
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = im2double(Igray);     % normaliza a [0,1] sin perder robustez

    % --- 1) Elegir umbrales ---
    switch lower(modo)
        
        case 'multithresh'          % k umbrales (k >= 1)
            if nargin < 3 || isempty(k_o_vect), k_o_vect = 2; end
            k = k_o_vect;
            th = multithresh(Igray, k);             % [t1 t2 ...] en [0,1]
            Iseg = imquantize(Igray, th);           % etiquetas 1..k+1

        
        otherwise
            error('Modo no reconocido. Usa: ''multithresh'' ');
    end

    % --- 2) Mostrar resultados y el histograma con líneas de umbral ---
    figure('Name','Segmentación por umbrales','Color','w');
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

    % (a) Original
    nexttile; imshow(I,[]); title('Original'); axis image off;

    % (b) Segmentada (en etiquetas)
    nexttile; 
    imagesc(Iseg); axis image off; 
    colormap(gca, turbo(double(max(Iseg(:))))); colorbar;
    title(sprintf('Segmentada (%d clases)', double(max(Iseg(:)))));

    % (c) Histograma con umbrales
    nexttile;
    % Como Igray está en [0,1], hacemos hist binario robusto:
    histogram(Igray(:), 256); grid on;
    xlabel('Intensidad (0..1)'); ylabel('N° píxeles');
    title('Histograma con umbrales');

    if exist('xline','file') == 2
        for t = th
            xline(t, '--r', sprintf('t=%.3f', t), 'LabelOrientation','horizontal');
        end
    else
        yl = ylim;
        for t = th
            hold on; plot([t t], yl, 'r--', 'LineWidth',1.5); hold off;
        end
    end
end

%Aplicando a la Imagen
[Iseg, th] = segmenta_con_umbrales(I, 'multithresh', 5);
disp(th);   % vector de 3 umbrales (0..1)