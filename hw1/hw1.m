%% Oct 14, 2019 
clear all;
clc;

%% Question a
load('TrainingSamplesDCT_8.mat'); 

TB = TrainsampleDCT_BG; 
[xtb,ytb] = size(TB);

TF = TrainsampleDCT_FG; 
[xtf,ytf] = size(TF);

PY = xtf/(xtf+xtb) % cheetah
PX = xtb/(xtf+xtb) % grass

%% Question b
    % Find the second largest index
row = 0;
col = 0;
for a = 1:xtb
    [row,col] = max(TB(a,2:end));
    XB(a) = col+1;
end

row2 = 0;
col2 = 0;
for b = 1:xtf
    [row2,col2] = max(TF(b,2:end));
    XF(b) = col2+1;
end

% Plot the histogram
figure(1) 
histogram(XB,'Normalization','probability') 
axis([0 35 0 0.45]) 
title('P_X_|_Y(x|grass)')
xlabel("Feture X")
ylabel("%")
figure(2) 
histogram(XF,'Normalization','probability') 
axis([0 35 0 0.2]) 
title('P_X_|_Y(x|cheetah)')
xlabel("Feture X")
ylabel("%")


%% Question c
% Load data
Imask = imread('cheetah.bmp'); 
Imask = im2double(Imask);
ZZ = load('Zig-Zag Pattern.txt'); 
ZZ = ZZ+1;
    
[x,y] = size(Imask);

% Using sliding window to rearrange data
count = 1;
for i=1:x-7
    for j=1:y-7
        SW = Imask(i:i+7,j:j+7);
        T = dct2(SW);
        T = abs(T);
        T(1) = 0;
        Rearrange(ZZ) = T;
        Rearrange2(count,:) = Rearrange;
        count = count+1;

    end
end

[x2,y2] = size(Rearrange2);

% Find the feature X
for k = 1:x2
    [x3,y3] = max(Rearrange2(k,:));
    X(k) = y3;
end

% Loss function calculation
[numF,orderF] = hist(XF,1:64);
[numB,orderB] = hist(XB,1:64);
count2 = 1;
for p = 1:64
    if PY*numF(p)/250 > PX*numB(p)/1053
        Compare(count2) = p;
        count2 = count2+1 ;
    end
end

% Create figure
count3 = 1; 
Seg(255,270)=0; 
for i=1:x-7
    for j=1:y-7
        if ismember(X(count3),Compare)
            Seg(i,j) = 1;
        else
            Seg(i,j) = 0;
        end
        count3 = count3 + 1;
    end
end
figure(3);
imagesc(Seg)
colormap(gray(255))

%% Question d
Imask = imread('cheetah_mask.bmp'); Imask = im2double(Imask);
% If the data in Seg and Imask are different, we define them as error. 
% Count the pixels that are different in two matrix.
count4 = 1;
for i = 1:x
    for j =1:y
        if Seg(i,j) ~= Imask(i,j)
           count4 = count4 + 1;
        end
    end
end

error = count4/(x*y)
