function [ output_KN, KN_A_2s, KN_B_2s ] = KN_fp2s_conv(input_KN,POF,NOF,KN_WD )


% clc
% clear
% load('GenNet_KN_Image.mat')
% KN_WD = 16;
% POF = 64;
% input_KN = KN_conv1;

%% Function Begin

MAX_absKN = max(max(max(abs(input_KN))));
F_KN = floor(log2(1/MAX_absKN));
KN_FRCbits = F_KN+KN_WD-1;

[NK_tmp, KK, NOF00]=size(input_KN);
%output_KN=round( input_KN * 2^KN_FRCbits )/2^KN_FRCbits;
output_KN = input_KN;

%NOF = ceil(NOF00/POF)*POF;
input_trunc_KN = zeros(NK_tmp, KK, NOF);
for i = 1:NOF00
    input_trunc_KN(:,:,i) = output_KN(:,:,i);
end

%Bytes_KN = (NK_tmp*KK*NOF/2)*(KN_WD/8); % bytes for one DDR3

KN_A00 = zeros(NK_tmp,KK,NOF);

% conv1_1_KN_A(:,:,1+0*POF:1*POF) = conv1_1_KN(:,:,1+0*POF:1*POF);
% conv1_1_KN_A(:,:,1+1*POF:2*POF) = conv1_1_KN(:,:,1+1*POF:2*POF);
for i = 0:1:NOF/POF-1
    KN_A00(:,:,1+i*POF:(i+1)*POF) = input_trunc_KN(:,:,1+i*POF:(i+1)*POF); 
end


KN_A01 = zeros(NK_tmp*KK,NOF);
for i = 1:NOF
    KN_A01(:,i) = reshape(KN_A00(:,:,i)',NK_tmp*KK,1 );
end

KN_A02 = zeros(NK_tmp*KK*NOF/POF,POF);
% conv1_1_KN_A02(1+0*NK_tmp*KK:1*NK_tmp*KK, :) =  conv1_1_KN_A01(:, 1+0*POF:1*POF);
% conv1_1_KN_A02(1+1*NK_tmp*KK:2*NK_tmp*KK, :) =  conv1_1_KN_A01(:, 1+1*POF:2*POF);
for i = 0:1:NOF/POF-1
    KN_A02(1+i*NK_tmp*KK:(i+1)*NK_tmp*KK, 1:POF) =  KN_A01(:, 1+i*POF:(i+1)*POF);
end

row_KN = NK_tmp*KK*NOF/POF;
col_KN = POF;
KN_A_2s=zeros(row_KN,col_KN);
for i=1:row_KN
    for j=1:col_KN
        %mem_tmp=round(KN_A02(i,j)*2^KN_FRCbits);
        mem_tmp = KN_A02(i,j);
        if mem_tmp < 0
            KN_A_2s(i,j) = mem_tmp+2^KN_WD;
        else
            KN_A_2s(i,j) = mem_tmp;
        end
    end
end

KN_B_2s = [];

end






