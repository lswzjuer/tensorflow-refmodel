function [ output_KN, KN_A_2s, KN_B_2s ] = KN_fp2s_fc(input_KN,POF_FC,KN_WD )


% clc
% clear
% %load('../value_VGG/VGG16_Bias.mat')
% %load('../value_VGG/VGG16_FT_data_prob.mat')
% load('../value_VGG/VGG16_KN.mat')
% KN_WD = 08;
% POF_FC = 512;
% input_KN = fc7_KN;

%% Function Begin

MAX_absKN = max(max(max(abs(input_KN))));
F_KN = floor(log2(1/MAX_absKN));

[NOF00,NIF]=size(input_KN);
NOF = ceil(NOF00/POF_FC)*POF_FC;
output_KN=round( input_KN * 2^(F_KN+KN_WD-1) )/2^(F_KN+KN_WD-1);
output_KN_INT=round( input_KN * 2^(F_KN+KN_WD-1) );

%Bytes_KN = (NK_tmp*KK*NOF/2)*(KN_WD/8); % bytes for one DDR3

KN_A00 = zeros(NOF,NIF);
% % KN_A00(1+0*POF_FC:1*POF_FC,:) = output_KN(1+0*POF_FC:1*POF_FC,:);
% % KN_A00(1+1*POF_FC:2*POF_FC,:) = output_KN(1+1*POF_FC:2*POF_FC,:);
for i = 0:1:NOF/POF_FC-1
    KN_A00(1+i*POF_FC:(i+1)*POF_FC,:) = output_KN(1+i*POF_FC:(i+1)*POF_FC,:);
end


KN_A01 = zeros(POF_FC,NIF*(NOF/POF_FC));
%KN_A01(:, 1+0*NIF:1*NIF) = KN_A00(1+0*POF_FC:1*POF_FC,:);
%KN_A01(:, 1+1*NIF:2*NIF) = KN_A00(1+1*POF_FC:2*POF_FC,:);
for i = 0 : 1 : (NOF/POF_FC)-1
    KN_A01(1:POF_FC, 1+i*NIF:(i+1)*NIF) = KN_A00(1+i*POF_FC:(i+1)*POF_FC,:);
end

KN_A02 = KN_A01';


row_KN = NIF*(NOF/POF_FC);
col_KN = POF_FC;
KN_A_2s=zeros(row_KN,col_KN);
for i=1:row_KN
    for j=1:col_KN
        mem_tmp=round(KN_A02(i,j)*2^(F_KN+KN_WD-1));
        if mem_tmp < 0
            KN_A_2s(i,j)=mem_tmp+2^KN_WD;
        else
            KN_A_2s(i,j)=mem_tmp;
        end
    end
end

KN_B_2s = [];

end






