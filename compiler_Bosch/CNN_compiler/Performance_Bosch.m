
Clk_Freq_MHz = 1200*1; %MHz

Extral = 1.0;

Itera_Tiles_CONV = ceil(NOX_CONV./Tox_CONV).*ceil(NOY_CONV./Toy_CONV).*ceil(NOF_CONV./Tof_CONV);
Cycle_1Iter_CONV = NKX_CONV.*NKY_CONV.*NKI_CONV.*ceil(NOX_CONV0 /POX).*ceil(Toy_CONV /POY).*ceil(Tof_CONV /POF);

Cycle_CONVs = Itera_Tiles_CONV.*Cycle_1Iter_CONV;
Cycle_CONV = sum(Cycle_CONVs)*Extral;

Delay_ms_1CONV_comp = 10^3*Cycle_CONVs/(Clk_Freq_MHz*10^6); % ms
Delay_ms_CONV_comp = 10^3*Cycle_CONV/(Clk_Freq_MHz*10^6); % ms


Itera_Tiles_DECV = ceil(NOX_DECV./Tox_DECV).*ceil(NOY_DECV./Toy_W_DECV).*ceil(NOF_DECV./Tof_DECV);
Cycle_1Iter_DECV = NKX_DECV.*NKY_DECV.*NKI_DECV.*ceil(NOX_DECV0/POX).*ceil(Toy_W_DECV/POY).*ceil(Tof_DECV/POF);

Cycle_DECVs = Itera_Tiles_DECV.*Cycle_1Iter_DECV;
Cycle_DECV = sum(Cycle_DECVs)*Extral;

Delay_ms_1DECV_comp = 10^3*Cycle_DECVs/(Clk_Freq_MHz*10^6); % ms
Delay_ms_DECV_comp = 10^3*Cycle_DECV/(Clk_Freq_MHz*10^6); % ms


Itera_Tiles_PLMX = ceil(NOX_PLMX./Tox_PLMX).*ceil(NOY_PLMX./Toy_PLMX).*ceil(NOF_PLMX./Tof_PLMX);
Cycle_1Iter_PLMX = ceil(NKX_PLMX).*ceil(NKY_PLMX).*ceil(Tox_PLMX/POX_PLMX) .*ceil(Toy_PLMX/POY_PLMX) .*ceil(Tof_PLMX/POF_PLMX);

Cycle_PLMXs = Itera_Tiles_PLMX.*Cycle_1Iter_PLMX;
%Cycle_PLMXs([4,5,9,10]) = 0;
Cycle_PLMX = sum(Cycle_PLMXs)*Extral;

Delay_ms_PLMX_comp = 10^3*Cycle_PLMX/(Clk_Freq_MHz*10^6); % ms

%fprintf('Delay_ms_CONV_comp = %4.2f, Delay_ms_PLMX_comp = %4.2f \n',Delay_ms_CONV_comp,Delay_ms_PLMX_comp)



%NOXP = ceil(NOX_CONV0./POX)*POX;
NUM_MAC_origin = NKX_CONV.*NKY_CONV.*NKI_CONV0.*NOX_CONV0                .*NOY_CONV0                  .*NOF_CONV0;
NUM_MAC_actual = NKX_CONV.*NKY_CONV.*NKI_CONV .*(ceil(NOX_CONV0/POX)*POX).*(Toy_CONV.*CR_CONV_num_toy).*(Tof_CONV.*CR_CONV_num_tof);

NUM_MAC_origin_DWIS = NKX_PLMX.*NKY_PLMX.*NOF_PLMX0.*NOX_PLMX0.*NOY_PLMX0.*CR_LAYER_IS_DWIS;
NUM_MAC_actual_DWIS = NKX_PLMX.*NKY_PLMX.*NOF_PLMX .*NOX_PLMX .*NOY_PLMX .*CR_LAYER_IS_DWIS*((POX*POY*POF)/(POX_PLMX*POF_PLMX));

NUM_WT_CONV        = sum(NKX_CONV.*NKY_CONV.*NKI_CONV0.*NOF_CONV0);
NUM_WT_CONV_DDR    = sum(NKX_CONV.*NKY_CONV.*NKI_CONV .*NOF_CONV );
NUM_INPX_CONV      = sum(NIF_CONV0.*NIX_CONV0.*NIY_CONV0);
NUM_INPX_CONV_DDR  = sum(NIF_CONV .*NIX_CONV .*NIY_CONV);
NUM_OUTPX_CONV     = sum(NOF_CONV0.*NOX_CONV0.*NOY_CONV0);
NUM_OUTPX_CONV_DDR = sum(NOF_CONV .*NOX_CONV .*NOY_CONV);

MAC_utilization_perlayer = NUM_MAC_origin./NUM_MAC_actual;
MAC_utilization = (sum(NUM_MAC_origin) + sum(NUM_MAC_origin_DWIS))/(sum(NUM_MAC_actual)+sum(NUM_MAC_actual_DWIS));
fprintf('The overall CONV MAC utilization is %4.2f%%. \n\n',MAC_utilization*100) % DDR memory transfer is not considered

NUM_WT_FCON    = sum(NIF_FCON0.*NOF_FCON0);
NUM_INPX_FCON  = sum(NIF_FCON0.*NBX_FCON0);
NUM_OUTPX_FCON = sum(NOF_FCON0.*NBX_FCON0);

NUM_INPX_PLMX  = sum(NIF_PLMX0.*NIX_PLMX0.*NIY_PLMX0);
NUM_OUTPX_PLMX = sum(NOF_PLMX0.*NOX_PLMX0.*NOY_PLMX0);

NUM_INPX_EWIS  = sum(NIF_EWIS0.*NIX_EWIS0.*NIY_EWIS0);
NUM_OUTPX_EWIS = sum(NOF_EWIS0.*NOX_EWIS0.*NOY_EWIS0);

%%

DDR3_NUM = 1;
DDR3_Width_bit = 256; % 64 bit
DDR3_Freq_MHz  = 600*0.7; % 1300 MHz % MAX = 266*8 = 2120; 44/64 = 0.6875 
DDR3_Bandwidth = (DDR3_Width_bit/8)*DDR3_Freq_MHz/10^3;

DMA_NUM = 1;
DMA_Width_bit = DMA_WIDTH;
DMA_Freq_MHz  = Clk_Freq_MHz;
DMA_Bandwidth = (DMA_Width_bit/8)*DMA_Freq_MHz/10^3;

DRAM_Bandwidth = min(DMA_Bandwidth,DDR3_Bandwidth); % 26.9 GB/s


% PER_len_pDPT_WR_CONV=[];
% for t = 1:NUM_TILE_CONV
%     for d = 1: NUM_dpt_WRpx_pTL(t)
%         PER_len_pDPT_WR_CONV = [PER_len_pDPT_WR_CONV;len_WRpx_pTil_CONV(t)];
%     end
% end
% Bytes_DRAM_WR_CONV = sum(PER_len_pDPT_WR_CONV);
% Delay_ms_WR_CONV = Bytes_DRAM_WR_CONV/(10^6*DRAM_Bandwidth) + 50*NUM_TILE_CONV/(Clk_Freq_MHz*10^3);
% %Delay_ms_WR_CONV = (Bytes_DRAM_WR_CONV*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3) + 50*NUM_TILE_CONV/(Clk_Freq_MHz*10^3); % 1226608/(200*10^3)
% 
% 
% PER_len_pDPT_RDpx_CONV=[];
% for t = 1:NUM_TILE_CONV
%     for d = 1: NUM_dpt_RDpx_pTL(t)
%         PER_len_pDPT_RDpx_CONV = [PER_len_pDPT_RDpx_CONV;len_RDpx_pTil(t)];
%     end
% end
% Bytes_DRAM_RDpx_CONV = sum(PER_len_pDPT_RDpx_CONV);
% Delay_ms_RDpx_CONV = Bytes_DRAM_RDpx_CONV/(10^6*DRAM_Bandwidth) + 50*NUM_TILE_CONV/(Clk_Freq_MHz*10^3);
% %Delay_ms_RDpx_CONV = (Bytes_DRAM_RDpx_CONV*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3) + 50*NUM_TILE_CONV/(Clk_Freq_MHz*10^3); % 2253347/(200*10^3)
% 
% 
% PER_len_pDPT_RDwt_CONV=[];
% for t = 1:NUM_TILE_CONV
%     for d = 1: NUM_dpt_RDwt_pTL(t)
%         PER_len_pDPT_RDwt_CONV = [PER_len_pDPT_RDwt_CONV;len_RDwt_pTil(t)];
%     end
% end
% Bytes_DRAM_RDwt_CONV = sum(PER_len_pDPT_RDwt_CONV);
% Delay_ms_RDwt_CONV = Bytes_DRAM_RDwt_CONV/(10^6*DRAM_Bandwidth) + 50*NUM_TILE_CONV/(Clk_Freq_MHz*10^3);
% %Delay_ms_RDwt_CONV = (Bytes_DRAM_RDwt_CONV*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3) + 30*NUM_TILE_CONV/(Clk_Freq_MHz*10^3); % (2727609-2253347)/(200*10^3)
% 
% PER_len_pDPT_RD_EW=[]; % TODO
% % for t = 1:NUM_TILE_CONV+1
% %     for d = 1: NUM_dpt_RDpx_pTL_EW(t)
% %         PER_len_pDPT_RD_EW = [PER_len_pDPT_RD_EW;len_RDpx_pTil_EW(t)];
% %     end
% % end
% Bytes_DRAM_RD_EW = sum(PER_len_pDPT_RD_EW);
% Delay_ms_RD_EW = Bytes_DRAM_RD_EW/(10^6*DRAM_Bandwidth);
% %Delay_ms_RD_EW = (Bytes_DRAM_RD_EW*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3);
% 
% 
% PER_len_pDPT_RD_PLMX=[];
% for t = 1:NUM_TILE_PLMX
%     for d = 1: NUM_dpt_RDpx_pTL_PLMX(t)
%         PER_len_pDPT_RD_PLMX = [PER_len_pDPT_RD_PLMX;len_RDpx_pTil_PLMX(t)];
%     end
% end
% Bytes_DRAM_RD_PLMX = sum(PER_len_pDPT_RD_PLMX);
% Delay_ms_RD_PLMX = Bytes_DRAM_RD_PLMX/(10^6*DRAM_Bandwidth) + 30*NUM_TILE_PLMX/(Clk_Freq_MHz*10^3);
% %Delay_ms_RD_PL = (Bytes_DRAM_RD_PL*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3) + 30*NUM_TILE_PLMX/(Clk_Freq_MHz*10^3);  % 477571/(200*10^3)
% 
% PER_len_pDPT_WR_PLMX=[];
% for t = 1:NUM_TILE_PLMX
%     for d = 1: NUM_dpt_WRpx_pTL_PLMX(t)
%         PER_len_pDPT_WR_PLMX = [PER_len_pDPT_WR_PLMX;len_WRpx_pTil_PLMX(t)];
%     end
% end
% Bytes_DRAM_WR_PLMX = sum(PER_len_pDPT_WR_PLMX);
% Delay_ms_WR_PLMX = Bytes_DRAM_WR_PLMX/(10^6*DRAM_Bandwidth) + 30*NUM_TILE_PLMX/(Clk_Freq_MHz*10^3);
% %Delay_ms_WR_PL = (Bytes_DRAM_WR_PL*8/DRAM_Width_bit)/(DRAM_NUM*DRAM_Freq_MHz*10^3) + 30*NUM_TILE_PLMX/(Clk_Freq_MHz*10^3); % 143360/(200*10^3)
% 
% Delay_ms_pTil_RDwt_FC = CR_FCON_bytes_1tile_wt_FCON/(10^6*DRAM_Bandwidth);
% Delay_ms_RD_FC = sum(NUM_tiles_wt_pFCON.*Delay_ms_pTil_RDwt_FC); 
% 
% Delay_ms_DRAM = Delay_ms_WR_CONV + Delay_ms_RDpx_CONV + Delay_ms_RDwt_CONV + Delay_ms_RD_EW + Delay_ms_RD_PLMX + Delay_ms_WR_PLMX + Delay_ms_RD_FC;
% 
% fprintf('Delay_ms_DRAM = %4.2f \n\n',Delay_ms_DRAM)


%%


Ext_DRAM_Bytes = 1*256/8; % interval between different dpt.

Byte_1T_RD_px_CONV = (NUM_dpt_RDpx_pCVpTil   .*(CR_CONV_inpx_cmd_size +Ext_DRAM_Bytes));
Byte_1T_RD_wt_CONV = (NUM_dpt_RDwt_pCVpTil   .*(CR_CONV_inwt_cmd_size +Ext_DRAM_Bytes));
Byte_1T_WR_px_CONV = (NUM_dpt_WRpx_pCVpTil   .*(CR_CONV_outpx_cmd_size+Ext_DRAM_Bytes));
Byte_1T_RD_px_DECV = (NUM_dpt_RDpx_pDECVpTil .*(CR_DECV_inpx_cmd_size +Ext_DRAM_Bytes));
Byte_1T_RD_wt_DECV = (NUM_dpt_RDwt_pDECVpTil .*(CR_DECV_inwt_cmd_size +Ext_DRAM_Bytes));
Byte_1T_WR_px_DECV = (NUM_dpt_WRpx_pDECVpTil .*(CR_DECV_outpx_cmd_size+Ext_DRAM_Bytes));
Byte_1T_RD_px_EWIS =               Tif_EWIS  .*(CR_EWIS_inpx_cmd_size +Ext_DRAM_Bytes);
Byte_1T_WR_px_EWIS =               Tof_EWIS  .*(CR_EWIS_outpx_cmd_size+Ext_DRAM_Bytes);
Byte_1T_RD_px_PLMX = (NUM_dpt_RDpx_pPLpTil   .*(CR_PLMX_inpx_cmd_size +Ext_DRAM_Bytes));
Byte_1T_WR_px_PLMX = (NUM_dpt_WRpx_pPLpTil   .*(CR_PLMX_outpx_cmd_size+Ext_DRAM_Bytes));
Byte_1T_RD_wt_FCON = (                         (CR_FCON_inwt_cmd_size +Ext_DRAM_Bytes));
Byte_1L_RD_px_FCON =  NIF_FCON.*NBX_FCON*(WD_PX/8);
Byte_1L_WR_px_FCON =  NOF_FCON.*NBX_FCON*(WD_PX/8);

Delay_ms_pTil_RDpx_CONV = Byte_1T_RD_px_CONV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDwt_CONV = Byte_1T_RD_wt_CONV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_WRpx_CONV = Byte_1T_WR_px_CONV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDpx_DECV = Byte_1T_RD_px_DECV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDwt_DECV = Byte_1T_RD_wt_DECV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_WRpx_DECV = Byte_1T_WR_px_DECV/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDpx_EWIS = Byte_1T_RD_px_EWIS/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_WRpx_EWIS = Byte_1T_WR_px_EWIS/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDpx_PLMX = Byte_1T_RD_px_PLMX/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_WRpx_PLMX = Byte_1T_WR_px_PLMX/(10^6*DRAM_Bandwidth);
Delay_ms_pTil_RDwt_FCON = Byte_1T_RD_wt_FCON/(10^6*DRAM_Bandwidth);
ms_1L_RD_px_FCON        = Byte_1L_RD_px_FCON/(10^6*DRAM_Bandwidth); % FCON RD px is not overlapped with any other DDR transfer and computation
ms_1L_WR_px_FCON        = Byte_1L_WR_px_FCON/(10^6*DRAM_Bandwidth); % FCON WR px is not overlapped with any other DDR transfer and computation

Delay_ms_FCON_MAC_1T = (Tif_FCON+(POF_FCON/POX)*(NUM_FC_BOX/POY))/(Clk_Freq_MHz*10^3);
Delay_ms_FCON_1T  = max(Delay_ms_pTil_RDwt_FCON,Delay_ms_FCON_MAC_1T);
Delay_ms_FCON_1L  = FCON_NUM_Tiles_1LAYER.*Delay_ms_FCON_1T + min(Delay_ms_pTil_RDwt_FCON,Delay_ms_FCON_MAC_1T) + ms_1L_RD_px_FCON + ms_1L_WR_px_FCON;
Delay_ms_FCON     = sum(Delay_ms_FCON_1L); 

MByte_RD_wt_FCON  = sum(FCON_NUM_Tiles_1LAYER.*Byte_1T_RD_wt_FCON)/10^6;
MByte_RD_px_FCON  = sum(Byte_1L_RD_px_FCON)/10^6;
MByte_WR_px_FCON  = sum(Byte_1L_WR_px_FCON)/10^6;

Delay_ms_pTil_MAC_CONV = Cycle_1Iter_CONV*Extral/(Clk_Freq_MHz*10^3); % ms
Delay_ms_pTil_MAC_DECV = Cycle_1Iter_DECV*Extral/(Clk_Freq_MHz*10^3); % ms
Delay_ms_pTil_POL_PLMX = Cycle_1Iter_PLMX*Extral/(Clk_Freq_MHz*10^3); % ms

Delay_ms_pExt_CONV = Delay_ms_pTil_RDpx_CONV+Delay_ms_pTil_RDwt_CONV+Delay_ms_pTil_WRpx_CONV;
Delay_ms_pExt_DECV = Delay_ms_pTil_RDpx_DECV+Delay_ms_pTil_RDwt_DECV+Delay_ms_pTil_WRpx_DECV;
Delay_ms_pExt_PLMX = Delay_ms_pTil_RDpx_PLMX+Delay_ms_pTil_WRpx_PLMX;


%Delay_ms_pTil_CONV = max(Delay_ms_pTil_MAC_CONV,Delay_ms_pTil_RDpx_CONV+Delay_ms_pTil_RDwt_CONV+Delay_ms_pTil_WRpx_CONV);
%Delay_ms_pTil_PM = max(Delay_ms_pTil_POL_PM,Delay_ms_pTil_RDpx_PM+Delay_ms_pTil_WRpx_PM);

%Delay_ms_pLAY_CONV = CONV_NUM_Tiles_1LAYER.*Delay_ms_pTil_CONV + Delay_ms_pExt_CONV;
%Delay_ms_pLAY_PM = PLMX_NUM_Tiles_1LAYER.*Delay_ms_pTil_PM + Delay_ms_pExt_PM;

%Delay_ms_CONV = sum (Delay_ms_pLAY_CONV);
%Delay_ms_PM = sum (Delay_ms_pLAY_PM);


Delay_ms_pLAY_CONV = zeros(NUM_CONV,1);
Byte_RD_px_1CONV   = zeros(NUM_CONV,1);
Byte_RD_wt_1CONV   = zeros(NUM_CONV,1);
Byte_WR_px_1CONV   = zeros(NUM_CONV,1);
for L = 1 : NUM_CONV
    Delay_ms_pTil = zeros(CONV_NUM_Tiles_1LAYER(L),1);
    Byte_RD_px_1T = zeros(CONV_NUM_Tiles_1LAYER(L),1);
    Byte_RD_wt_1T = zeros(CONV_NUM_Tiles_1LAYER(L),1);
    Byte_WR_px_1T = zeros(CONV_NUM_Tiles_1LAYER(L),1);
    if CONV_NUM_Tiles_1LAYER(L) == 1
        Delay_ms_pTil(1) = Delay_ms_pTil_MAC_CONV(L);
    elseif CR_CONV_num_tof(L) == 1
        for T = 1:CONV_NUM_Tiles_1LAYER(L)
            if T == 1
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_RDpx_CONV(L));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_CONV(L);
            elseif T == CONV_NUM_Tiles_1LAYER(L)
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_WRpx_CONV(L));
                Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
            else % rd and wr ddr simultaneously
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),max(Delay_ms_pTil_RDpx_CONV(L),Delay_ms_pTil_WRpx_CONV(L)));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_CONV(L);
                Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
            end
        end
    elseif CR_CONV_num_toy(L) == 1
        for T = 1:CONV_NUM_Tiles_1LAYER(L)
            if T == 1
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_RDwt_CONV(L));
                Byte_RD_wt_1T(T) = Byte_1T_RD_wt_CONV(L);
            elseif T == CONV_NUM_Tiles_1LAYER(L)
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_WRpx_CONV(L));
                Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
            else
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),max(Delay_ms_pTil_RDwt_CONV(L),Delay_ms_pTil_WRpx_CONV(L)));
                Byte_RD_wt_1T(T) = Byte_1T_RD_wt_CONV(L);
                Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
            end
        end
    else
        for Ty = 1:CR_CONV_num_toy(L)
            for Tf = 1:CR_CONV_num_tof(L)
                T = Tf + (Ty-1)*CR_CONV_num_tof(L);
                if Ty == 1 && Tf == 1
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_RDpx_CONV(L));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_CONV(L);
                elseif Ty == CR_CONV_num_toy(L) && Tf == CR_CONV_num_tof(L)
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),Delay_ms_pTil_WRpx_CONV(L));
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
                elseif Ty == CR_CONV_num_toy(L) % rd and wr ddr simultaneously
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L),max(Delay_ms_pTil_RDpx_CONV(L)+Delay_ms_pTil_RDwt_CONV(L),Delay_ms_pTil_WRpx_CONV(L)));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_CONV(L);
                    Byte_RD_wt_1T(T) = Byte_1T_RD_wt_CONV(L);
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
                else % rd and wr ddr simultaneously
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_CONV(L), max(Delay_ms_pTil_RDpx_CONV(L),Delay_ms_pTil_WRpx_CONV(L)));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_CONV(L);
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_CONV(L);
                end
            end
        end        
    end
    Delay_ms_pLAY_CONV(L) = sum(Delay_ms_pTil) + Delay_ms_pExt_CONV(L);
    Byte_RD_px_1CONV(L)   = sum(Byte_RD_px_1T) + Byte_1T_RD_px_CONV(L);
    Byte_RD_wt_1CONV(L)   = sum(Byte_RD_wt_1T) + Byte_1T_RD_wt_CONV(L);
    Byte_WR_px_1CONV(L)   = sum(Byte_WR_px_1T) + Byte_1T_WR_px_CONV(L);
end
Delay_ms_CONV = sum (Delay_ms_pLAY_CONV);
MByte_RD_px_CONV = sum(Byte_RD_px_1CONV)/10^6;
MByte_RD_wt_CONV = sum(Byte_RD_wt_1CONV)/10^6;
MByte_WR_px_CONV = sum(Byte_WR_px_1CONV)/10^6;

Delay_ms_pLAY_DECV = zeros(NUM_DECV,1);
Byte_RD_px_1DECV   = zeros(NUM_DECV,1);
Byte_RD_wt_1DECV   = zeros(NUM_DECV,1);
Byte_WR_px_1DECV   = zeros(NUM_DECV,1);
for L = 1 : NUM_DECV
    Delay_ms_pTil = zeros(DECV_NUM_Tiles_1LAYER(L),1);
    Byte_RD_px_1T = zeros(DECV_NUM_Tiles_1LAYER(L),1);
    Byte_RD_wt_1T = zeros(DECV_NUM_Tiles_1LAYER(L),1);
    Byte_WR_px_1T = zeros(DECV_NUM_Tiles_1LAYER(L),1);
    if DECV_NUM_Tiles_1LAYER(L) == 1
        Delay_ms_pTil(1) = Delay_ms_pTil_MAC_DECV(L);
    elseif CR_DECV_num_tof(L) == 1
        for T = 1:DECV_NUM_Tiles_1LAYER(L)
            if T == 1
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_RDpx_DECV(L));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_DECV(L);
            elseif T == DECV_NUM_Tiles_1LAYER(L)
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_WRpx_DECV(L));
                Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
            else % rd and wr ddr simultaneously
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),max(Delay_ms_pTil_RDpx_DECV(L),Delay_ms_pTil_WRpx_DECV(L)));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_DECV(L);
                Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
            end
        end
    elseif CR_DECV_num_toy(L) == 1
        for T = 1:DECV_NUM_Tiles_1LAYER(L)
            if T == 1
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_RDwt_DECV(L));
                Byte_RD_wt_1T(T) = Byte_1T_RD_wt_DECV(L);
            elseif T == DECV_NUM_Tiles_1LAYER(L)
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_WRpx_DECV(L));
                Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
            else
                Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),max(Delay_ms_pTil_RDwt_DECV(L),Delay_ms_pTil_WRpx_DECV(L)));
                Byte_RD_wt_1T(T) = Byte_1T_RD_wt_DECV(L);
                Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
            end
        end
    else
        for Ty = 1:CR_DECV_num_toy(L)
            for Tf = 1:CR_DECV_num_tof(L)
                T = Tf + (Ty-1)*CR_DECV_num_tof(L);
                if Ty == 1 && Tf == 1
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_RDpx_DECV(L));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_DECV(L);
                elseif Ty == CR_DECV_num_toy(L) && Tf == CR_DECV_num_tof(L)
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),Delay_ms_pTil_WRpx_DECV(L));
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
                elseif Ty == CR_DECV_num_toy(L) % rd and wr ddr simultaneously
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L),max(Delay_ms_pTil_RDpx_DECV(L)+Delay_ms_pTil_RDwt_DECV(L),Delay_ms_pTil_WRpx_DECV(L)));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_DECV(L);
                    Byte_RD_wt_1T(T) = Byte_1T_RD_wt_DECV(L);
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
                else % rd and wr ddr simultaneously
                    Delay_ms_pTil(T) = max(Delay_ms_pTil_MAC_DECV(L), max(Delay_ms_pTil_RDpx_DECV(L),Delay_ms_pTil_WRpx_DECV(L)));
                    Byte_RD_px_1T(T) = Byte_1T_RD_px_DECV(L);
                    Byte_WR_px_1T(T) = Byte_1T_WR_px_DECV(L);
                end
            end
        end        
    end
    Delay_ms_pLAY_DECV(L) = sum(Delay_ms_pTil) + Delay_ms_pExt_DECV(L);
    Byte_RD_px_1DECV(L)   = sum(Byte_RD_px_1T) + Byte_1T_RD_px_DECV(L);
    Byte_RD_wt_1DECV(L)   = sum(Byte_RD_wt_1T) + Byte_1T_RD_wt_DECV(L);
    Byte_WR_px_1DECV(L)   = sum(Byte_WR_px_1T) + Byte_1T_WR_px_DECV(L);
end
Delay_ms_DECV = sum (Delay_ms_pLAY_DECV);
MByte_RD_px_DECV = sum(Byte_RD_px_1DECV)/10^6;
MByte_RD_wt_DECV = sum(Byte_RD_wt_1DECV)/10^6;
MByte_WR_px_DECV = sum(Byte_WR_px_1DECV)/10^6;


Delay_ms_pLAY_PLMX = zeros(NUM_PLMX,1);
Byte_RD_px_1PLMX   = zeros(NUM_PLMX,1);
Byte_WR_px_1PLMX   = zeros(NUM_PLMX,1);
for L = 1 : NUM_PLMX
    Delay_ms_pTil = zeros(PLMX_NUM_Tiles_1LAYER(L),1);
    Byte_RD_px_1T = zeros(PLMX_NUM_Tiles_1LAYER(L),1);
    Byte_WR_px_1T = zeros(PLMX_NUM_Tiles_1LAYER(L),1);
    if PLMX_NUM_Tiles_1LAYER(L) == 1
        Delay_ms_pTil(1) = Delay_ms_pTil_POL_PLMX(L);
    else
        for T = 1:PLMX_NUM_Tiles_1LAYER(L)
            if T == 1
                Delay_ms_pTil(T) = max(Delay_ms_pTil_POL_PLMX(L),Delay_ms_pTil_RDpx_PLMX(L));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_PLMX(L);
            elseif T == PLMX_NUM_Tiles_1LAYER(L)
                Delay_ms_pTil(T) = max(Delay_ms_pTil_POL_PLMX(L),Delay_ms_pTil_WRpx_PLMX(L));
                Byte_WR_px_1T(T) = Byte_1T_WR_px_PLMX(L);
            else
                Delay_ms_pTil(T) = max(Delay_ms_pTil_POL_PLMX(L),max(Delay_ms_pTil_RDpx_PLMX(L),Delay_ms_pTil_WRpx_PLMX(L)));
                Byte_RD_px_1T(T) = Byte_1T_RD_px_PLMX(L);
                Byte_WR_px_1T(T) = Byte_1T_WR_px_PLMX(L);
            end
        end
    end
    Delay_ms_pLAY_PLMX(L) = sum(Delay_ms_pTil) + Delay_ms_pExt_PLMX(L);
    Byte_RD_px_1PLMX(L)   = sum(Byte_RD_px_1T) + Byte_1T_RD_px_PLMX(L);
    Byte_WR_px_1PLMX(L)   = sum(Byte_WR_px_1T) + Byte_1T_WR_px_PLMX(L);
end
Delay_ms_PLMX = sum (Delay_ms_pLAY_PLMX);
MByte_RD_px_PLMX = sum(Byte_RD_px_1PLMX)/10^6;
MByte_WR_px_PLMX = sum(Byte_WR_px_1PLMX)/10^6;

Byte_RD_px_1EWIS = Byte_1T_RD_px_EWIS.*CR_EWIS_num_toy;
Byte_WR_px_1EWIS = Byte_1T_WR_px_EWIS.*CR_EWIS_num_toy;
MByte_RD_px_EWIS = sum(Byte_RD_px_1EWIS)/10^6;
MByte_WR_px_EWIS = sum(Byte_WR_px_1EWIS)/10^6;
Delay_ms_EWIS_1L = max(Delay_ms_pTil_RDpx_EWIS,Delay_ms_pTil_WRpx_EWIS).*CR_EWIS_num_toy;
Delay_ms_EWIS = sum(Delay_ms_EWIS_1L);

%Cycle_PLAV = NKX_PLAV*NKY_PLAV*(TOXGRP_AV_M1+1)*(TOY1STR_AV_M1+1)*(TOFGRP_AV_M1+1)*CONV_NUM_Tiles_1LAYER(end)*Extral;
Cycle_PLAV = 0;
Delay_ms_PLAV = 10^3*Cycle_PLAV/(Clk_Freq_MHz*10^6); % ms

Delay_ms_total = Delay_ms_CONV + Delay_ms_DECV + Delay_ms_PLMX + Delay_ms_PLAV + Delay_ms_EWIS + Delay_ms_FCON;

GOP_pCONV = 2*NUM_MAC_origin/10^9;
Per_pCONV = GOP_pCONV/sum(GOP_pCONV);
GOPS_pCONV = 10^3*GOP_pCONV./Delay_ms_pLAY_CONV;
GOP_pPLMX = NKX_PLMX.*NKY_PLMX.*NOX_PLMX0.*NOY_PLMX0.*NIF_PLMX0;
GOP_CONV = 2*sum(NUM_MAC_origin)/10^9;
GOP_PLMX = sum(GOP_pPLMX)/10^9;
GOP_FCON = 2*sum(NIF_FCON.*NOF_FCON)/10^9;
GOP_all = GOP_CONV + GOP_PLMX + GOP_FCON;
GOPS_CONV = GOP_CONV/(Delay_ms_CONV/10^3);
GOPS_all = GOP_all/(Delay_ms_total/10^3);



fprintf('Assume frequency = %4.2f MHz\n',Clk_Freq_MHz)
fprintf('Assume DDR effective bandwidth = %4.2f GB/s\n', DRAM_Bandwidth)
fprintf('The overall CONV MAC utilization is %4.2f%%. \n',MAC_utilization*100) % DDR memory transfer is not considered
fprintf('Delay of CONV  = %4.2f ms. \n',Delay_ms_CONV)
fprintf('Delay of DECV  = %4.2f ms. \n',Delay_ms_DECV)
fprintf('Delay of PLMX  = %4.2f ms. \n',Delay_ms_PLMX)
fprintf('Delay of PLAV  = %4.2f ms. \n',Delay_ms_PLAV)
fprintf('Delay of EWIS  = %4.2f ms. \n',Delay_ms_EWIS)
fprintf('Delay of FCON  = %4.2f ms. \n',Delay_ms_FCON)
fprintf('Estimated total delay = %4.2f ms. \n',Delay_ms_total)
fprintf('Estimated throughput  = %4.2f GOPS. \n\n',GOPS_all)


% GOP_pCONV_DSP = 2*(NKX_CONV.*NKY_CONV.*NIF_CONV0).*(ceil(NOX_CONV0./POX).*POX).*(ceil(NOY_CONV0./POY).*POY).*(ceil(NOF_CONV0./NOF_CONV).*NOF_CONV);
% GOP_pCONV_DSP = GOP_pCONV_DSP/10^9;
% DSP_Efficiency_pCONV = GOP_pCONV./GOP_pCONV_DSP;
% DSP_Efficiency = sum(GOP_pCONV)./sum(GOP_pCONV_DSP);

%%

MByte_RD_px_DDR = (MByte_RD_px_CONV + MByte_RD_px_FCON + MByte_RD_px_PLMX + MByte_RD_px_EWIS);
MByte_RD_wt_DDR = (MByte_RD_wt_CONV + MByte_RD_wt_FCON);
MByte_WR_px_DDR = (MByte_WR_px_CONV + MByte_WR_px_PLMX + MByte_WR_px_FCON + MByte_WR_px_EWIS);


fprintf('# of CONV operations is %4.2f GOP\n', GOP_CONV)
fprintf('Original size of input pixels  is %4.2f MByte\n', (NUM_INPX_CONV+NUM_INPX_PLMX+NUM_INPX_EWIS+NUM_INPX_FCON)*(WD_PX/8)/10^6)
fprintf('Original size of weights       is %4.2f MByte\n', (NUM_WT_CONV+NUM_WT_FCON)*(WD_WT/8)/10^6)
fprintf('Original size of output pixels is %4.2f MByte\n', (NUM_OUTPX_CONV+NUM_OUTPX_PLMX+NUM_OUTPX_EWIS+NUM_OUTPX_FCON)*(WD_PX/8)/10^6)
fprintf('\n')
fprintf('Data size of DDR read  pixels  is %4.2f MByte\n', MByte_RD_px_DDR)
fprintf('Data size of DDR read  weights is %4.2f MByte\n', MByte_RD_wt_DDR)
fprintf('Data size of DDR write pixels  is %4.2f MByte\n', MByte_WR_px_DDR)

%%

Average_BW_DDR = (MByte_RD_px_DDR+MByte_RD_wt_DDR+MByte_WR_px_DDR)/Delay_ms_total;
fprintf('Average bandwidth [GB/s] external memory is %4.2f GB/s\n', Average_BW_DDR)


MByte_DDR = (max(CR_LAYER_outpx_addr)+Bytes_WT_CONV+Bytes_WT_DECV+Bytes_WT_FCON+Bytes_PROP+Bytes_image)/10^6;
fprintf('Average memory footprint [MB] external memory is %4.2f MB\n', MByte_DDR)


Average_BW_Buffer = ((Delay_ms_CONV_comp+Delay_ms_PLMX_comp)/Delay_ms_total)*((4*POX*POY*2+POF)*(Clk_Freq_MHz/1000));
fprintf('Average bandwidth [GB/s] local memory is %4.2f GB/s\n', Average_BW_Buffer)


bit_buffer = 2*(BUF_INPUT_width_CONV.*BUF_INPUT_num_CONV.*BUF_INPUT_depth_CONV + BUF_OUTPUT_width_CONV.*BUF_OUTPUT_num_CONV.*BUF_OUTPUT_depth_CONV + BUF_WEIGHT_width_CONV.*BUF_WEIGHT_num_CONV.*BUF_WEIGHT_depth_CONV)*WD_PX;
MByte_buffer_average = sum(bit_buffer)/(8*NUM_CONV*10^6);
fprintf('Average memory footprint [MB] local memory is %4.2f MB\n', MByte_buffer_average)
MByte_buffer_peak = max(bit_buffer)/(8*10^6);
fprintf('Peak memory footprint [MB] local memory is %4.2f MB\n', MByte_buffer_peak)




