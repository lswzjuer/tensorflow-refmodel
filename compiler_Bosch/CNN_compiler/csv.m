%% BAHAA added code to generate csv file

fid = fopen('parameters.csv', 'W');

fprintf(fid, 'idx,CONV_WITH_RELU,FCON_WITH_RELU,CONV_WITH_BIAS,WITH_BNOM,CONV_WITH_ELEMENT_WISE,CONV_LAYER,PLMX_LAYER,FCON_AFTER_CONV,FCON_AFTER_PLMX,DATA_ROUTER_SELECTION,INBUF_RDADDR_KEY,CONV_STR_IS_1,CONV_STR_IS_2,DRFIFO_WREN_THRESHOLD,DRFIFO_RDEN_THRESHOLD,TOY_IS_NOY,CONV_DONE_COUNT,NKX,NKY,PAD_DIV_STRIDE,TIF,STRIDE,MAX_PAD_STR_M1,MAX_PAD_STR,TIY,TOX,TOX_DIV_POX,END_TOYPAD,END_NOYPAD,PAD_E_CV_POX,PAD_S_CV_POY,PAD_E_CV_NKX_HIGH,PAD_E_CV_NKX_LOW,PAD_S_CV_NKY_HIGH,PAD_S_CV_NKY_LOW,PAD_N_CV_POY,PAD_N_CV_NKY,HAS_K3S1_PLMX,IS_K3S1_PLMX,NKX_PLMX,NKY_PLMX,R_WRDRAM_PLMX,PAD_E_END_PLMX,PAD_S_END_PLMX,NUM_TOXGRP_PLMX,NUM_TOYGRP_PLMX,NUM_TOFGRP_PLMX,NUM_TIXGRP_PLMX,TIXGRP1ST_PLMX,TIXY1ST_PLMX,LOC_PAD_E_PLMX,LOC_PAD_S_PLMX,NUM_TILE_FC_MAC,NIF_FCON,NOF_FCON,NIFGRP_FC,NIFGRP_POF,TOFGRPM1_TOXTOY14POX,TOXTOY14POX,TOFTOXTOY132POX,WORDS_RDPX,WORDS_WRPX,NUM_ROWSTR,END_ROW,END_ROW_STR,TOY_GRP,TOF_GRP,TOF1OUPX_BSCV,ACCU_TOF_BSCV,NOY_GRP,NOF_GRP,TIX1POX_STR,TIX1POX_TIY1POY,END_ROWPOY,TOXGRP_POY,TOXGRP_TOY,KX_KY_TIF,KKNIF_DELAY,MACOUT_RSBIT_CV,MACOUT_RSBIT_FC,LEN_RDPX_CV,LEN_RDWT_CV,LEN_WRPX_CV,LEN_RDPX_PLMX,LEN_WRPX_PLMX,CSR_STATUS_REG,CSR_CONTROL_REG,CSR_DESCRIPTOR_FILL_LEVEL_REG,CSR_RESPONSE_FILL_LEVEL_REG,CSR_SEQUENCE_NUMBER_REG,CSR_GLOBAL_INTERRUPT_MASK,CSR_IRQ_SET_MASK,CSR_IRQ_SET_OFFSET,DESCRIPTOR_CONTROL_GO_EARLY_DONE_ENABLE,DESCRIPTOR_CONTROL_GO_TRANSFER_COMPLETE_IRQ,POX,POY,POF,POX_PLMX,POY_PLMX,POF_PLMX,POF_EW,NUM_CONV,NUM_PLMX,NUM_FCON,NUM_LAYER,STPL,KX_GAVP,KY_GAVP,TOXGRP_AV_M1,TOY1STR_AV_M1,TOFGRP_AV_M1,RDAD_ADD_AV,INPX1_0FILL,END_RDAD_AV,WD_DIV,DIVISION_AV,WORDS_R_GAVP,WDAD_R_GAVP,WORDS_W_GAVP,WDAD_W_GAVP,WORD_BSCV,WDAD_BSCV,WDAD_BN_MULADD,WORD_BN_MULADD,NUM_MACREG_POX,PX_WD,KN_WD_CV,KN_WD_FC,WD_BSCV,WD_BSFC,NUM_MACREG_POX,WD_BNADD,MAC_ACCU_WD,PX_AD,POX_2POWER,NUM_POX_DMA,HAS_INPX_SCALE,CVID_INPX_SCALE_1,CVID_INPX_SCALE_2,HAS_DILATED,ID_SSD_FC6,HAS_GAVP,with_GAVP,IS_PVANET_OB,IS_PVANET_TL,POF_FCON,BUF_FCWT_WORDS,NIF_FC1,NOF_FC1,NIF_FC2,NOF_FC2,NIF_FC3,NOF_FC3,NIF_FC4,NOF_FC4,NIF_FC5,NOF_FC5,NIF_FC6,NOF_FC6,NOFAD_FC1,NOFAD_FC2,NOFAD_FC3,NOFAD_FC4,NOFAD_FC5,NOFAD_FC6,NOY_ENDPLMX,NOF_ENDPLMX,NUM_DPT_WR_CV,NUM_DPT_RD_CV,NUM_DPT_WR_PLMX,NUM_DPT_RD_PLMX,NUM_DPT_RD_EW,WDAD_DPT_WR_CV,WDAD_DPT_RD_CV,WDAD_DPT_WR_PLMX,WDAD_DPT_RD_PLMX,WDAD_DPT_RD_EW,NUM_TILE_CONV,NUM_TILE_PLMX,TILING_BYTES_FC,NUM_TILING_FC,NUM_BK,BUF_INPX_ALL,BUF_OUPX_ALL,BUF_INPX_1BK,BUF_OUPX_1BK,POF1OUPX,BYTES_IMAGE,INWT_WORDS,INWT_WDAD,INWT_WD,WD_RDCVWT,INPX_WORDS,INPX_WDAD,INPX_WIDTH,OUPX_WIDTH,OUPX_WORD,OUPX_WDAD,DDR3_BADDR_KN_FC6,BUF_WTFC_A_BASEADDR,BUF_WT_B_BASEADDR,BUF_PX_A_BASEADDR,BUF_PX_B_BASEADDR,BUF_OUTFC_A_BASEADDR,BUF_OUTFC_B_BASEADDR,MACOU_RSBIT,BUF_PX_AB_BDEC,BUF_WT_AB_BDEC,BUF_OUTFC_AB_BDEC,DDR3_ENDADDR_WT,BYTES_BBOXES,BYTES_ANCHOR,BYTES_SCORES,DDR3_BADDR_BBOXES,DDR3_BADDR_ANCHOR,DDR3_BADDR_SCORES,INPX_PROP_1MAP_DMA_M1,PROPOSAL_CH_HEIGHT,PROPOSAL_CH_WIDTH,PROPOSAL_NUM_ANCHORS,PROPOSAL_NUM_ANCHORS_PER_BUFFER,ROI_NUM_DOWNSCALES,ROI_NUM_CHANNELS,ROI_CH_HEIGHT,ROI_CH_WIDTH,ROI_NUM_CHANNELS_PER_BUFFER,FCON_AFTER_ROIPL\n');

%PROPOSAL_CH_HEIGHT (csr_ch_height) (OB:66, TL:132)
%PROPOSAL_CH_WIDTH (csr_ch_width) (OB: 64, TL:96)
%PROPOSAL_NUM_ANCHORS (csr_num_anchors) (OB:49, TL:7)
%PROPOSAL_NUM_ANCHORS_PER_BUFFER (csr_num_anchors_per_buffer) (OB:15, TL:5)

%PROPOSAL_NUM_STDDEV (csr_num_stddev)
%PROPOSAL_NMS_THRESHOLD (csr_nms_threshold)
%PROPOSAL_NUM_POST_NMS (csr_numostNMS)
%ROI_GRID_DIVISION_EQUIVALENT (csr_grid_divisin_equivalent)
%ROI_NUM_DOWNSCALES (csr_num_downscales)
%ROI_NUM_CHANNELS (csr_num_channels)
%ROI_NUM_CHANNELS_PER_BUFFER (csr_num_channels_per_buffer)
%ROI_CH_HEIGHT (csr_ch_h)
%ROI_CH_WIDTH (csr_ch_w)

if is_PVANET_OB
  PROPOSAL_CH_HEIGHT = NOY_OB;
  PROPOSAL_CH_WIDTH  = NOX_OB;
  PROPOSAL_NUM_ANCHORS = ANC_OB;
  PROPOSAL_NUM_ANCHORS_PER_BUFFER = 15;
  ROI_NUM_DOWNSCALES=4;
  ROI_NUM_CHANNELS=256;
  ROI_CH_HEIGHT=NOY_OB;
  ROI_CH_WIDTH=NOX_OB;
  ROI_NUM_CHANNELS_PER_BUFFER=8;
else
  PROPOSAL_CH_HEIGHT = NOY_TL;
  PROPOSAL_CH_WIDTH  = NOX_TL;
  PROPOSAL_NUM_ANCHORS = ANC_TL;
  PROPOSAL_NUM_ANCHORS_PER_BUFFER = 5;
  ROI_NUM_DOWNSCALES=3;
  ROI_NUM_CHANNELS=256;
  ROI_CH_HEIGHT=132;
  ROI_CH_WIDTH=80;
  ROI_NUM_CHANNELS_PER_BUFFER = 4;
end

conv_cnt = 0;
plmx_cnt = 0;
fc_cnt = 0;
n_conv_inc = 0;
n_plmx_inc = 0;
n_fc_inc = 0;
DATA_ROUTER_SELECTION = '00000000';

for i = 1:NUM_LAYER


%prints the layer no
fprintf(fid,'%d,',i);

% Assuming the last layers to be Fully connected ones
 if i > NUM_LAYER
    i = NUM_LAYER;
    fc_cnt = fc_cnt + 1 - n_fc_inc;
    n_fc_inc = 0;
else
  if CR_LAYER_IS_CONV(i)
    conv_cnt = conv_cnt + 1 -n_conv_inc;
    n_conv_inc = 0;
  elseif CR_LAYER_IS_PLMX(i)
    plmx_cnt = plmx_cnt + 1 - n_plmx_inc;
    n_plmx_inc = 0;
  end

end
 % else
 %   fc_cnt = fc_cnt+1 - n_fc_inc;
 %   n_fc_inc = 1;
 % end
  
  if conv_cnt == 0 
      conv_cnt = 1;
      n_conv_inc = 1;
  end
  if plmx_cnt == 0
      plmx_cnt = 1;
      n_plmx_inc = 1;
  end
  if fc_cnt == 0
      fc_cnt = 1;
      n_fc_inc = 1;
  end

    if CR_CONV_with_ReLU(conv_cnt) == 1
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end
    if CR_FCON_with_ReLU(fc_cnt) == 1
        fprintf(fid,'1,');
    else
    fprintf(fid,','); % Empty for with_RELU_FC
    end
    if CR_CONV_with_Bias(conv_cnt) == 1
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end

% GTH REVIEW
%     if CR_CONV_with_BNOM(conv_cnt) == 1
%         fprintf(fid,'1,');
%     else
%         fprintf(fid,'0,');
%     end

%     if with_EltWise(conv_cnt) == 1
%         fprintf(fid,'1,');
%     else
%         fprintf(fid,'0,');
%     end
if  i == NUM_LAYER && n_fc_inc == 0

        fprintf(fid,'0,');
        fprintf(fid,'0,');

else
    if CR_LAYER_IS_CONV(i) == 1
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end

    if CR_LAYER_IS_PLMX(i) == 1
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end
end
% GTH REVIEW
%     FCON_after_CONV_reg = 0;
%     if CR_LAYER_IS_CONV(end) == 1 && NUM_FCON ~= 0
%         FCON_after_CONV_reg = 1;
%     end
%     fprintf(fid,'%d,',FCON_after_CONV_reg);
% 
%     FCON_after_PLMX_reg = 0;
%     if CR_LAYER_IS_PLMX(end) == 1 && NUM_FCON ~= 0
%         FCON_after_PLMX_reg = 1;
%     end
%     fprintf(fid,'%d,',FCON_after_PLMX_reg);

% GTH REVIEW - change NKX to NKX_CONV, STR_CONV, PAD_CONV
%% conv array
    %IS_K11S4P0  : route 6 with bit position 110
    if NKX(conv_cnt) <= 11 && STR(conv_cnt) == 4 && PAD(conv_cnt) == 0
        DATA_ROUTER_SELECTION(7) = '1';
    else
        DATA_ROUTER_SELECTION(7) = '0';
    end

    %IS_K05S1P2  : route 3 with bit position 011
    if NKX(conv_cnt) <= 5 && STR(conv_cnt) == 1 && PAD(conv_cnt) == 2
        DATA_ROUTER_SELECTION(4) = '1';
    else
        DATA_ROUTER_SELECTION(4) = '0';
    end

    %IS_K07S2P3  : route 5 with bit position 101
    if NKX(conv_cnt) <= 7 && STR(conv_cnt) == 2 && PAD(conv_cnt) == 3
        DATA_ROUTER_SELECTION(6) = '1';
    else
        DATA_ROUTER_SELECTION(6) = '0';
    end

    %IS_K03S1P1  : route 2 with bit position 010
    if NKX(conv_cnt) <= 3 && STR(conv_cnt) == 1 && PAD(conv_cnt) == 1
        DATA_ROUTER_SELECTION(3) = '1';
    else
        DATA_ROUTER_SELECTION(3) = '0';
    end

    %IS_K01S2P0  : route 7 and 0 with bit position 111,000
    if NKX(conv_cnt) <= 1 && STR(conv_cnt) == 2 && PAD(conv_cnt) == 0
        DATA_ROUTER_SELECTION(1) = '1';
    else
        DATA_ROUTER_SELECTION(1) = '0';
    end

    %IS_K03S1P0  : route 1 with bit position 001
    if NKX(conv_cnt) <= 3 && STR(conv_cnt) == 1 && PAD(conv_cnt) == 0
        DATA_ROUTER_SELECTION(2) = '1';
    else
        DATA_ROUTER_SELECTION(2) = '0';
    end

    %IS_K06S2P1  : route 4 with bit position 100
    if NKX(conv_cnt) <= 6 && STR(conv_cnt) == 2 && PAD(conv_cnt) == 1
        DATA_ROUTER_SELECTION(5) = '1';
    else
        DATA_ROUTER_SELECTION(5) = '0';
    end
        
    DATA_ROUTER_SELECTION(8) = DATA_ROUTER_SELECTION(1);
%DATA_ROUTER_SELECTION bit order is based on string(left to right)
    switch DATA_ROUTER_SELECTION
       case '10000001'
            DATA_ROUTE = 0;
       case '01000000'
            DATA_ROUTE = 1;
       case '00100000'
            DATA_ROUTE = 2;
       case '00010000'
            DATA_ROUTE = 3;
       case '00001000'
            DATA_ROUTE = 4;
       case '00000100'
            DATA_ROUTE = 5;
       case '00000010'
            DATA_ROUTE = 6;
    end

    %fprintf(fid,'%d,', bin2dec(DATA_ROUTER_SELECTION));
    %printf('%s,', DATA_ROUTER_SELECTION);
    %printf('%d,', DATA_ROUTE);
    fprintf(fid,'%d,', DATA_ROUTE);

    fprintf(fid,'%d,', CR_CONV_NIX1BUF(conv_cnt));

    if STR(conv_cnt) == 1
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end

    if STR(conv_cnt) == 2
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end

    % GTH REVIEW - in general change R_<var> to CR_LAYER_<var>.  Verify in Param_LAYER.m. Change index to i (current layer) rather than conv_cnt

    fprintf(fid,'%d,', R_LBUF_WREN(conv_cnt));

    fprintf(fid,'%d,', R_LBUF_RDEN(conv_cnt));


    if Toy(conv_cnt) == NOY(conv_cnt)
        fprintf(fid,'1,');
    else
        fprintf(fid,'0,');
    end

%GTH FOUND
    fprintf(fid,'%d,',R_KXKY_CV_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,',R_KX_CV_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_KY_CV_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_PAD1STR_CV(conv_cnt));
%GTH NOT FOUND Tif_CONV ???
    fprintf(fid,'%d,', Tif(conv_cnt)-1);
%GTH FOUND
    fprintf(fid,'%d,', R_STRIDE_M1(conv_cnt));
%GTH NOT FOUND
    fprintf(fid,'%d,', R_PAD_STR_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_PAD_STR(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_TIYDMA_M1(conv_cnt));
%GTH NOT FOUND
    fprintf(fid,'%d,', TOX_GRP(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_TOX_GRP_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_END_TOYPAD_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', R_END_NOYPAD_M1(conv_cnt));
%GTH FOUND
    fprintf(fid,'%d,', bin2dec(R_PAD_E_CV_POX(conv_cnt,:)));
%GTH FOUND
    fprintf(fid,'%d,', bin2dec(R_PAD_S_CV_POY(conv_cnt,:)));

    



%GTH FOUND   
    R_PAD_CV_bin = '';
    for x = 1:POX
        bin_tmp = dec2bin(R_PAD_E_CV_NKX(conv_cnt,x),WD_PADE);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
%      dividing R_PAD_E_CV_NKX to PAD_E_CV_NKX_HIGH and LOW      %fpga name : R_PAD_E_CV_NKX
    fprintf(fid,'%d,', bin2dec(R_PAD_CV_bin(1:8)));%Higher bits
    fprintf(fid,'%d,', bin2dec(R_PAD_CV_bin( 9:40)));%lower bits


%GTH FOUND
    R_PAD_CV_bin = '';
    for x = 1:POY
        bin_tmp = dec2bin(R_PAD_S_CV_NKY(conv_cnt,x),WD_PADS);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
%      dividing R_PAD_S_CV_NKY to PAD_S_CV_NKY_HIGH and LOW             %fpga name: R_PAD_S_CV_NKY
    fprintf(fid,'%d,', bin2dec(R_PAD_CV_bin(1:8 )));%Higher bits
    fprintf(fid,'%d,', bin2dec(R_PAD_CV_bin(9:40)));%lower bits


%GTH FOUND
fprintf(fid,'%d,', bin2dec(R_PAD_N_CV_POY(conv_cnt,:)));%fpga name: R_PAD_N_CV_POY

    R_PAD_CV_bin = '';
    for x = 1:4
        bin_tmp = dec2bin(R_PAD_N_CV_NKY(conv_cnt,x),WD_PADN);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
fprintf(fid,'%d,', bin2dec(R_PAD_CV_bin));%fpga : R_PAD_N_CV_NKY, new: PAD_N_CV_NKY



%% Pooling

HAS_K3S1_PLMX = 1;
R_IS_K3S1_PLMX = zeros(NUM_PLMX,1);

%GTH NOT FOUND
for L = 1:NUM_PLMX
    if NKX_PLMX(L) == 3 && NKY_PLMX(L) == 3 && STR_PLMX(L) == 1 && PAD_PLMX(L) == 1
        R_IS_K3S1_PLMX(L) = 1;
        HAS_K3S1_PLMX = 1;
    end
end
    fprintf(fid,'%d,',HAS_K3S1_PLMX);

    fprintf(fid,'%d,',R_IS_K3S1_PLMX(plmx_cnt));



%GTH FOUND
R_KXPL_M1 = NKX_PLMX-1;
    fprintf(fid,'%d,',(R_KXPL_M1(plmx_cnt)));

%GTH FOUND
R_KYPL_M1 = NKY_PLMX-1;
    fprintf(fid,'%d,',(R_KYPL_M1(plmx_cnt)));



%GTH NOT FOUND

R_WRDRAM_PLMX = ones(NUM_PLMX,1);
if CR_LAYER_IS_PLMX(end) == 1
    R_WRDRAM_PLMX(end) = 0;
end

    fprintf(fid,'%d,',R_WRDRAM_PLMX(plmx_cnt));

%GTH FOUND - calulation already exists in Param.m
R_PADE_END_PLMX = ceil((NOX_PLMX0-(NOX_PLMX-Tox_PLMX))./POX_PLMX)-1;

    fprintf(fid,'%d,',R_PADE_END_PLMX(plmx_cnt));


%GTH FOUND - calulation already exists in Param.m
R_PADS_END_PLMX = ceil((NOY_PLMX0-(NOY_PLMX-Toy_PLMX))./POY_PLMX)-1;
    fprintf(fid,'%d,',R_PADS_END_PLMX(plmx_cnt));


%GTH FOUND - calulation already exists in Param.m
R_TOXGRP_PLMX_M1 = ceil(Tox_PLMX./POX_PLMX) - 1;
    fprintf(fid,'%d,',R_TOXGRP_PLMX_M1(plmx_cnt));

%GTH FOUND - calulation already exists in Param.m
R_TOYGRP_PLMX_M1 = ceil(Toy_PLMX./POY_PLMX) - 1;

    fprintf(fid,'%d,',R_TOYGRP_PLMX_M1(plmx_cnt));


%GTH FOUND - calulation already exists in Param.m
R_TOFGRP_PLMX_M1 = ceil(Tof_PLMX./POF_PLMX) - 1;
    fprintf(fid,'%d,',R_TOFGRP_PLMX_M1(plmx_cnt));


%GTH FOUND - calulation already exists in Param.m 
R_TIXGRP_PLMX = ceil(Tix_PLMX./POX_PLMX);
    fprintf(fid,'%d,',R_TIXGRP_PLMX(plmx_cnt));


%GTH FOUND
R_TIXGRP1ST_PLMX = ceil(Tix_PLMX./(POX_PLMX.*STR_PLMX));
    fprintf(fid,'%d,',R_TIXGRP1ST_PLMX(plmx_cnt));


%GTH FOUND
R_TIXY1ST_PLMX = ceil((Tix_PLMX.*Tiy_PLMX)./(POX_PLMX.*STR_PLMX)); % STR_PLMX
    fprintf(fid,'%d,',R_TIXY1ST_PLMX(plmx_cnt));



%GTH FOUND
%GTH REVIEW - calulation already exists in Param.m - NIX_PLMX1, NIY_PLMX1, PAD_E_PLMX, PAD_S_PLMX

NIX_PLMX1  = (NOX_PLMX0-1).*STR_PLMX+NKX_PLMX-2*PAD_PLMX;
NIY_PLMX1  = (NOY_PLMX0-1).*STR_PLMX+NKY_PLMX-2*PAD_PLMX;
PAD_E_PLMX = (NIX_PLMX1 - NIX_PLMX0).*(~PAD_PLMX);
PAD_S_PLMX = (NIY_PLMX1 - NIY_PLMX0).*(~PAD_PLMX);
for L = 1:NUM_PLMX
   if PAD_PLMX(L) == 1
       PAD_E_PLMX(L) = 1;
   end
   if PAD_PLMX(L) == 1
       PAD_S_PLMX(L) = 1;
   end
end

%GTH FOUND
%GTH REVIEW - calculation already exists
LOC_PAD_E_PLMX_idx = NOX_PLMX0-floor(NOX_PLMX0./POX_PLMX).*POX_PLMX;
%LOC_PAD_E_PLMX_idx = NOX_PLMX0+PAD_PLMX-floor((NOX_PLMX0+PAD_PLMX)./POX_PLMX).*POX_PLMX;
LOC_PAD_E_PLMX_bin = repmat('0',NUM_PLMX,POX_PLMX);
for L = 1:NUM_PLMX
    if PAD_E_PLMX(L) == 1
        if LOC_PAD_E_PLMX_idx(L) == 0
            LOC_PAD_E_PLMX_idx(L) = POX_PLMX(L);
        end
        LOC_PAD_E_PLMX_bin(L,LOC_PAD_E_PLMX_idx(L)) = '1';
    end
    LOC_PAD_E_PLMX_bin(L,:) = fliplr(LOC_PAD_E_PLMX_bin(L,:));
end
    %str  = sprintf('%s',LOC_PAD_E_PLMX_bin(plmx_cnt,:));
    fprintf(fid,'%d,',bin2dec(LOC_PAD_E_PLMX_bin(plmx_cnt,:)));
    %fprintf(fid,'%d,',bin2dec(str));


    fprintf(fid,'%d,',PAD_S_PLMX(plmx_cnt));


%% fully_connected

% GTH Review - This logic is now commented out of Param.m.
% GTH NOT FOUND
if (NUM_FCON>0)

R_TILE_MAC_FC_M1 = zeros(NUM_FCON,1);
for L = 1:NUM_FCON
    if NIF_FCON(L) <= FCWT_WORDS
        R_TILE_MAC_FC_M1(L) = NUM_TILE_DMA_pFCON(L)-1;
    else
        R_TILE_MAC_FC_M1(L) = ceil(NOF_FCON(L)/POF_FCON)-1;
    end
end

    fprintf(fid,'%d,', R_TILE_MAC_FC_M1(fc_cnt));

% GTH FOUND
R_NIF_FC_M1 = NIF_FCON - 1;

    fprintf(fid,'%d,', R_NIF_FC_M1(fc_cnt));

% GTH NOT FOUND
R_NOF_FC_M1 = NOF_FCON - 1;

    fprintf(fid,'%d,', R_NOF_FC_M1(fc_cnt));


% GTH NOT FOUND
R_NIFGRP_FC_M1 = ceil(FCWT_WORDS./NIF_FCON) - 1;

    fprintf(fid,'%d,',R_NIFGRP_FC_M1(fc_cnt));


% GTH NOT FOUND
R_NIFGRP_POF_FC = POF_FCON*ceil(FCWT_WORDS./NIF_FCON);

    fprintf(fid,'%d,',R_NIFGRP_POF_FC(fc_cnt));


else
    
%     fprintf(fid,'reg R_TILE_MAC_FC_M1 [1:0];\n');
%     fprintf(fid,'reg R_NIF_FC_M1 [1:0];\n');
%     fprintf(fid,'reg R_NOF_FC_M1 [1:0];\n');
%     fprintf(fid,'reg R_NIFGRP_FC_M1 [1:0];\n');
%     fprintf(fid,'reg R_NIFGRP_POF_FC [1:0];\n\n\n');
    
end


%% EltWise
% GTH Review - This logic is now removed from Param.m
% GTH NOT FOUND
R_TOFGRPM1_TOXTOY14POX = (ceil(Tof./POF_EW)-1).*ceil(Tox.*Toy/(4*POX)); % used for Eltwise
    fprintf(fid,'%d,', R_TOFGRPM1_TOXTOY14POX(conv_cnt));

% GTH NOT FOUND
R_TOXTOY14POX_M1 = ceil(Tox.*Toy/(4*POX))-1; % used for Eltwise
    fprintf(fid,'%d,', R_TOXTOY14POX_M1(conv_cnt));


% GTH NOT FOUND
R_TOFTOXTOY132POX_M1 = ceil(Tof/POF_EW).*ceil(Tox.*Toy/(4*POX))-1; % used for Eltwise

    fprintf(fid,'%d,', R_TOFTOXTOY132POX_M1(conv_cnt));

%% end of one DMA dpt transaction

% DRAM RDpx for Scatter
Bytes_RDpx_pDPT_CV1 = NKX(1)*NKY(1)*ceil(NIF(1)/NUM_BK)*Tox(1)*(Toy(1)/POY)*(WD_PX/8)*FCT_DMA;
Toy_8PAD = Toy;
for L=1:NUM_CONV
   if Toy(L) == 7 && Tox(L) < 28
       Toy_8PAD(L) = 8;
   end
end
len_RDpx_pTil_Bytes_CV1EW = Tox.*Toy_8PAD*(WD_PX/8)*FCT_DMA; % = len_RDpx_pTil_Bytes_EW*8/DMA_WIDTH = integers
R_WORDS_RDPX_CV1EW_M1 = len_RDpx_pTil_Bytes_CV1EW*8/DMA_WIDTH - 1;

len_RDpx_pTil_Bytes_PLMX = Tix_PLMX.*Tiy_PLMX*(WD_PX/8)*FCT_DMA;
R_WORDS_RDPX_PLMX_M1 = len_RDpx_pTil_Bytes_PLMX*8/DMA_WIDTH-1;

R_WORDS_RDPX_M1 = zeros(NUM_LAYER,1); % for conv1, Eltwise, Pool_max
cnt_CONV = 1;
cnt_POOL = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        R_WORDS_RDPX_M1(L) = R_WORDS_RDPX_CV1EW_M1(cnt_CONV);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        R_WORDS_RDPX_M1(L) = R_WORDS_RDPX_PLMX_M1(cnt_POOL);
        cnt_POOL = cnt_POOL + 1;
    end
end

% GTH FOUND
    fprintf(fid,'%d,', R_WORDS_RDPX_M1(i));


% DRAM WRpx for Gather
len_WRpx_pTil_Bytes_CV = Tox.*Toy_raw*(WD_PX/8)*FCT_DMA;
len_WRpx_pTil_Bytes_CV_8pad = Tox.*Toy*(WD_PX/8)*FCT_DMA;
R_WORDS_WRPX_CV_M1 = len_WRpx_pTil_Bytes_CV_8pad*8/DMA_WIDTH - 1;

% len_WRpx_pTil_Bytes_PLMX = Tox_PLMX.*Toy_PLMX*(WD_PX/8)*FCT_DMA; % = len_RDpx_pTil_Bytes_EW*8/DMA_WIDTH = integers
% if PL_global_ID(end) == NUM_LAYER % the last key layer not write ouput to DRAM
%     len_WRpx_pTil_Bytes_PLMX(NUM_PLMX) = 0;
% end
% if (sum(mod(len_WRpx_pTil_Bytes_PLMX*8/DMA_WIDTH,1))~=0)
%     fprintf('\n WW3!! len_WRpx_pTil_Bytes_PLMX*8/DMA_WIDTH NOT integer\n\n');
% end
% len_WRpx_pTil_Bytes_PLMX = ceil(len_WRpx_pTil_Bytes_PLMX*8/DMA_WIDTH)*DMA_WIDTH/8;

% if PL_global_ID(end) == NUM_LAYER % the last key layer not write ouput to DRAM
%     len_WRpx_pTil_Bytes_PLMX(NUM_PLMX) = 0;
% end
R_WORDS_WRPX_PLMX_M1 = len_WRpx_pTil_Bytes_PLMX*8/DMA_WIDTH - 1;

R_WORDS_WRPX_M1 = zeros(NUM_LAYER,1);
cnt_CONV = 1;
cnt_POOL = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        R_WORDS_WRPX_M1(L) = R_WORDS_WRPX_CV_M1(cnt_CONV);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        R_WORDS_WRPX_M1(L) = R_WORDS_WRPX_PLMX_M1(cnt_POOL);
        cnt_POOL = cnt_POOL + 1;
    end
end
%R_WORDS_WRPX_M1(NUM_LAYER) = 0;

% GTH FOUND
    fprintf(fid,'%d,', R_WORDS_WRPX_M1(i));



%%
% R_TOFTOXTOY12POXPOY_M1 = ceil(Tof/2).*(Tox/POX).*(Toy/POY)-1;
% fprintf(fid,'// R_TOFTOXTOY12POXPOY_M1 = 1*1*(TOF/2)*(TOX/POX)*(TOY/POY)-1 \n');
% fprintf(fid,'reg [%02d-1:0] R_TOFTOXTOY12POXPOY_M1 [NUM_CONV-1:0];\n',ceil(log2(max(R_TOFTOXTOY12POXPOY_M1)+1)));
% fprintf(fid,'initial begin \n');
% for i = 1:NUM_CONV
%     fprintf(fid,'	R_TOFTOXTOY12POXPOY_M1[%02d] = %04d; \n', i-1, R_TOFTOXTOY12POXPOY_M1(i));
% end
% fprintf(fid,'end \n\n');

% GTH REVIEW - calculation already exists in Param_LAYER
R_NUM_ROWSTR = floor(NIX./(NUM_POX_DMA*POX)).*STR;

% GTH FOUND
    fprintf(fid,'%d,',R_NUM_ROWSTR(conv_cnt));


R_END_ROW = max(floor(NIX./(NUM_POX_DMA*POX))-1,0);

% GTH FOUND
    fprintf(fid,'%d,',R_END_ROW(conv_cnt));


R_END_ROWSTR = max(floor(NIX./(NUM_POX_DMA*POX)).*STR-1,0);

% GTH FOUND
    fprintf(fid,'%d,',R_END_ROWSTR(conv_cnt));



%TBD
%R_TIF1POF_M1 = max(floor(ceil(Tif./Pof) - 1),0);
%R_TIF1POF_M1 = max(floor(ceil(Tif_DMA_CV./Pof) - 1),0);
% fprintf(fid,'// R_TIF1POF_M1 = TIF/POF - 1 \n');
% fprintf(fid,'reg [%02d-1:0] R_TIF1POF_M1 [NUM_CONV-1:0];\n',ceil(log2(max(R_TIF1POF_M1)+1)));
% fprintf(fid,'initial begin \n');
% for i = 1:NUM_CONV
%    fprintf(fid,'%d''d%02d,',ceil(log2(max(R_TIF1POF_M1)+1)),R_TIF1POF_M1(conv_cnt));
% end
% fprintf(fid,'end \n\n');

% GTH Review - calculation already exists in Param_LAYER.m
% GTH FOUND
R_TOY_GRP_M1 = floor(Toy./Py - 1);

    fprintf(fid,'%d,', R_TOY_GRP_M1(conv_cnt));




% GTH FOUND
R_TOF_GRP_M1 = ceil(Tof./Pof) - 1;

    fprintf(fid,'%d,', R_TOF_GRP_M1(conv_cnt));




R_TOF1OUPX_BSCV = ceil(Tof/BUF_OUPX_ALL).*with_Bias;

% GTH FOUND
    fprintf(fid,'%d,',R_TOF1OUPX_BSCV(conv_cnt));


%R_NOF1OUPX_BSCV = ceil(NOF/BUF_OUPX_ALL).*with_Bias;
%R_ACCU_TOF_BSCV = zeros(NUM_CONV,1);
%R_ACCU_TOF_BSCV(1) = 0;
%for i = 2:NUM_CONV
%        R_ACCU_TOF_BSCV(i) = R_ACCU_TOF_BSCV(i-1)+R_NOF1OUPX_BSCV(i-1);
%end

% GTH FOUND
    fprintf(fid,'%d,',R_ACCU_TOF_BSCV(conv_cnt));



%NOYGRP_M1_CV = ceil(NOY./Toy) - 1;
%NOYGRP_M1_PLMX = ceil(NOY_PLMX./Toy_PLMX) - 1;
%R_NOY_GRP_M1 = zeros(NUM_LAYER,1);
%cnt_CV=1; cnt_PLMX=1;
%    if CR_LAYER_IS_CONV(i) == 1
%        R_NOY_GRP_M1(i) =  NOYGRP_M1_CV(cnt_CV);
%        cnt_CV=cnt_CV+1;
%    end
%    if CR_LAYER_IS_PLMX(i) == 1
%        R_NOY_GRP_M1(i) =  NOYGRP_M1_PLMX(cnt_PLMX);
%        cnt_PLMX=cnt_PLMX+1;
%    end

% GTH FOUND
    fprintf(fid,'%d,',R_NOY_GRP_M1(i));


%NOFGRP_M1_CV = ceil(NOF./Tof) - 1;
%NOFGRP_M1_PLMX = ceil(NOF_PLMX./Tof_PLMX) - 1;
%R_NOF_GRP_M1 = zeros(NUM_LAYER,1);
%cnt_CV=1; cnt_PLMX=1;
%    if CR_LAYER_IS_CONV(i) == 1
%        R_NOF_GRP_M1(i) =  NOFGRP_M1_CV(cnt_CV);
%        cnt_CV=cnt_CV+1;
%    end
%    if CR_LAYER_IS_PLMX(i) == 1
%        R_NOF_GRP_M1(i) =  NOFGRP_M1_PLMX(cnt_PLMX);
%        cnt_PLMX=cnt_PLMX+1;
%    end

% GTH FOUND
    fprintf(fid,'%d,', R_NOF_GRP_M1(i));




R_TIX1POX = ceil(NIX./(STR*POX));
% TBD check
%     fprintf(fid,'%d''d%02d,',ceil(log2(max(R_TIX1POX)+1)), R_TIX1POX(conv_cnt));

% GTH REVIEW - calculation already exists in Parm_LAYER.m
R_TIX1POX_STR = R_TIX1POX.*STR;

% GTH FOUND
    fprintf(fid,'%d,',R_TIX1POX_STR(conv_cnt));


R_TIX1POX_TIY1POY = R_TIX1POX .* (ceil(ceil(Tiy_DMA_PAD./STR)/POY).*STR);

% GTH FOUND
    fprintf(fid,'%d,', R_TIX1POX_TIY1POY(conv_cnt));


R_END_ROWPOY = ceil(ceil(Tiy_DMA_PAD./STR)/POY).*(Tif_DMA_CV/NUM_BK)-1;
if HAS_DILATED == 1
R_END_ROWPOY(ID_SSD_FC6) = ceil((Tiy(ID_SSD_FC6)-PAD(ID_SSD_FC6))./((POY+1).*STR(ID_SSD_FC6))).*(Tif_DMA_CV(ID_SSD_FC6)/NUM_BK) - 1;
end

% GTH FOUND
    fprintf(fid,'%d,', R_END_ROWPOY(conv_cnt));



R_TOXGRP_POY = TOX_GRP.*Py;
if HAS_DILATED == 1
R_TOXGRP_POY(ID_SSD_FC6) = (Py(ID_SSD_FC6)*Tox(ID_SSD_FC6))/POX;
end

% GTH FOUND
    fprintf(fid,'%d,',R_TOXGRP_POY(conv_cnt));


R_TOXGRP_TOY = TOX_GRP.*Toy;
if HAS_DILATED == 1
R_TOXGRP_TOY(ID_SSD_FC6) = Tox(ID_SSD_FC6)*Toy(ID_SSD_FC6)/POX;
end

% GTH FOUND
    fprintf(fid,'%d,', R_TOXGRP_TOY(conv_cnt));

R_KX_KY_TIF = NKX .* NKY .* Tif;

% GTH FOUND
    fprintf(fid,'%d,',R_KX_KY_TIF(conv_cnt));

R_KKNIF_DELAY = NKX.*NKY.*NIF+13-1;

% GTH FOUND
    fprintf(fid,'%d,', R_KKNIF_DELAY(conv_cnt));


R_MACOUT_RSBIT_CV = R_MACOUT_RSBIT_CV;

% GTH FOUND
if ~isempty(R_MACOUT_RSBIT_CV)
        fprintf(fid,'%d,', R_MACOUT_RSBIT_CV(conv_cnt));
end

R_MACOUT_RSBIT_FC = R_MACOUT_RSBIT_FC;

% GTH FOUND
if ~isempty(R_MACOUT_RSBIT_FC)
        fprintf(fid,'%d,', R_MACOUT_RSBIT_FC(fc_cnt));
end
%% CONCAT


%% DMA descriptors

% before DDR3_BADDR_IMAGE
DDR3_ENDADDR_WT = hex2dec(DDR3_BADDR_KN_FC6) + (JTAG_WORDS*JTAG_WIDTH*(NUM_JTAG_MIF_FC-1)/8);


%GTH REVIEW - These dma length registers have been replaced with new dma registers.

R_LEN_RDPX_CV = len_RDpx_pTil_Bytes_CV;

    fprintf(fid,'%d,',R_LEN_RDPX_CV(conv_cnt));


R_LEN_RDWT_CV = len_RDwt_pTil_Bytes_CV;

    fprintf(fid,'%d,', R_LEN_RDWT_CV(conv_cnt));

R_LEN_WRPX_CV = len_WRpx_pTil_Bytes_CV;

    fprintf(fid,'%d,', R_LEN_WRPX_CV(conv_cnt));



R_LEN_RDPX_PLMX = len_RDpx_pTil_Bytes_PLMX;

    fprintf(fid,'%d,',R_LEN_RDPX_PLMX(plmx_cnt));


R_LEN_WRPX_PLMX = len_WRpx_pTil_Bytes_PLMX;

    fprintf(fid,'%d,', R_LEN_WRPX_PLMX(plmx_cnt));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          parameters           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(fid, '%d,', 0);
fprintf(fid, '%d,', 1);
fprintf(fid, '%d,', 2);
fprintf(fid, '%d,', 3);
fprintf(fid, '%d,', 4);
fprintf(fid, '%d,', bin2dec('10000'));
fprintf(fid, '%d,', bin2dec('1000000000'));
fprintf(fid, '%d,', 9);
fprintf(fid, '%d,', hex2dec('81000000'));
fprintf(fid, '%d,', hex2dec('80004000'));
fprintf(fid, '%d,', POX);
fprintf(fid, '%d,', POY);
fprintf(fid, '%d,', POF);
fprintf(fid, '%d,', POX_PLMX);
fprintf(fid, '%d,', POY_PLMX);
fprintf(fid, '%d,', POF_PLMX);
fprintf(fid, '%d,', POF_EW);
fprintf(fid, '%d,', NUM_CONV);
fprintf(fid, '%d,', NUM_PLMX);
fprintf(fid, '%d,', NUM_FCON);
fprintf(fid, '%d,', NUM_LAYER);
fprintf(fid, '%d,',STR_PLMX(1));
fprintf(fid, '%d,',NKX_GAVP);
fprintf(fid, '%d,',NKY_GAVP);
fprintf(fid, '%d,', TOXGRP_AV_M1);
fprintf(fid, '%d,', TOY1STR_AV_M1);
fprintf(fid, '%d,', TOFGRP_AV_M1);
fprintf(fid, '%d,', RDAD_ADD_AV);
fprintf(fid, '%d,', INPX1_0FILL);
fprintf(fid, '%d,', END_RDAD_AV);
fprintf(fid, '%d,', WD_DIV);
fprintf(fid, '%d,', DIVISION_AV);
fprintf(fid, '%d,', WORDS_R_GAVP);
fprintf(fid, '%d,', WDAD_R_GAVP);
fprintf(fid, '%d,', WORDS_W_GAVP);
fprintf(fid, '%d,', WDAD_W_GAVP);
fprintf(fid, '%d,',   WORD_BSCV);
fprintf(fid, '%d,', WDAD_BSCV);
fprintf(fid, '%d,',   WDAD_BN_MULADD);
fprintf(fid, '%d,', WORD_BN_MULADD);
fprintf(fid, '%d,',(POX-1)*1+3+1);
fprintf(fid, '%d,',WD_PX);
fprintf(fid, '%d,',WD_WT);
fprintf(fid, '%d,',WD_WT);
fprintf(fid, '%d,',WD_BSCV);
fprintf(fid, '%d,',WD_BSFC);
fprintf(fid, '%d,',WD_BNMUL);
fprintf(fid, '%d,',WD_BNADD);
fprintf(fid, '%d,',6);
fprintf(fid, '%d,',PX_AD);
fprintf(fid, '%d,',POX_2POWER);
fprintf(fid, '%d,',NUM_POX_DMA);
fprintf(fid, '%d,', HAS_INPX_SCALE);
fprintf(fid, '%d,', CVID_INPX_SCALE_1);
fprintf(fid, '%d,', CVID_INPX_SCALE_2);
fprintf(fid, '%d,', HAS_DILATED);
fprintf(fid, '%d,', ID_SSD_FC6-1);
fprintf(fid, '%d,', HAS_GAVP);
fprintf(fid, '%s,',with_POOL_AVE); %TODO         
fprintf(fid, '%d,',   is_PVANET_OB);
fprintf(fid, '%d,', is_PVANET_TL);
fprintf(fid, '%d,',POF_FCON);
fprintf(fid, '%d,',FCWT_WORDS);
%GTH REVIEW - changed to a single array called NIF_FCON and NOF_FCON in Param.m
fprintf(fid, '%d,',NIF_FC1);
fprintf(fid, '%d,',NOF_FC1);
fprintf(fid, '%d,',NIF_FC2);
fprintf(fid, '%d,',NOF_FC2);
fprintf(fid, '%d,',NIF_FC3);
fprintf(fid, '%d,',NOF_FC3);
fprintf(fid, '%d,',NIF_FC4);
fprintf(fid, '%d,',NOF_FC4);
fprintf(fid, '%d,',NIF_FC5);
fprintf(fid, '%d,',NOF_FC5);
fprintf(fid, '%d,',NIF_FC6);
fprintf(fid, '%d,',NOF_FC6);
fprintf(fid, '%d,',NOFAD_FC1);
fprintf(fid, '%d,',NOFAD_FC2);
fprintf(fid, '%d,',NOFAD_FC3);
fprintf(fid, '%d,',NOFAD_FC4);
fprintf(fid, '%d,',NOFAD_FC5);
fprintf(fid, '%d,',NOFAD_FC6);
fprintf(fid, '%d,',NOY_PLMX(end));
fprintf(fid, '%d,',NOF_PLMX(end));
fprintf(fid, '%d,',  NUM_DPT_WR_CV); 
fprintf(fid, '%d,',  NUM_DPT_RD_CV);
fprintf(fid, '%d,',  NUM_DPT_WR_PLMX); 
fprintf(fid, '%d,',  NUM_DPT_RD_PLMX);
fprintf(fid, '%d,',NUM_DPT_RD_EW);
fprintf(fid, '%d,',  WDAD_DPT_WR_CV); 
fprintf(fid, '%d,',  WDAD_DPT_RD_CV);
fprintf(fid, '%d,',  WDAD_DPT_WR_PLMX); 
fprintf(fid, '%d,',  WDAD_DPT_RD_PLMX);
fprintf(fid, '%d,',WDAD_DPT_RD_EW);
fprintf(fid, '%d,',NUM_TILE_CONV);
fprintf(fid, '%d,',NUM_TILE_PLMX);
fprintf(fid, '%d,',TILING_BYTES_FC);
fprintf(fid, '%d,', NUM_TILING_FC );
fprintf(fid, '%d,',NUM_BK);
% GTH FOUND
fprintf(fid, '%d,',BUF_INPX_ALL);
% GTH FOUND
fprintf(fid, '%d,',BUF_OUPX_ALL);
% GTH FOUND
fprintf(fid, '%d,',BUF_INPX_1BK);
% GTH FOUND
fprintf(fid, '%d,',BUF_OUPX_1BK);
%GTH FOUND
fprintf(fid, '%d,',POF1OUPX);
%GTH FOUND
fprintf(fid, '%d,',Bytes_image);
%GTH NOT FOUND
fprintf(fid, '%d,',INWT_WORDS);
%GTH FOUND
fprintf(fid, '%d,',INWT_WDAD);
%GTH FOUND
fprintf(fid, '%d,',INWT_WD);
fprintf(fid, '%d,',WD_RDCVWT);
%GTH FOUND
fprintf(fid, '%d,',INPX_WORDS);
%GTH FOUND
fprintf(fid, '%d,',INPX_WDAD);
%GTH FOUND
fprintf(fid, '%d,',INPX_WIDTH);
%GTH FOUND
fprintf(fid, '%d,',OUPX_WIDTH);
%GTH FOUND
fprintf(fid, '%d,',OUPX_WORD);
%GTH FOUND
fprintf(fid, '%d,',OUPX_WDAD);
%GTH REVIEW - Notes in Param_LAYER.m state new equivalent is DDR3_BDEC_WT_FCON
fprintf(fid, '%d,',hex2dec(DDR3_BADDR_KN_FC6));

fprintf(fid, '%d,',hex2dec('00000000'));
fprintf(fid, '%d,',hex2dec('00000000'));
fprintf(fid, '%d,',hex2dec('00080000'));
fprintf(fid, '%d,',hex2dec('00080000'));
fprintf(fid, '%d,',hex2dec('00100000'));
fprintf(fid, '%d,',hex2dec('00100000'));
fprintf(fid, '%d,',bin2dec(MACOU_RSBIT_str));
%GTH REVIEW - commented out in Param.m
fprintf(fid, '%d,', BUF_PX_AB_BDEC);
%GTH REVIEW - commented out in Param.m
fprintf(fid, '%d,', BUF_WT_AB_BDEC);
%GTH REVIEW - commented out in Param.m
fprintf(fid, '%d,', BUF_OUTFC_AB_BDEC);
%GTH REVIEW - commented out in Param.m
fprintf(fid, '%d,', DDR3_ENDADDR_WT);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',BYTES_BBOXES);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',BYTES_ANCHOR);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',BYTES_SCORES);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',DDR3_BADDR_BBOXES);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',DDR3_BADDR_ANCHOR);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',DDR3_BADDR_SCORES);
%GTH REVIEW - commented out in Param.m
fprintf(fid,'%d,',INPX_PROP_1MAP_DMA_M1);

% GTH REVIEW - Yufei took info from kevin regarding prop and roi layers and added it to Param.m but does not look complete.
fprintf(fid,'%d,',PROPOSAL_CH_HEIGHT);
fprintf(fid,'%d,',PROPOSAL_CH_WIDTH);
fprintf(fid,'%d,',PROPOSAL_NUM_ANCHORS);
fprintf(fid,'%d,',PROPOSAL_NUM_ANCHORS_PER_BUFFER);
fprintf(fid,'%d,',ROI_NUM_DOWNSCALES);
fprintf(fid,'%d,',ROI_NUM_CHANNELS);
fprintf(fid,'%d,',ROI_CH_HEIGHT);
fprintf(fid,'%d,',ROI_CH_WIDTH);
fprintf(fid,'%d,',ROI_NUM_CHANNELS_PER_BUFFER);

fprintf(fid,'%d',1);
    fprintf(fid,'\n');
if  i == NUM_LAYER && n_fc_inc == 0
   i = NUM_LAYER + fc_cnt;
end

end
fclose (fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

