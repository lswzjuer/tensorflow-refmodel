%% run through main_Param_CNN.m

% ALL CR_LAYER_* variables need to be included in csv!!!
% CR_LAYER_* are parameters stored in the Configuration RAM.

  CR_LAYER_LAYER_TYPE                 = zeros(NUM_LAYER,1); %  5-bit
  CR_LAYER_INPX_BUF_MODE              = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_WT_BUF_MODE                = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_OUTPX_BUF_MODE             = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_MACOUT_RSBIT               = zeros(NUM_LAYER,1); %  5-bit
  CR_LAYER_with_ReLU                  = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_with_Bias                  = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_PAD                        = zeros(NUM_LAYER,1); %  3-bit, zero padding
  CR_LAYER_STRIDE                     = zeros(NUM_LAYER,1); %  3-bit, STRIDE
  CR_LAYER_RDADDR_ROTATE_EN           = zeros(NUM_LAYER,1); %  1-bit,
  CR_LAYER_NIX1BUF                    = zeros(NUM_LAYER,1); % 12-bit, INBUF_RDADDR_KEY
  CR_LAYER_LBUF_WREN                  = zeros(NUM_LAYER,1); %  7-bit
  CR_LAYER_LBUF_RDEN                  = zeros(NUM_LAYER,1); %  7-bit
  CR_LAYER_KXKY_CV_M1                 = zeros(NUM_LAYER,1); %  8-bit
  CR_LAYER_NKX_M1                     = zeros(NUM_LAYER,1); %  4-bit, KX_CV_M1
  CR_LAYER_NKY_M1                     = zeros(NUM_LAYER,1); %  4-bit, KY_CV_M1
  CR_LAYER_PAD_DIV_STRIDE             = zeros(NUM_LAYER,1); %  3-bit, PAD_DIV_STRIDE, csr_W_PAD1STR_CV

  CR_LAYER_MAX_PAD_STR                = zeros(NUM_LAYER,1); %  3-bit, MAX_PAD_STR
  CR_LAYER_TIYDMA_M1                  = zeros(NUM_LAYER,1); %  8-bit
  CR_LAYER_TOXDMA_GRP                 = zeros(NUM_LAYER,1); %  7-bit, for DMA
  CR_LAYER_NUM_ROWSTR                 = zeros(NUM_LAYER,1); %  6-bit, W_NUM_ROWSTR
  CR_LAYER_END_ROW                    = zeros(NUM_LAYER,1); %  6-bit, W_END_ROW
  CR_LAYER_END_ROWSTR                 = zeros(NUM_LAYER,1); %  8-bit, W_END_ROWSTR
  CR_LAYER_TOX_GRP_M1                 = zeros(NUM_LAYER,1); %  8-bit, for computation 
  CR_LAYER_TOY_GRP_M1                 = zeros(NUM_LAYER,1); %  8-bit, W_TOY_GRP_M1 
  CR_LAYER_TOF_GRP_M1                 = zeros(NUM_LAYER,1); % 10-bit, W_TOF_GRP_M1 
  CR_LAYER_TOF1OUPX_BSCV              = zeros(NUM_LAYER,1); %  7-bit, W_TOF1OUPX_BSCV
  CR_LAYER_TIX1POX_STR                = zeros(NUM_LAYER,1); %  8-bit, W_TIX1POX_STR
  CR_LAYER_TIX1POX_TIY1POY            = zeros(NUM_LAYER,1); % 11-bit, W_TIX1POX_TIY1POY
  CR_LAYER_END_ROWPOY                 = zeros(NUM_LAYER,1); % 12-bit, W_END_ROWPOY
  CR_LAYER_TOXGRP_POY                 = zeros(NUM_LAYER,1); % 10-bit, W_TOXGRP_POY
  CR_LAYER_TOXGRP_TOY                 = zeros(NUM_LAYER,1); % 11-bit, W_TOXGRP_TOY
  CR_LAYER_KX_KY_TIF                  = zeros(NUM_LAYER,1); % 14-bit, W_KX_KY_TIF
  CR_LAYER_KKNIF_DELAY                = zeros(NUM_LAYER,1); % 15-bit, W_KKNIF_DELAY
  CR_LAYER_TOY_IS_NOY                 = zeros(NUM_LAYER,1); %  1-bit,
  CR_LAYER_ACCU_TOF_BSCV              = zeros(NUM_LAYER,1); % 12-bit, WDAD_BSCV = BUF_BIAS_CONV_WDAD = 12-bit, W_ACCU_TOF_BSCV
  CR_LAYER_NOY_GRP_M1                 = zeros(NUM_LAYER,1); %  7-bit, W_NOY_GRP_M1
  CR_LAYER_NOF_GRP_M1                 = zeros(NUM_LAYER,1); %  6-bit, W_NOF_GRP_M1
  CR_LAYER_WORDS_RDPX_M1              = zeros(NUM_LAYER,1); % 12-bit, W_WORDS_RDPX_M1 for Eltwise, Pool_max
  CR_LAYER_WORDS_WRPX_M1              = zeros(NUM_LAYER,1); % 10-bit, W_WORDS_WRPX_M1

% Used by PLMX 
  CR_LAYER_IS_STR1_PAD1_PLMX          = zeros(NUM_LAYER,1); %  1-bit,
% CR_LAYER_NKX_PLMX_M1                = zeros(NUM_LAYER,1); %  3-bit, merged with NKX_M1
% CR_LAYER_NKY_PLMX_M1                = zeros(NUM_LAYER,1); %  3-bit, merged with NKY_M1
  CR_LAYER_PADE_END_PLMX              = zeros(NUM_LAYER,1); %  5-bit
  CR_LAYER_PADS_END_PLMX              = zeros(NUM_LAYER,1); %  8-bit
% CR_LAYER_TOXGRP_PLMX_M1             = zeros(NUM_LAYER,1); %  5-bit
% CR_LAYER_TOYGRP_PLMX_M1             = zeros(NUM_LAYER,1); %  8-bit
% CR_LAYER_TOFGRP_PLMX_M1             = zeros(NUM_LAYER,1); %  6-bit
  CR_LAYER_TIXGRP_PLMX                = zeros(NUM_LAYER,1); %  5-bit
  CR_LAYER_TIXGRP1ST_PLMX             = zeros(NUM_LAYER,1); %  4-bit
  CR_LAYER_TIXY1ST_PLMX               = zeros(NUM_LAYER,1); % 11-bit
  CR_LAYER_LOC_PAD_E_PLMX             = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_LOC_PAD_S_PLMX             = zeros(NUM_LAYER,1); %  1-bit

  
% Used by EWIS/GAPL 
  CR_LAYER_BUFS_PXIN_DEPTHS           = zeros(NUM_LAYER,1); % 13-bit, csr_BUFS_PXIN_DEPTHS_GAP, //e.g. 4 * ceil(7/32) * 7 * ceil(2048/8) = 4*1792 = 7168 (max = 4*2048) (BUffer depth to store 1 tile)
  CR_LAYER_BUFS_PXOU_DEPTHS           = zeros(NUM_LAYER,1); % 13-bit
  
  
% Used by GAPL
  CR_LAYER_DIVISION_GAP               = zeros(NUM_LAYER,1); % 12-bit, csr_DIVISION_GAP
% CR_LAYER_TOFGRP_GAP_M1              = zeros(NUM_LAYER,1); % 10-bit, csr_TOFGRP_GAP_M1 // merged with CR_LAYER_TOF_GRP_M1 
% CR_LAYER_TIX_GAP                    = zeros(NUM_LAYER,1); %  8-bit, merged with CR_LAYER_TIX
% CR_LAYER_TIY_GAP                    = zeros(NUM_LAYER,1); %  8-bit, merged with CR_LAYER_TIY
  CR_LAYER_NUM_DDR_WORDS_GAP          = zeros(NUM_LAYER,1); %  2-bit, csr_NUM_DDR_WORDS_GAP,(Number of DDR words each TIX occupies)
  CR_LAYER_NUM_SKIP_RDADDR_WORDS_GAP  = zeros(NUM_LAYER,1); %  2-bit, csr_NUM_SKIP_RDADDR_WORDS_GAP,         //  (NUM_DDR_WORDS_GAP*(NUM_POX_DMA*POX) - TIX_GAP)/BUF_INPX_1BK  e.g. (1*32 - 7)/8= 3 (range 0-3)                                         
  CR_LAYER_NUM_VALID_RDADDR_WORDS_GAP = zeros(NUM_LAYER,1); %  2-bit, csr_NUM_VALID_RDADDR_WORDS_GAP, //  NUM_POX_DMA - csr_NUM_SKIP_RDADDR_WORDS_GAP e.g. 4-3 =1 (range 0-3)       
% CR_LAYER_BUFS_PXIN_DEPTHS_GAP       = zeros(NUM_LAYER,1); % 13-bit, csr_BUFS_PXIN_DEPTHS_GAP, //e.g. 4 * ceil(7/32) * 7 * ceil(2048/8) = 4*1792 = 7168 (max = 4*2048) (BUffer depth to store 1 tile)
  
  
% Used by PROP
  CR_LAYER_ch_width                   = zeros(NUM_LAYER,1); %  8-bit, Input Feature Map Width
  CR_LAYER_ch_height                  = zeros(NUM_LAYER,1); %  8-bit, Input Feature Map Height
  CR_LAYER_min_w                      = zeros(NUM_LAYER,1); % 16-bit, bit_frac = 4; Min Box Width  (pre-NMS), Keep proposal if the box's size is ? the minimum width
  CR_LAYER_min_h                      = zeros(NUM_LAYER,1); % 16-bit, bit_frac = 4; Min Box Width  (pre-NMS), Keep proposal if the box's size is ? the minimum height
  CR_LAYER_img_w                      = zeros(NUM_LAYER,1); % 11-bit, Per input to first conv layer; used for clipping after anchor shifting
  CR_LAYER_img_h                      = zeros(NUM_LAYER,1); % 11-bit, Per input to first conv layer; used for clipping after anchor shifting
  CR_LAYER_num_anchors                = zeros(NUM_LAYER,1); %  7-bit, Number of Base Anchors
  CR_LAYER_numPostNMS                 = zeros(NUM_LAYER,1); %  9-bit
  CR_LAYER_nms_threshold              = zeros(NUM_LAYER,1); %  8-bit
  CR_LAYER_num_anchors_per_buffer     = zeros(NUM_LAYER,1); %  5-bit
  CR_LAYER_num_stddev                 = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_div_by_n_correction        = zeros(NUM_LAYER,1); %  8-bit
  CR_LAYER_enable_variance_by_apprx   = zeros(NUM_LAYER,1); %  1-bit
  CR_LAYER_ANCHOR_CTR_X               = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_ANCHOR_CTR_Y               = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_ANCHOR_W                   = zeros(NUM_LAYER,64); % 16-bit
  CR_LAYER_ANCHOR_H                   = zeros(NUM_LAYER,64); % 16-bit
  
% Used by ROIP
  CR_LAYER_SPATIAL_SCALE              = zeros(NUM_LAYER,1);
  CR_LAYER_POOL_SIZE                  = zeros(NUM_LAYER,1);
  CR_LAYER_NUM_CHANNELS               = zeros(NUM_LAYER,1);
  CR_LAYER_NUM_CHANNELS_PER_BUFFER    = zeros(NUM_LAYER,1);
  CR_LAYER_NUM_FRACTIONAL_BITS        = zeros(NUM_LAYER,1);
  CR_LAYER_GRID_DIVISION_EQUIVALENT   = zeros(NUM_LAYER,1);
  CR_LAYER_CH_HEIGHT_ROIPL            = zeros(NUM_LAYER,1);
  CR_LAYER_CH_WIDTH_ROIPL             = zeros(NUM_LAYER,1);

% Used by FCON
  CR_LAYER_NUM_ADDRS_1BOX_M1          = zeros(NUM_LAYER,1); % 11-bit, csr_NUM_ADDRS_1BOX_M1, # of INPX2 addresses of one box
  CR_LAYER_NIX_NUM_ADDRS              = zeros(NUM_LAYER,1); %  2-bit, csr_NIX_NUM_ADDRS, # of INPX2 addresses of one input row
  CR_LAYER_NUM_VALID_PX_M1            = zeros(NUM_LAYER,1); %  4-bit, csr_NUM_VALID_PX_M1
  CR_LAYER_NUM_BOX_1BUF_M1            = zeros(NUM_LAYER,1); %  5-bti, csr_NUM_BOX_1BUF_M1, # of boxes stored in one input buffer
  CR_LAYER_NUM_NIX_INPX2              = zeros(NUM_LAYER,1); %  2-bit, csr_NUM_NIX_INPX2, # of feature rows (NOX_CONV) in one inpx2 address
  CR_LAYER_NIF_FCON_M1                = zeros(NUM_LAYER,1); % 15-bit, csr_NIF_FCON_M1
  CR_LAYER_NOF1POF_FCON_M1            = zeros(NUM_LAYER,1); %  8-bit, csr_NOF1POF_FCON_M1
  CR_LAYER_ROI_TILES_FCON_M1          = zeros(NUM_LAYER,1); %  5-bit, csr_ROI_TILES_FCON_M1

% Used by CONV, DECV
  CR_LAYER_END_TOYPAD_M1              = zeros(NUM_LAYER,1); %  5-bit,
  CR_LAYER_END_NOYPAD_M1              = zeros(NUM_LAYER,1); %  7-bit, W_END_NOYPAD_M1, TODO check
  CR_LAYER_PAD_E_CV_POX               = zeros(NUM_LAYER,1); %  8-bit = POX bits
  CR_LAYER_PAD_S_CV_POY               = zeros(NUM_LAYER,1); %  8-bit = POY bits
  CR_LAYER_PAD_E_CV_NKX               = zeros(NUM_LAYER,1); % 32-bit = POX*WD_KXKY = 8*4 = 32-bit
  CR_LAYER_PAD_S_CV_NKY               = zeros(NUM_LAYER,1); % 32-bit = POY*WD_KXKY = 8*4 = 32-bit
  CR_LAYER_PAD_N_CV_POY               = zeros(NUM_LAYER,1); %  8-bit = POY bits
  CR_LAYER_PAD_N_CV_NKY               = zeros(NUM_LAYER,1); % 32-bit = WD_KXKY*MAX_PAD1STR = 4*8 = 32-bit, W_PAD_N_CV_NKY need to change bit width

  % MAX number of input layers of one layer is 16 
  % MAX number of layers in one CNN is 512 -> 9-bit = LAYER_ID_WIDTH
  CR_LAYER_inpx_layer_id_1            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH,
  CR_LAYER_inpx_layer_id_2            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH,
  CR_LAYER_inpx_layer_id_3            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH,
  CR_LAYER_inpx_layer_id_4            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH,
  CR_LAYER_inpx_layer_id_5            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH,
  CR_LAYER_inpx_layer_id_6            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_7            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_8            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_9            = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_10           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_11           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_12           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_13           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_14           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_15           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_layer_id_16           = zeros(NUM_LAYER,1); % 10-bit = LAYER_ID_WIDTH, 
  CR_LAYER_inpx_num_layer             = CR_LAYER_inpx_num_layer; % 5-bit, csr_inpx_num_layer
  
% Used by dma_ctrl
  CR_LAYER_inpx_cmd_size              = zeros(NUM_LAYER,1); % 32-bit = CMD_SIZE_WIDTH
  CR_LAYER_inpx_addr_adjust           = zeros(NUM_LAYER,1); % 16-bit = ADDR_WIDTH
  CR_LAYER_inwt_addr                  = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  CR_LAYER_inwt_cmd_size              = zeros(NUM_LAYER,1); % 32-bit = CMD_SIZE_WIDTH
  CR_LAYER_outpx_addr                 = CR_LAYER_outpx_addr;% 32-bit = ADDR_WIDTH, defined in DMA_dpt.m
  CR_LAYER_outpx_cmd_size             = zeros(NUM_LAYER,1); % 32-bit = CMD_SIZE_WIDTH
  CR_LAYER_TIX                        = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_TIY                        = zeros(NUM_LAYER,1); % 16-bit
  CR_LAYER_TIF                        = zeros(NUM_LAYER,1); % 14-bit!!!
  CR_LAYER_TOF                        = zeros(NUM_LAYER,1); % 11-bit  
  CR_LAYER_num_tif                    = ones (NUM_LAYER,1); % 12-bit = TILE_CNT_WIDTH
  CR_LAYER_num_toy                    = ones (NUM_LAYER,1); % 12-bit = TILE_CNT_WIDTH
  CR_LAYER_num_tof                    = ones (NUM_LAYER,1); % 12-bit = TILE_CNT_WIDTH
  CR_LAYER_num_tbx                    = ones (NUM_LAYER,1); % 12-bit = TILE_CNT_WIDTH
  CR_LAYER_offset_tiy                 = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  CR_LAYER_offset_wt_tof              = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  CR_LAYER_offset_toy                 = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  CR_LAYER_offset_tof                 = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  CR_LAYER_offset_of                  = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH

% Used by dma_ctrl
  lut_inpx_addr                       = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  lut_offset_if                       = zeros(NUM_LAYER,1); % 32-bit = ADDR_WIDTH
  lut_nif                             = zeros(NUM_LAYER,1); % 16-bit

  % TODO TIX TIY TOX TOY TIF TOF 
  

%%

for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 0;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 2;
    end
    if CR_LAYER_IS_PROP(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 3;
        CR_LAYER_INPX_BUF_MODE(L) = 1;
    end
    if CR_LAYER_IS_GAPL(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 4;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 5;
        CR_LAYER_INPX_BUF_MODE(L) = 1;
        CR_LAYER_OUTPX_BUF_MODE(L) = 1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 6;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 7;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_LAYER_TYPE(L) = 8;
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV and DECV layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_NEAR = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_STRIDE      (L) = STR_CONV(cnt_CONV);
        CR_LAYER_PAD         (L) = PAD_CONV(cnt_CONV);
        CR_LAYER_MACOUT_RSBIT(L) = CR_CONV_MACOUT_RSBIT_CV(cnt_CONV);
        CR_LAYER_with_ReLU   (L) = CR_CONV_with_ReLU(cnt_CONV);
        CR_LAYER_with_Bias   (L) = CR_CONV_with_Bias(cnt_CONV);
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_STRIDE      (L) = STR_D(cnt_DECV);
        CR_LAYER_PAD         (L) = PAD_DECV(cnt_DECV);
        CR_LAYER_MACOUT_RSBIT(L) = CR_DECV_MACOUT_RSBIT_CV(cnt_DECV);
        cnt_DECV = cnt_DECV+1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_STRIDE      (L) = STR_PLMX(cnt_PLMX);
        CR_LAYER_PAD         (L) = PAD_PLMX(cnt_PLMX);
        cnt_PLMX = cnt_PLMX+1;
    end
    if CR_LAYER_IS_PROP(L) == 1
        CR_LAYER_STRIDE      (L) = CR_PROP_STRIDE(cnt_PROP);
        cnt_PROP = cnt_PROP+1;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        CR_LAYER_with_ReLU   (L) = CR_EWIS_with_ReLU(cnt_EWIS);
        cnt_EWIS = cnt_EWIS+1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_MACOUT_RSBIT(L) = CR_FCON_MACOUT_RSBIT_FC(cnt_FCON);
        CR_LAYER_with_ReLU   (L) = CR_FCON_with_ReLU(cnt_FCON);
        CR_LAYER_with_Bias   (L) = CR_FCON_with_Bias(cnt_FCON);
        cnt_FCON = cnt_FCON+1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_STRIDE      (L) = STR_NEAR(cnt_NEAR);
        cnt_NEAR = cnt_NEAR+1;
    end
end

cnt_CONV = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        if NIX_CONV0(cnt_CONV) + PAD_CONV(cnt_CONV) + (POX-1)*STR_CONV(cnt_CONV) > NIX_CONV(cnt_CONV)
            CR_LAYER_RDADDR_ROTATE_EN(L) = 1;
        end
        cnt_CONV = cnt_CONV+1;
    end
end

cnt_CONV = 1; cnt_DECV = 1; cnt_NEAR = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1;
R_TIX1POX_CONV = ceil(NIX_CONV  ./(STR_CONV*POX));
R_TIX1POX_DECV = ceil(NIX_W_DECV./(STR_D   *POX));
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_NIX1BUF        (L) = ceil(NIX_CONV(cnt_CONV)/(POX*STR_CONV(cnt_CONV))); % 12-bit
        CR_LAYER_LBUF_WREN      (L) = max(NKX_CONV(cnt_CONV)*(NKY_CONV(cnt_CONV)-STR_CONV(cnt_CONV)),0); % 7-bit
        CR_LAYER_LBUF_RDEN      (L) = NKX_CONV(cnt_CONV)*STR_CONV(cnt_CONV); % 7-bit
        CR_LAYER_KXKY_CV_M1     (L) = NKX_CONV(cnt_CONV)*NKY_CONV(cnt_CONV)-1; % 8-bit
        CR_LAYER_NKX_M1         (L) = NKX_CONV(cnt_CONV)-1; % 4-bit, KX_CV_M1
        CR_LAYER_NKY_M1         (L) = NKY_CONV(cnt_CONV)-1; % 4-bit, KY_CV_M1
        CR_LAYER_PAD_DIV_STRIDE (L) = floor(PAD_CONV(cnt_CONV)/STR_CONV(cnt_CONV)); % 3-bit, North Pad
        CR_LAYER_MAX_PAD_STR    (L) = max(mod(PAD_CONV((cnt_CONV)),STR_CONV((cnt_CONV))),0); % 3-bit % North Pad
        CR_LAYER_TIYDMA_M1      (L) = Tiy_CONV_DMA(cnt_CONV)-1; % 8-bit
        CR_LAYER_TOXDMA_GRP     (L) = Tox_CONV(cnt_CONV)/POX; % 7-bit for DMA
        CR_LAYER_TOX_GRP_M1     (L) = ceil(NOX_CONV0(cnt_CONV)/POX) - 1; % 7-bit for computation
        CR_LAYER_NUM_ROWSTR     (L) = floor(NIX_CONV(cnt_CONV)/(NUM_POX_DMA*POX))*STR_CONV(cnt_CONV); % 6-bit, W_NUM_ROWSTR
        CR_LAYER_END_ROW        (L) = max(floor(NIX_CONV(cnt_CONV)/(NUM_POX_DMA*POX))-1,0); % 6-bit, W_END_ROW
        CR_LAYER_END_ROWSTR     (L) = max(floor(NIX_CONV(cnt_CONV)/(NUM_POX_DMA*POX))*STR_CONV(cnt_CONV)-1,0); % 8-bit, W_END_ROWSTR
        CR_LAYER_TOY_GRP_M1     (L) = ceil(Toy_CONV(cnt_CONV)/POY) - 1; % 5-bit, W_TOY_GRP_M1
        CR_LAYER_TOF_GRP_M1     (L) = ceil(Tof_CONV(cnt_CONV)/POF) - 1; % 5-bit, W_TOF_GRP_M1
        CR_LAYER_TOF1OUPX_BSCV  (L) = ceil(Tof_CONV(cnt_CONV)/BUF_OUPX_ALL)*CR_CONV_with_Bias(cnt_CONV); % 7-bit, W_TOF1OUPX_BSCV
        CR_LAYER_TIX1POX_STR    (L) = R_TIX1POX_CONV(cnt_CONV)*STR_CONV(cnt_CONV); % 8-bit, W_TIX1POX_STR
        CR_LAYER_TIX1POX_TIY1POY(L) = R_TIX1POX_CONV(cnt_CONV)*(ceil(ceil(Tiy_CONV_DMA_PAD(cnt_CONV)/STR_CONV(cnt_CONV))/POY)*STR_CONV(cnt_CONV)); % 11-bit, W_TIX1POX_TIY1POY
        CR_LAYER_END_ROWPOY     (L) = ceil(ceil(Tiy_CONV_DMA_PAD(cnt_CONV)/STR_CONV(cnt_CONV))/POY)*NIF_CONV0(cnt_CONV)-1; % 12-bit W_END_ROWPOY. NOTE: only read NIF_CONV0 from DDR
        CR_LAYER_TOXGRP_POY     (L) = (Tox_CONV(cnt_CONV)/POX)*POY; % 10-bit, W_TOXGRP_POY
        CR_LAYER_TOXGRP_TOY     (L) = (Tox_CONV(cnt_CONV)/POX)*Toy_CONV(cnt_CONV); % 11-bit, W_TOXGRP_TOY
        CR_LAYER_KX_KY_TIF      (L) = NKX_CONV(cnt_CONV)*NKY_CONV(cnt_CONV)*NIF_CONV(cnt_CONV);      % 14-bit, W_KX_KY_TIF
        CR_LAYER_KKNIF_DELAY    (L) = NKX_CONV(cnt_CONV)*NKY_CONV(cnt_CONV)*NIF_CONV(cnt_CONV)+13-1; % 15-bit W_KKNIF_DELAY
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_NIX1BUF        (L) = ceil(NIX_W_DECV(cnt_DECV)/(POX*STR_D(cnt_DECV))); % 12-bit
        CR_LAYER_LBUF_WREN      (L) = max(NKX_DECV(cnt_DECV)*(NKY_DECV(cnt_DECV)-STR_D(cnt_DECV)),0);
        CR_LAYER_LBUF_RDEN      (L) = NKX_DECV(cnt_DECV)*STR_D(cnt_DECV);
        CR_LAYER_KXKY_CV_M1     (L) = NKX_DECV(cnt_DECV)*NKY_DECV(cnt_DECV)-1; % 8-bit
        CR_LAYER_NKX_M1         (L) = NKX_DECV(cnt_DECV)-1; % 4-bit
        CR_LAYER_NKY_M1         (L) = NKY_DECV(cnt_DECV)-1; % 4-bit
        CR_LAYER_PAD_DIV_STRIDE (L) = floor(PAD_D(cnt_DECV)/STR_D(cnt_DECV)); % 3-bit
        CR_LAYER_MAX_PAD_STR    (L) = max(mod(PAD_D(cnt_DECV),STR_D(cnt_DECV)),0); % 3-bit
        CR_LAYER_TIYDMA_M1      (L) = Tiy_W_DECV_DMA(cnt_DECV)-1; % 8-bit  % RSP: CR_DECV_TIYDMA_M1 = Tiy_DECV_DMA-1; % 8-bit
        CR_LAYER_TOXDMA_GRP     (L) = Tox_W_DECV(cnt_DECV)/POX; % 7-bit for DMA  % RSP: CR_DECV_TOXDMA_GRP = Tox_DECV/POX; % 7-bit for DMA
        CR_LAYER_TOX_GRP_M1     (L) = ceil(NOX_DECV0(cnt_DECV)/POX) - 1; % 7-bit for computation % NOTE: RSP: Should not be NOX_W_DECV since we need the exact number i.e. 112 and not a multiple of PX_AD i.e. 128
        CR_LAYER_NUM_ROWSTR     (L) = floor(NIX_W_DECV(cnt_DECV)/(NUM_POX_DMA*POX))*STR_D(cnt_DECV); % 6-bit, W_NUM_ROWSTR
        CR_LAYER_END_ROW        (L) = max(floor(NIX_W_DECV(cnt_DECV)/(NUM_POX_DMA*POX))-1,0); % 6-bit, W_END_ROW
        CR_LAYER_END_ROWSTR     (L) = max(floor(NIX_W_DECV(cnt_DECV)/(NUM_POX_DMA*POX))*STR_D(cnt_DECV)-1,0); % 8-bit, W_END_ROWSTR
        CR_LAYER_TOY_GRP_M1     (L) = ceil(Toy_W_DECV(cnt_DECV)/POY) - 1; % 5-bit, W_TOY_GRP_M1
        CR_LAYER_TOF_GRP_M1     (L) = ceil(Tof_DECV(cnt_DECV)/POF) - 1; % 5-bit, W_TOF_GRP_M1
        CR_LAYER_TOF1OUPX_BSCV  (L) = 0; % RSP: Deconv Updates: No BIAS for Deconv
        CR_LAYER_TIX1POX_STR    (L) = R_TIX1POX_DECV(cnt_DECV)*STR_D(cnt_DECV); % 8-bit, W_TIX1POX_STR
        CR_LAYER_TIX1POX_TIY1POY(L) = R_TIX1POX_DECV(cnt_DECV)*(ceil(ceil(Tiy_DECV_DMA_PAD(cnt_DECV)/STR_D(cnt_DECV))/POY)*STR_D(cnt_DECV)); % 11-bit, W_TIX1POX_TIY1POY
        CR_LAYER_END_ROWPOY     (L) = ceil(ceil(Tiy_W_DECV_DMA_PAD(cnt_DECV)/STR_D(cnt_DECV))/POY)*NIF_DECV0(cnt_DECV)-1; % 12-bit W_END_ROWPOY. NOTE: only read NIF_CONV0 from DDR
        CR_LAYER_TOXGRP_POY     (L) = (Tox_W_DECV(cnt_DECV)/POX)*POY; % 10-bit, W_TOXGRP_POY
        CR_LAYER_TOXGRP_TOY     (L) = (Tox_W_DECV(cnt_DECV)/POX)*Toy_W_DECV(cnt_DECV); % 11-bit, W_TOXGRP_TOY
        CR_LAYER_KX_KY_TIF      (L) = NKX_DECV(cnt_DECV)*NKY_DECV(cnt_DECV)*NIF_DECV(cnt_DECV); % 14-bit, W_KX_KY_TIF
        CR_LAYER_KKNIF_DELAY    (L) = NKX_DECV(cnt_DECV)*NKY_DECV(cnt_DECV)*NIF_DECV(cnt_DECV)+13-1; % 15-bit W_KKNIF_DELAY
        cnt_DECV = cnt_DECV+1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_NKX_M1         (L) = NKX_PLMX(cnt_PLMX)-1; % 4-bit
        CR_LAYER_NKY_M1         (L) = NKY_PLMX(cnt_PLMX)-1; % 4-bit
        CR_LAYER_TOX_GRP_M1     (L) = CR_PLMX_TOXGRP_PLMX_M1(cnt_PLMX);
        CR_LAYER_TOY_GRP_M1     (L) = CR_PLMX_TOYGRP_PLMX_M1(cnt_PLMX);
        CR_LAYER_TOF_GRP_M1     (L) = CR_PLMX_TOFGRP_PLMX_M1(cnt_PLMX);
        cnt_PLMX = cnt_PLMX+1;
    end    
    if CR_LAYER_IS_EWIS(L) == 1
        cnt_EWIS = cnt_EWIS+1;
    end
    if CR_LAYER_IS_GAPL(L) == 1
	    CR_LAYER_TOF_GRP_M1     (L) = CR_GAPL_TOFGRP_GAP_M1(cnt_GAPL);
        cnt_GAPL = cnt_GAPL+1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_TOX_GRP_M1     (L) = ceil(POF_FCON/POX)-1; % used by dmabuf_gather
        cnt_FCON = cnt_FCON+1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_TOF_GRP_M1     (L) = CR_NEAR_TOFGRP_NEAR_M1(cnt_NEAR);
        cnt_NEAR = cnt_NEAR+1;
    end
end  


cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        if Toy_CONV(cnt_CONV) == NOY_CONV(cnt_CONV)
        	CR_LAYER_TOY_IS_NOY(L) = 1;
        end
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        if Toy_W_DECV(cnt_DECV) == NOY_W_DECV(cnt_DECV)
        	CR_LAYER_TOY_IS_NOY(L) = 1;
        end
        cnt_DECV = cnt_DECV+1;
    end
end


    
if NUM_CONV>0    
    R_NOF1OUPX_CONV = ceil(NOF_CONV/BUF_OUPX_ALL).*CR_CONV_with_Bias;
end
if NUM_FCON>0
    R_NOF1OUPX_FCON = ceil(NOF_FCON/BUF_OUPX_ALL).*CR_FCON_with_Bias;
end
CR_CONV_ACCU_TOF_BSCV      = zeros(NUM_CONV,1);
CR_FCON_ACCU_TOF_BSCV      = zeros(NUM_FCON,1);
CR_CONV_ACCU_TOF_BSCV(1) = 0;
CR_FCON_ACCU_TOF_BSCV(1) = 0;
for i = 2:NUM_CONV
    CR_CONV_ACCU_TOF_BSCV(i) = CR_CONV_ACCU_TOF_BSCV(i-1)+R_NOF1OUPX_CONV(i-1);
end 
for i = 2:NUM_FCON
    CR_FCON_ACCU_TOF_BSCV(i) = CR_FCON_ACCU_TOF_BSCV(i-1)+R_NOF1OUPX_FCON(i-1);
end 
    
cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
	    CR_LAYER_ACCU_TOF_BSCV(L) = CR_CONV_ACCU_TOF_BSCV(cnt_CONV);
	    cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_FCON(L) == 1
	    CR_LAYER_ACCU_TOF_BSCV(L) = CR_FCON_ACCU_TOF_BSCV(cnt_FCON);
	    cnt_FCON = cnt_FCON+1;
    end
end    
   



NOYGRP_M1_CONV = ceil(NOY_CONV./Toy_CONV) - 1;
NOYGRP_M1_PLMX = ceil(NOY_PLMX./Toy_PLMX) - 1;
NOYGRP_M1_DECV = ceil(NOY_DECV./Toy_W_DECV) - 1; % RSP : Deconv Updates
cnt_CONV=1; cnt_PLMX=1; cnt_DECV=1;
for L=1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_NOY_GRP_M1(L) =  max(NOYGRP_M1_CONV(cnt_CONV),0);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_NOY_GRP_M1(L) =  max(NOYGRP_M1_PLMX(cnt_PLMX),0);
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_NOY_GRP_M1(L) =  max(NOYGRP_M1_CONV(cnt_DECV),0);
        cnt_DECV = cnt_DECV + 1;
    end
end

NOFGRP_M1_CONV = ceil(NOF_CONV./Tof_CONV) - 1;
NOFGRP_M1_PLMX = ceil(NOF_PLMX./Tof_PLMX) - 1;
NOFGRP_M1_DECV = ceil(NOF_DECV./Tof_DECV) - 1; % RSP: Deconv Updates
cnt_CONV=1; cnt_PLMX=1; cnt_DECV=1;
for L=1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_NOF_GRP_M1(L) =  max(NOFGRP_M1_CONV(cnt_CONV),0);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_NOF_GRP_M1(L) =  max(NOFGRP_M1_PLMX(cnt_PLMX),0);
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_NOF_GRP_M1(L) =  max(NOFGRP_M1_DECV(cnt_DECV),0);
        cnt_DECV = cnt_DECV + 1;
    end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pooling Max (PLMX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_IS_STR1_PAD1_PLMX(L) =  CR_PLMX_IS_STR1_PAD1_PLMX(cnt_PLMX);
%       CR_LAYER_NKX_PLMX_M1      (L) =  CR_PLMX_NKX_PLMX_M1      (cnt_PLMX);
%       CR_LAYER_NKY_PLMX_M1      (L) =  CR_PLMX_NKY_PLMX_M1      (cnt_PLMX);
        CR_LAYER_PADE_END_PLMX    (L) =  CR_PLMX_PADE_END_PLMX    (cnt_PLMX);
        CR_LAYER_PADS_END_PLMX    (L) =  CR_PLMX_PADS_END_PLMX    (cnt_PLMX);
%       CR_LAYER_TOXGRP_PLMX_M1   (L) =  CR_PLMX_TOXGRP_PLMX_M1   (cnt_PLMX);
%       CR_LAYER_TOYGRP_PLMX_M1   (L) =  CR_PLMX_TOYGRP_PLMX_M1   (cnt_PLMX);
%       CR_LAYER_TOFGRP_PLMX_M1   (L) =  CR_PLMX_TOFGRP_PLMX_M1   (cnt_PLMX);
        CR_LAYER_TIXGRP_PLMX      (L) =  CR_PLMX_TIXGRP_PLMX      (cnt_PLMX);
        CR_LAYER_TIXGRP1ST_PLMX   (L) =  CR_PLMX_TIXGRP1ST_PLMX   (cnt_PLMX);
        CR_LAYER_TIXY1ST_PLMX     (L) =  CR_PLMX_TIXY1ST_PLMX     (cnt_PLMX);
        CR_LAYER_LOC_PAD_E_PLMX   (L) =  CR_PLMX_LOC_PAD_E_PLMX   (cnt_PLMX);
        CR_LAYER_LOC_PAD_S_PLMX   (L) =  CR_PLMX_LOC_PAD_S_PLMX   (cnt_PLMX);
        cnt_PLMX = cnt_PLMX + 1;
    end
end



%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Eltwise (EWIS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
%for L=1:NUM_LAYER
%    if CR_LAYER_IS_EWIS(L) == 1
%        CR_LAYER_BUFS_PXIN_DEPTHS_EW (L) =  CR_EWIS_BUFS_PXIN_DEPTHS_EW (cnt_EWIS);
%        CR_LAYER_BUFS_PXOU_DEPTHS_EW (L) =  CR_EWIS_BUFS_PXOU_DEPTHS_EW (cnt_EWIS);
%        cnt_EWIS = cnt_EWIS + 1;
%    end
%end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global Average Pooling (GAPL) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_GAPL(L) == 1
        CR_LAYER_DIVISION_GAP              (L) = CR_GAPL_DIVISION_GAP               (cnt_GAPL);
%       CR_LAYER_TOFGRP_GAP_M1             (L) = CR_GAPL_TOFGRP_GAP_M1              (cnt_GAPL);
%       CR_LAYER_TIX_GAP                   (L) = CR_GAPL_TIX_GAP                    (cnt_GAPL);
%       CR_LAYER_TIY_GAP                   (L) = CR_GAPL_TIY_GAP                    (cnt_GAPL);
        CR_LAYER_NUM_DDR_WORDS_GAP         (L) = CR_GAPL_NUM_DDR_WORDS_GAP          (cnt_GAPL);
        CR_LAYER_NUM_SKIP_RDADDR_WORDS_GAP (L) = CR_GAPL_NUM_SKIP_RDADDR_WORDS_GAP  (cnt_GAPL); 
        CR_LAYER_NUM_VALID_RDADDR_WORDS_GAP(L) = CR_GAPL_NUM_VALID_RDADDR_WORDS_GAP (cnt_GAPL);
        CR_LAYER_BUFS_PXIN_DEPTHS          (L) = CR_GAPL_BUFS_PXIN_DEPTHS_GAP       (cnt_GAPL);
        cnt_GAPL = cnt_GAPL + 1;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        CR_LAYER_BUFS_PXIN_DEPTHS          (L) = CR_EWIS_BUFS_PXIN_DEPTHS_EW        (cnt_EWIS);
        CR_LAYER_BUFS_PXOU_DEPTHS          (L) = CR_EWIS_BUFS_PXOU_DEPTHS_EW        (cnt_EWIS);
        cnt_EWIS = cnt_EWIS + 1;
    end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Proposal (PROP) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_PROP(L) == 1
        CR_LAYER_ch_width                 (L) = CR_PROP_ch_width                 (cnt_PROP);
        CR_LAYER_ch_height                (L) = CR_PROP_ch_height                (cnt_PROP);
        CR_LAYER_min_w                    (L) = CR_PROP_min_w                    (cnt_PROP);
        CR_LAYER_min_h                    (L) = CR_PROP_min_h                    (cnt_PROP);
        CR_LAYER_img_w                    (L) = CR_PROP_img_w                    (cnt_PROP);
        CR_LAYER_img_h                    (L) = CR_PROP_img_h                    (cnt_PROP);
        CR_LAYER_num_anchors              (L) = CR_PROP_num_anchors              (cnt_PROP); 
        CR_LAYER_numPostNMS               (L) = CR_PROP_numPostNMS               (cnt_PROP); 
        CR_LAYER_nms_threshold            (L) = CR_PROP_nms_threshold            (cnt_PROP);
        CR_LAYER_num_anchors_per_buffer   (L) = CR_PROP_num_anchors_per_buffer   (cnt_PROP);
        CR_LAYER_num_stddev               (L) = CR_PROP_num_stddev               (cnt_PROP); 
        CR_LAYER_div_by_n_correction      (L) = CR_PROP_div_by_n_correction      (cnt_PROP); 
        CR_LAYER_enable_variance_by_apprx (L) = CR_PROP_enable_variance_by_apprx (cnt_PROP); 
        cnt_PROP = cnt_PROP + 1;
    end
end

if NUM_PROP >0
    for L=1:NUM_LAYER % global register, NOT layer based
        CR_LAYER_ANCHOR_CTR_X             (L) = CR_PROP_ANCHOR_CTR_X             (1);
        CR_LAYER_ANCHOR_CTR_Y             (L) = CR_PROP_ANCHOR_CTR_Y             (1);
    end
    for L=1:NUM_LAYER % global register, NOT layer based
        for ii = 1:64
            CR_LAYER_ANCHOR_W             (L,ii) = CR_PROP_ANCHOR_W              (1,ii);
            CR_LAYER_ANCHOR_H             (L,ii) = CR_PROP_ANCHOR_H              (1,ii);
        end
    end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROIPooling (ROIP) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_ROIP(L) == 1
        CR_LAYER_SPATIAL_SCALE             (L) = CR_ROIP_SPATIAL_SCALE            (cnt_ROIP);
        CR_LAYER_POOL_SIZE                 (L) = CR_ROIP_POOL_SIZE                (cnt_ROIP);
        CR_LAYER_NUM_CHANNELS              (L) = CR_ROIP_NUM_CHANNELS             (cnt_ROIP);
        CR_LAYER_NUM_CHANNELS_PER_BUFFER   (L) = CR_ROIP_NUM_CHANNELS_PER_BUFFER  (cnt_ROIP);
        CR_LAYER_NUM_FRACTIONAL_BITS       (L) = CR_ROIP_NUM_FRACTIONAL_BITS      (cnt_ROIP);
        CR_LAYER_GRID_DIVISION_EQUIVALENT  (L) = CR_ROIP_GRID_DIVISION_EQUIVALENT (cnt_ROIP);
        CR_LAYER_CH_HEIGHT_ROIPL           (L) = CR_ROIP_CH_HEIGHT_ROIPL          (cnt_ROIP);
        CR_LAYER_CH_WIDTH_ROIPL            (L) = CR_ROIP_CH_WIDTH_ROIPL           (cnt_ROIP);
        cnt_ROIP = cnt_ROIP + 1;
    end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fully_connected (FCON) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_NUM_ADDRS_1BOX_M1(L) = CR_FCON_NUM_ADDRS_1BOX_M1(cnt_FCON);
        CR_LAYER_NIX_NUM_ADDRS    (L) = CR_FCON_NIX_NUM_ADDRS    (cnt_FCON);
        CR_LAYER_NUM_VALID_PX_M1  (L) = CR_FCON_NUM_VALID_PX_M1  (cnt_FCON);
        CR_LAYER_NUM_BOX_1BUF_M1  (L) = CR_FCON_NUM_BOX_1BUF_M1  (cnt_FCON);
        CR_LAYER_NUM_NIX_INPX2    (L) = CR_FCON_NUM_NIX_INPX2    (cnt_FCON);
        CR_LAYER_NIF_FCON_M1      (L) = CR_FCON_NIF_FCON_M1      (cnt_FCON);
        CR_LAYER_NOF1POF_FCON_M1  (L) = CR_FCON_NOF1POF_FCON_M1  (cnt_FCON);
        CR_LAYER_ROI_TILES_FCON_M1(L) = CR_FCON_ROI_TILES_FCON_M1(cnt_FCON);
        cnt_FCON = cnt_FCON + 1;
    end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of one DMA dpt transaction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%R_WORDS_RDPX_EWIS_M1 = (NIX_EWIS/PX_AD).*Tiy_EWIS-1;
R_WORDS_RDPX_EWIS_M1 = CR_EWIS_inpx_cmd_size*8/DMA_WIDTH-1;
R_WORDS_RDPX_PLMX_M1 = CR_PLMX_inpx_cmd_size*8/DMA_WIDTH-1;
R_WORDS_RDPX_ROIP_M1 = CR_ROIP_inpx_cmd_size*8/DMA_WIDTH-1;
R_WORDS_RDPX_NEAR_M1 = CR_NEAR_inpx_cmd_size*8/DMA_WIDTH-1;
cnt_EWIS = 1;cnt_PLMX = 1;cnt_ROIP=1;cnt_FCON = 1;cnt_NEAR = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_EWIS(L) == 1
        CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_EWIS_M1(cnt_EWIS);
        cnt_EWIS = cnt_EWIS + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_PLMX_M1(cnt_PLMX);
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_NEAR_M1(cnt_NEAR);
        cnt_NEAR = cnt_NEAR + 1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_ROIP_M1(cnt_ROIP);
        cnt_ROIP = cnt_ROIP + 1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_FCON_M1(cnt_FCON);
        cnt_FCON = cnt_FCON + 1;
    end
end

R_WORDS_WRPX_CONV_M1 = CR_CONV_offset_toy*8/DMA_WIDTH - 1;
R_WORDS_WRPX_PLMX_M1 = CR_PLMX_offset_toy*8/DMA_WIDTH - 1;
R_WORDS_WRPX_NEAR_M1 = CR_NEAR_offset_toy*8/DMA_WIDTH - 1;
R_WORDS_WRPX_DECV_M1 = CR_DECV_offset_toy*8/DMA_WIDTH - 1; % RSP: Deconv Updates
R_WORDS_WRPX_ROIP_M1 = CR_ROIP_outpx_cmd_size*8/DMA_WIDTH - 1;
R_WORDS_WRPX_FCON_M1 = CR_FCON_outpx_cmd_size*8/DMA_WIDTH - 1;
cnt_CONV = 1;cnt_PLMX = 1;cnt_DECV = 1;cnt_ROIP=1;cnt_FCON = 1;cnt_NEAR = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_CONV_M1(cnt_CONV);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_PLMX_M1(cnt_PLMX);
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_NEAR_M1(cnt_NEAR);
        cnt_NEAR = cnt_NEAR + 1;
    end
    if CR_LAYER_IS_DECV(L) == 1 % RSP: Deconv Updates
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_DECV_M1(cnt_DECV);
        cnt_DECV = cnt_DECV + 1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_ROIP_M1(cnt_ROIP);
        cnt_ROIP = cnt_ROIP + 1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_FCON_M1(cnt_FCON);
        cnt_FCON = cnt_FCON + 1;
    end
end



%%  %%%%%%%%%%%%% CONV and DECV zero padding %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1;cnt_PLMX = 1;cnt_DECV = 1;
for L=1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_END_TOYPAD_M1(L) = CR_CONV_END_TOYPAD_M1(cnt_CONV);
        CR_LAYER_END_NOYPAD_M1(L) = CR_CONV_END_NOYPAD_M1(cnt_CONV);
        CR_LAYER_PAD_E_CV_POX (L) = CR_CONV_PAD_E_CV_POX (cnt_CONV);
        CR_LAYER_PAD_S_CV_POY (L) = CR_CONV_PAD_S_CV_POY (cnt_CONV);
        CR_LAYER_PAD_E_CV_NKX (L) = CR_CONV_PAD_E_CV_NKX (cnt_CONV);
        CR_LAYER_PAD_S_CV_NKY (L) = CR_CONV_PAD_S_CV_NKY (cnt_CONV);
        CR_LAYER_PAD_N_CV_POY (L) = CR_CONV_PAD_N_CV_POY (cnt_CONV);
        CR_LAYER_PAD_N_CV_NKY (L) = CR_CONV_PAD_N_CV_NKY (cnt_CONV);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_END_TOYPAD_M1(L) = CR_DECV_END_TOYPAD_M1(cnt_DECV);
        CR_LAYER_END_NOYPAD_M1(L) = CR_DECV_END_NOYPAD_M1(cnt_DECV);
        CR_LAYER_PAD_E_CV_POX (L) = CR_DECV_PAD_E_CV_POX (cnt_DECV);
        CR_LAYER_PAD_S_CV_POY (L) = CR_DECV_PAD_S_CV_POY (cnt_DECV);
        CR_LAYER_PAD_E_CV_NKX (L) = CR_DECV_PAD_E_CV_NKX (cnt_DECV);
        CR_LAYER_PAD_S_CV_NKY (L) = CR_DECV_PAD_S_CV_NKY (cnt_DECV);
        CR_LAYER_PAD_N_CV_POY (L) = CR_DECV_PAD_N_CV_POY (cnt_DECV);
        CR_LAYER_PAD_N_CV_NKY (L) = CR_DECV_PAD_N_CV_NKY (cnt_DECV);
        cnt_DECV = cnt_DECV + 1;
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ID of input layers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R_inpx_layer_id = zeros(NUM_LAYER,16);
for L = 1:NUM_LAYER
    for ii = 1:length(input_layers_ID{L})
        R_inpx_layer_id(L,ii) = input_layers_ID{L}(ii);
    end
end

for L = 1:NUM_LAYER
    CR_LAYER_inpx_layer_id_1 (L) = R_inpx_layer_id(L,1 );
    CR_LAYER_inpx_layer_id_2 (L) = R_inpx_layer_id(L,2 );
    CR_LAYER_inpx_layer_id_3 (L) = R_inpx_layer_id(L,3 );
    CR_LAYER_inpx_layer_id_4 (L) = R_inpx_layer_id(L,4 );
    CR_LAYER_inpx_layer_id_5 (L) = R_inpx_layer_id(L,5 );
    CR_LAYER_inpx_layer_id_6 (L) = R_inpx_layer_id(L,6 );
    CR_LAYER_inpx_layer_id_7 (L) = R_inpx_layer_id(L,7 );
    CR_LAYER_inpx_layer_id_8 (L) = R_inpx_layer_id(L,8 );
    CR_LAYER_inpx_layer_id_9 (L) = R_inpx_layer_id(L,9 );
    CR_LAYER_inpx_layer_id_10(L) = R_inpx_layer_id(L,10);
    CR_LAYER_inpx_layer_id_11(L) = R_inpx_layer_id(L,11);
    CR_LAYER_inpx_layer_id_12(L) = R_inpx_layer_id(L,12);
    CR_LAYER_inpx_layer_id_13(L) = R_inpx_layer_id(L,13);
    CR_LAYER_inpx_layer_id_14(L) = R_inpx_layer_id(L,14);
    CR_LAYER_inpx_layer_id_15(L) = R_inpx_layer_id(L,15);
    CR_LAYER_inpx_layer_id_16(L) = R_inpx_layer_id(L,16);
end


%%  %%%%%%%%%%%%% DMA control logic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; cnt_NEAR = 1; 
for L=1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_CONV_inpx_cmd_size   (cnt_CONV);
        CR_LAYER_inpx_addr_adjust(L) = CR_CONV_inpx_addr_adjust(cnt_CONV);
        CR_LAYER_inwt_addr       (L) = CR_CONV_inwt_addr       (cnt_CONV);
        CR_LAYER_inwt_cmd_size   (L) = CR_CONV_inwt_cmd_size   (cnt_CONV);
        CR_LAYER_outpx_cmd_size  (L) = CR_CONV_outpx_cmd_size  (cnt_CONV);
        CR_LAYER_TIX             (L) = NIX_CONV0               (cnt_CONV);
        CR_LAYER_TIY             (L) = Tiy_CONV_DMA            (cnt_CONV);
        CR_LAYER_TIF             (L) = NIF_CONV                (cnt_CONV);
        CR_LAYER_TOF             (L) = Tof_CONV                (cnt_CONV);
        CR_LAYER_num_toy         (L) = CR_CONV_num_toy         (cnt_CONV);
        CR_LAYER_num_tof         (L) = CR_CONV_num_tof         (cnt_CONV);
        CR_LAYER_offset_tiy      (L) = CR_CONV_offset_tiy      (cnt_CONV);
        CR_LAYER_offset_wt_tof   (L) = CR_CONV_offset_wt_tof   (cnt_CONV);
        CR_LAYER_offset_toy      (L) = CR_CONV_offset_toy      (cnt_CONV);
        CR_LAYER_offset_tof      (L) = CR_CONV_offset_tof      (cnt_CONV);
        CR_LAYER_offset_of       (L) = CR_CONV_offset_of       (cnt_CONV);
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_DECV_inpx_cmd_size   (cnt_DECV);
        CR_LAYER_inpx_addr_adjust(L) = CR_DECV_inpx_addr_adjust(cnt_DECV);
        CR_LAYER_inwt_addr       (L) = CR_DECV_inwt_addr       (cnt_DECV);
        CR_LAYER_inwt_cmd_size   (L) = CR_DECV_inwt_cmd_size   (cnt_DECV);
        CR_LAYER_outpx_cmd_size  (L) = CR_DECV_outpx_cmd_size  (cnt_DECV);
        CR_LAYER_TIX             (L) = NIX_DECV0               (cnt_DECV);
        CR_LAYER_TIY             (L) = Tiy_W_DECV_DMA          (cnt_DECV); % 8-bit  % RSP: CR_DECV_TIYDMA_M1 = Tiy_DECV_DMA-1; % 8-bit
        CR_LAYER_TIF             (L) = NIF_DECV                (cnt_DECV);
        CR_LAYER_TOF             (L) = Tof_DECV                (cnt_DECV);
        CR_LAYER_num_toy         (L) = CR_DECV_num_toy         (cnt_DECV);
        CR_LAYER_num_tof         (L) = CR_DECV_num_tof         (cnt_DECV);
        CR_LAYER_offset_tiy      (L) = CR_DECV_offset_tiy      (cnt_DECV);
        CR_LAYER_offset_wt_tof   (L) = CR_DECV_offset_wt_tof   (cnt_DECV);
        CR_LAYER_offset_toy      (L) = CR_DECV_offset_toy      (cnt_DECV);
        CR_LAYER_offset_tof      (L) = CR_DECV_offset_tof      (cnt_DECV);
        CR_LAYER_offset_of       (L) = CR_DECV_offset_of       (cnt_DECV);
        cnt_DECV = cnt_DECV + 1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_PLMX_inpx_cmd_size   (cnt_PLMX);
        CR_LAYER_inpx_addr_adjust(L) = CR_PLMX_inpx_addr_adjust(cnt_PLMX);
        CR_LAYER_outpx_cmd_size  (L) = CR_PLMX_outpx_cmd_size  (cnt_PLMX);
        CR_LAYER_TIX             (L) = Tix_PLMX                (cnt_PLMX);
        CR_LAYER_TIY             (L) = Tiy_PLMX                (cnt_PLMX);
        CR_LAYER_TIF             (L) = Tif_PLMX                (cnt_PLMX);
        CR_LAYER_TOF             (L) = Tof_PLMX                (cnt_PLMX);
        CR_LAYER_num_toy         (L) = CR_PLMX_num_toy         (cnt_PLMX);
        CR_LAYER_num_tif         (L) = CR_PLMX_num_tof         (cnt_PLMX);
        CR_LAYER_num_tof         (L) = CR_PLMX_num_tof         (cnt_PLMX);
        CR_LAYER_offset_tiy      (L) = CR_PLMX_offset_tiy      (cnt_PLMX);
        CR_LAYER_offset_toy      (L) = CR_PLMX_offset_toy      (cnt_PLMX);
        CR_LAYER_offset_tof      (L) = CR_PLMX_offset_tof      (cnt_PLMX);
        CR_LAYER_offset_of       (L) = CR_PLMX_offset_of       (cnt_PLMX);
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_GAPL(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_GAPL_inpx_cmd_size   (cnt_GAPL);
        CR_LAYER_outpx_cmd_size  (L) = CR_GAPL_outpx_cmd_size  (cnt_GAPL);
        CR_LAYER_TIX             (L) = NIX_GAPL0               (cnt_GAPL);
        CR_LAYER_TIY             (L) = NIY_GAPL0               (cnt_GAPL);
        CR_LAYER_TIF             (L) = Tif_GAPL                (cnt_GAPL); % GAPL only divide input fmap into tiles
        CR_LAYER_TOF             (L) = Tif_GAPL                (cnt_GAPL); % GAPL only divide input fmap into tiles 
        CR_LAYER_num_tif         (L) = CR_GAPL_num_tif         (cnt_GAPL); % GAPL only divide input fmap into tiles
        cnt_GAPL = cnt_GAPL + 1;
    end
    if CR_LAYER_IS_PROP(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_PROP_inpx_cmd_size   (cnt_PROP);
        CR_LAYER_inwt_cmd_size   (L) = CR_PROP_inwt_cmd_size   (cnt_PROP);
        CR_LAYER_inwt_addr       (L) = CR_PROP_inwt_addr       (cnt_PROP);
        CR_LAYER_outpx_cmd_size  (L) = CR_PROP_outpx_cmd_size  (cnt_PROP);
        cnt_PROP = cnt_PROP + 1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        CR_LAYER_TIF             (L) = Tof_ROIP                (cnt_ROIP);
        CR_LAYER_TOF             (L) = Tof_ROIP                (cnt_ROIP);
        CR_LAYER_inpx_cmd_size   (L) = CR_ROIP_inpx_cmd_size   (cnt_ROIP);
        CR_LAYER_inwt_cmd_size   (L) = CR_ROIP_inwt_cmd_size   (cnt_ROIP);
        CR_LAYER_inwt_addr       (L) = CR_ROIP_inwt_addr       (cnt_ROIP);
        CR_LAYER_outpx_cmd_size  (L) = CR_ROIP_outpx_cmd_size  (cnt_ROIP);
        CR_LAYER_num_tif         (L) = CR_ROIP_num_tof         (cnt_ROIP); % Tof_ROIP == Tif_ROIP
        CR_LAYER_num_tof         (L) = CR_ROIP_num_tof         (cnt_ROIP);
        CR_LAYER_offset_toy      (L) = CR_ROIP_offset_toy      (cnt_ROIP);
        CR_LAYER_offset_tof      (L) = CR_ROIP_offset_tof      (cnt_ROIP);
        cnt_ROIP = cnt_ROIP + 1;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_EWIS_inpx_cmd_size   (cnt_EWIS);
        CR_LAYER_outpx_cmd_size  (L) = CR_EWIS_outpx_cmd_size  (cnt_EWIS);
        CR_LAYER_TIX             (L) = NIX_EWIS0               (cnt_EWIS);
        CR_LAYER_TIY             (L) = NIY_EWIS0               (cnt_EWIS);
        CR_LAYER_TIF             (L) = Tif_EWIS                (cnt_EWIS);
        CR_LAYER_TOF             (L) = Tof_EWIS                (cnt_EWIS);
        CR_LAYER_num_toy         (L) = CR_EWIS_num_toy         (cnt_EWIS);
        CR_LAYER_offset_tiy      (L) = CR_EWIS_offset_tiy      (cnt_EWIS);
        CR_LAYER_offset_toy      (L) = CR_EWIS_offset_toy      (cnt_EWIS);
        CR_LAYER_offset_of       (L) = CR_EWIS_offset_of       (cnt_EWIS);
        cnt_EWIS = cnt_EWIS + 1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_FCON_inpx_cmd_size   (cnt_FCON);
        CR_LAYER_inwt_addr       (L) = CR_FCON_inwt_addr       (cnt_FCON);
        CR_LAYER_inwt_cmd_size   (L) = CR_FCON_inwt_cmd_size   (cnt_FCON);
        CR_LAYER_outpx_cmd_size  (L) = CR_FCON_outpx_cmd_size  (cnt_FCON);
        CR_LAYER_TIF             (L) = Tif_FCON                (cnt_FCON);
        CR_LAYER_TOF             (L) = Tbx_FCON                (cnt_FCON); % used for dma wr
        CR_LAYER_num_tif         (L) = CR_FCON_num_tif         (cnt_FCON);
        CR_LAYER_num_toy         (L) = 1;
        CR_LAYER_num_tof         (L) = CR_FCON_num_tof         (cnt_FCON);
        CR_LAYER_num_tbx         (L) = CR_FCON_num_tbx         (cnt_FCON);
        CR_LAYER_offset_toy      (L) = 0;
        CR_LAYER_offset_of       (L) = CR_FCON_offset_of       (cnt_FCON);
        CR_LAYER_offset_tof      (L) = CR_FCON_offset_tof      (cnt_FCON);
        cnt_FCON = cnt_FCON + 1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        CR_LAYER_inpx_cmd_size   (L) = CR_NEAR_inpx_cmd_size    (cnt_NEAR);
        CR_LAYER_inpx_addr_adjust(L) = CR_NEAR_inpx_addr_adjust (cnt_NEAR);
        CR_LAYER_outpx_cmd_size  (L) = CR_NEAR_outpx_cmd_size   (cnt_NEAR);
        CR_LAYER_TIX             (L) = NIX_NEAR0                (cnt_NEAR);
        CR_LAYER_TIY             (L) = Tiy_NEAR                 (cnt_NEAR);
        CR_LAYER_TIF             (L) = Tif_NEAR                 (cnt_NEAR);
        CR_LAYER_TOF             (L) = Tof_NEAR                 (cnt_NEAR);
        CR_LAYER_num_toy         (L) = CR_NEAR_num_toy          (cnt_NEAR);
        CR_LAYER_num_tof         (L) = CR_NEAR_num_tof          (cnt_NEAR);
        CR_LAYER_offset_tiy      (L) = CR_NEAR_offset_tiy       (cnt_NEAR);
        CR_LAYER_offset_toy      (L) = CR_NEAR_offset_toy       (cnt_NEAR);
        CR_LAYER_offset_tof      (L) = CR_NEAR_offset_tof       (cnt_NEAR);
        CR_LAYER_offset_of       (L) = CR_NEAR_offset_of        (cnt_NEAR);
        cnt_NEAR = cnt_NEAR + 1;
    end
end


% start from input image, and the last LAYER is not stored in LUT
% 1st addr of LUT stores image information
% 2nd addr of LUT stores the 1st LAYER information
% 3rd addr of LUT stores the 2nd LAYER information
lut_inpx_addr(1) = DDR3_BDEC_IMAGE;
lut_offset_if(1) = Bytes_1OutMap_DATA;
lut_nif      (1) = NIF_LAYER0(1);
for L=2:NUM_LAYER
        lut_inpx_addr(L) = CR_LAYER_outpx_addr(L-1);
        lut_nif      (L) = NOF_LAYER0         (L-1);
end
cnt_CONV = 1; cnt_DECV = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1; cnt_NEAR = 1; 
for L=2:NUM_LAYER
    if CR_LAYER_IS_CONV(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_CONV (cnt_CONV);
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_DECV (cnt_DECV);
        cnt_DECV = cnt_DECV+1;
    end
    if CR_LAYER_IS_NEAR(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_NEAR (cnt_NEAR);
        cnt_NEAR = cnt_NEAR+1;
    end
    if CR_LAYER_IS_PLMX(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_PLMX (cnt_PLMX);
        cnt_PLMX = cnt_PLMX+1;
    end
    if CR_LAYER_IS_GAPL(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_GAPL (cnt_GAPL);
        cnt_GAPL = cnt_GAPL+1;
    end
    if CR_LAYER_IS_ROIP(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_ROIP (cnt_ROIP);
        cnt_ROIP = cnt_ROIP+1;
    end
    if CR_LAYER_IS_EWIS(L-1) == 1
        lut_offset_if(L) = Bytes_1OutMap_EWIS (cnt_EWIS);
        cnt_EWIS = cnt_EWIS+1;
    end
end

fid = fopen('./DMA_LUT.bin','w');
bin_tmp="";
for L=1:NUM_LAYER
    bin_tmp=strcat(dec2bin(lut_offset_if(L),32),dec2bin(lut_inpx_addr(L),32));
    bin_tmp=strcat(dec2bin(lut_nif(L),16),bin_tmp);
    fprintf(fid,bin_tmp);
    fprintf(fid,'\n');
end
fclose(fid);
  


