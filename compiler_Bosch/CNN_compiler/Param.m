%% run through main_Param_CNN.m


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pooling Max (PLMX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R_IS_S1P1_PLMX = zeros(NUM_PLMX,1);
for L = 1:NUM_PLMX
    if NKX_PLMX(L) <= 4 && NKY_PLMX(L) <= 4 && STR_PLMX(L) == 1 && PAD_PLMX(L) == 1
        R_IS_S1P1_PLMX(L) = 1;
    end
end
CR_PLMX_IS_STR1_PAD1_PLMX = R_IS_S1P1_PLMX; % 1-bit

CR_PLMX_PADE_END_PLMX = ceil((NOX_PLMX0-(NOX_PLMX-Tox_PLMX))./POX_PLMX)-1; % 5-bit

CR_PLMX_PADS_END_PLMX = ceil((NOY_PLMX0-(NOY_PLMX-Toy_PLMX))./POY_PLMX)-1; % 8-bit

CR_PLMX_TOXGRP_PLMX_M1 = ceil(Tox_PLMX./POX_PLMX) - 1; % 5-bit

CR_PLMX_TOYGRP_PLMX_M1 = ceil(Toy_PLMX./POY_PLMX) - 1; % 8-bit

CR_PLMX_TOFGRP_PLMX_M1 = ceil(Tof_PLMX./POF_PLMX) - 1; % 6-bit

CR_PLMX_TIXGRP_PLMX = ceil(Tix_PLMX./POX_PLMX); % 5-bit

CR_PLMX_TIXGRP1ST_PLMX = ceil(Tix_PLMX./(POX_PLMX.*STR_PLMX)); % 4-bit

CR_PLMX_TIXY1ST_PLMX = ceil((Tix_PLMX.*Tiy_PLMX)./(POX_PLMX.*STR_PLMX)); % 11-bit


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

LOC_PAD_E_PLMX_idx = NOX_PLMX0-floor(NOX_PLMX0./POX_PLMX).*POX_PLMX;
%LOC_PAD_E_PLMX_idx = NOX_PLMX0+PAD_PLMX-floor((NOX_PLMX0+PAD_PLMX)./POX_PLMX).*POX_PLMX;
LOC_PAD_E_PLMX_bin = repmat('0',NUM_PLMX,POX_PLMX);
for L = 1:NUM_PLMX
    if PAD_E_PLMX(L) == 1
        if LOC_PAD_E_PLMX_idx(L) == 0
            LOC_PAD_E_PLMX_idx(L) = POX_PLMX;
        end
        LOC_PAD_E_PLMX_bin(L,LOC_PAD_E_PLMX_idx(L)) = '1';
    end
    LOC_PAD_E_PLMX_bin(L,:) = fliplr(LOC_PAD_E_PLMX_bin(L,:));
end

CR_PLMX_LOC_PAD_E_PLMX = zeros(NUM_PLMX,1); % POX_PLMX = 16-bit, W_LOC_PAD_E_PL
for i = 1:NUM_PLMX
    CR_PLMX_LOC_PAD_E_PLMX(i) = bin2dec(LOC_PAD_E_PLMX_bin(i,:));
end

CR_PLMX_LOC_PAD_S_PLMX = PAD_S_PLMX; % 1-bit, W_LOC_PAD_S_PL



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Nearest neighbor (NEAR or NN) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CR_NEAR_TOXGRP_NEAR_M1 = ceil(Tox_NEAR./POX_NEAR) - 1; % 5-bit
CR_NEAR_TOYGRP_NEAR_M1 = ceil(Toy_NEAR./POY_NEAR) - 1; % 8-bit
CR_NEAR_TOFGRP_NEAR_M1 = ceil(Tof_NEAR./POF_NEAR) - 1; % 6-bit



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Element-wise (EWIS) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CR_EWIS_WORDS_RDPX_M1       = (NIX_EWIS/PX_AD).*Tiy_EWIS-1;
% CR_EWIS_TIF_M1              = Tif_EWIS-1;
  CR_EWIS_BUFS_PXIN_DEPTHS_EW = (NIX_EWIS/PX_AD).*Tiy_EWIS.*ceil(Tif_EWIS/(BUF_INPX_ALL/2)); % 12-bit
  CR_EWIS_BUFS_PXOU_DEPTHS_EW = (NOX_EWIS/PX_AD).*Toy_EWIS.*ceil(Tof_EWIS/ BUF_OUPX_ALL); % 12-bit


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Global Average Pooling (GAPL) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if NUM_GAPL > 0
    
  CR_GAPL_DIVISION_GAP = round(2^WD_DIV/(NKX_GAPL.*NKY_GAPL)); % csr_DIVISION_GAP
  CR_GAPL_TOFGRP_GAP_M1 = Tof_GAPL/BUF_OUPX_ALL - 1; % csr_TOFGRP_GAP_M1
% CR_GAPL_TIX_GAP = NIX_GAPL0;
% CR_GAPL_TIY_GAP = NIY_GAPL0;
  CR_GAPL_NUM_DDR_WORDS_GAP = ceil(NIX_GAPL0/PX_AD); % csr_NUM_DDR_WORDS_GAP,(Number of DDR words each TIX occupies)
  CR_GAPL_NUM_SKIP_RDADDR_WORDS_GAP = floor((CR_GAPL_NUM_DDR_WORDS_GAP*PX_AD - NIX_GAPL0)/BUF_INPX_1BK); % csr_NUM_SKIP_RDADDR_WORDS_GAP,         //  (NUM_DDR_WORDS_GAP*(NUM_POX_DMA*POX) - TIX_GAP)/BUF_INPX_1BK  e.g. (1*32 - 7)/8= 3 (range 0-3)                                         
  CR_GAPL_NUM_VALID_RDADDR_WORDS_GAP = NUM_POX_DMA - CR_GAPL_NUM_SKIP_RDADDR_WORDS_GAP; % csr_NUM_VALID_RDADDR_WORDS_GAP, //  NUM_POX_DMA - csr_NUM_SKIP_RDADDR_WORDS_GAP e.g. 4-3 =1 (range 0-3)       
  CR_GAPL_BUFS_PXIN_DEPTHS_GAP =  NUM_POX_DMA.*ceil(NIX_GAPL0/PX_AD).*NIY_GAPL0.*ceil(Tif_GAPL/BUF_INPX_1BK); % csr_BUFS_PXIN_DEPTHS_GAP, //e.g. 4 * ceil(7/32) * 7 * ceil(2048/8) = 4*1792 = 7168 (max = 4*2048) (BUffer depth to store 1 tile)

end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Proposal (PROP) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CR_PROP_ch_width    = NIX_PROP0;         % 8-bit, Input Feature Map Width
CR_PROP_ch_height   = NIY_PROP0;         % 8-bit, Input Feature Map Height
CR_PROP_min_w       = 0;                 % 16-bit, bit_frac = 4; Min Box Width  (pre-NMS), Keep proposal if the box's size is ? the minimum width
CR_PROP_min_h       = 0;                 % 16-bit, bit_frac = 4; Min Box Height (pre-NMS), Keep proposal if the box's size is ? the minimum height
CR_PROP_img_w       = NIX_CONV0(1);      % 11-bit, Per input to first conv layer; used for clipping after anchor shifting
CR_PROP_img_h       = NIY_CONV0(1);      % 11-bit, Per input to first conv layer; used for clipping after anchor shifting
CR_PROP_num_anchors = num_anchors;       % 7-bit, Number of Base Anchors
CR_PROP_numPostNMS  = NBX_ROIP0;         % 9-bit
CR_PROP_enable_variance_by_apprx   = 0   *ones(NUM_PROP,1);

%   Field Name	DLA 2.0?	Field Size	sign	int	frac	Description	Comment
% csr_anchor_ctr_x	Y       16      1	11	4	Anchor Center X	Common center for all base anchors
% csr_anchor_ctr_y	Y       16      1	11	4	Anchor Center Y	Common center for all base anchors
% csr_anchor_w      Y       49x16	1	11	4	Anchor Width	Supports a maximum of 49 base anchors (7 ratios x 7 scales)
% csr_anchor_h      Y       49x16	1	11	4	Anchor Height	--> One W/H pair for each ratio / scale combination


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROIPooling (ROIP) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ROIP module processes all ROIs in one tile.
CR_ROIP_SPATIAL_SCALE            = log2(NIX_CONV0(1)./NIX_ROIP0); % Num of down scales, Number of PLMX layers prior to ROI Pooling in the network
CR_ROIP_POOL_SIZE                = NOX_ROIP0; % Only square grids are supported.
CR_ROIP_NUM_CHANNELS             = NOF_ROIP0;
CR_ROIP_NUM_CHANNELS_PER_BUFFER  = Tof_ROIP/BUF_INPX_1BK;
CR_ROIP_NUM_FRACTIONAL_BITS      = 4; % Num of fractional bits for input ROI co-ordinates
CR_ROIP_GRID_DIVISION_EQUIVALENT = round(2^16./NOX_ROIP0);
CR_ROIP_CH_HEIGHT_ROIPL          = NIY_ROIP0; % Height of inpx fmap
CR_ROIP_CH_WIDTH_ROIPL           = NIX_ROIP0; % Width of inpx fmap

for C = 1:NUM_ROIP
   if  NOX_ROIP0(C) ~= NOY_ROIP0
       fprintf('Error @ Param.m: DLA ROIP cannot support NOX_ROIP0 != NOY_ROIP0 \n\n')
   end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fully_connected (FCON) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CR_FCON_NUM_ADDRS_1BOX_M1 = zeros(NUM_FCON,1); % 11-bit, csr_NUM_ADDRS_1BOX_M1, # of INPX2 addresses of one box
CR_FCON_NIX_NUM_ADDRS     = zeros(NUM_FCON,1); %  2-bit, csr_NIX_NUM_ADDRS, # of INPX2 addresses of one input row
CR_FCON_NUM_VALID_PX_M1   = zeros(NUM_FCON,1); %  4-bit, csr_NUM_VALID_PX_M1
CR_FCON_NUM_BOX_1BUF_M1   = zeros(NUM_FCON,1); %  5-bti, csr_NUM_BOX_1BUF_M1, # of boxes stored in one input buffer
CR_FCON_NUM_NIX_INPX2     = zeros(NUM_FCON,1); %  2-bit, csr_NUM_NIX_INPX2, # of feature rows (NOX_CONV) in one inpx2 address

for L = 1:NUM_FCON
    Prior_Layer = input_layers_ID{ID_global_FCON(L)}; % Assume FCON only has one input layer
    if CR_LAYER_IS_FCON(Prior_Layer)
        ID_Local = find(ID_global_FCON == Prior_Layer);
        CR_FCON_NUM_ADDRS_1BOX_M1(L) = NOF_FCON(ID_Local)/(2*POX) - 1;
        CR_FCON_NIX_NUM_ADDRS(L) = 1;
        CR_FCON_NUM_VALID_PX_M1(L) = 2*POX-1;
        CR_FCON_NUM_BOX_1BUF_M1(L) = max(Tbx_FCON(L)/BUF_INPX_1BK - 1,0);
        CR_FCON_NUM_NIX_INPX2(L) = 1-1; 
    end       
    if CR_LAYER_IS_ROIP(Prior_Layer)
        ID_Local = find(ID_global_ROIP == Prior_Layer);
        CR_FCON_NUM_ADDRS_1BOX_M1(L) = ceil( (NOY_ROIP(ID_Local)*NOF_ROIP(ID_Local))/(2*POX/NOX_ROIP(ID_Local)) ) - 1;
%       CR_FCON_NUM_ADDRS_1BOX_M1(L) = (NOY_ROIP(ID_Local)/(2*POX/NOX_ROIP(ID_Local)))*NOF_ROIP(ID_Local) - 1;
        CR_FCON_NIX_NUM_ADDRS(L) = ceil(NOX_ROIP(ID_Local)/(2*POX));
        CR_FCON_NUM_VALID_PX_M1(L) = NOX_ROIP0(ID_Local)-1;
        CR_FCON_NUM_BOX_1BUF_M1(L) = max(Tbx_FCON(L)/BUF_INPX_1BK - 1,0);
        CR_FCON_NUM_NIX_INPX2(L) = 2*POX/NOX_ROIP(ID_Local)-1; 
    end
    if CR_LAYER_IS_CONV(Prior_Layer)
        ID_Local = find(ID_global_CONV == Prior_Layer);
        if NOX_CONV0(ID_Local) <= 2*POX
            CR_FCON_NUM_ADDRS_1BOX_M1(L) = (NOX_CONV(ID_Local)/(2*POX))*NOY_CONV0(ID_Local)*NOF_CONV(ID_Local) - 1;
            CR_FCON_NIX_NUM_ADDRS(L) = ceil(PX_AD/(2*POX));
            CR_FCON_NUM_VALID_PX_M1(L) = NOX_CONV0(ID_Local)-1;
            CR_FCON_NUM_BOX_1BUF_M1(L) = 0;
            CR_FCON_NUM_NIX_INPX2(L) = 1-1; 
        else
            fprintf('Error @ Param.m: DLA cannot support NOX_CONV0 > 2*POX before FCON\n\n')
            Error
        end
    end
    if CR_LAYER_IS_PLMX(Prior_Layer)
        ID_Local = find(ID_global_PLMX == Prior_Layer);
        if NOX_PLMX0(ID_Local) <= 2*POX
            CR_FCON_NUM_ADDRS_1BOX_M1(L) = (NOX_PLMX(ID_Local)/(2*POX))*NOY_PLMX0(ID_Local)*NOF_PLMX(ID_Local) - 1;
            CR_FCON_NIX_NUM_ADDRS(L) = ceil(PX_AD/(2*POX));
            CR_FCON_NUM_VALID_PX_M1(L) = NOX_PLMX0(ID_Local)-1;
            CR_FCON_NUM_BOX_1BUF_M1(L) = 0;
            CR_FCON_NUM_NIX_INPX2(L) = 1-1; 
        else
            fprintf('Error @ Param.m: DLA cannot support NOX_PLMX0 > 2*POX before FCON\n\n')
            Error
        end
    end
    if CR_LAYER_IS_GAPL(Prior_Layer)
        ID_Local = find(ID_global_GAPL == Prior_Layer);
        CR_FCON_NUM_ADDRS_1BOX_M1(L) = NOF_GAPL(ID_Local)/(2*POX) - 1;
        CR_FCON_NIX_NUM_ADDRS(L) = 1;
        CR_FCON_NUM_VALID_PX_M1(L) = 2*POX-1;
        CR_FCON_NUM_BOX_1BUF_M1(L) = 0; % # of boxes (NBX) = 1
        CR_FCON_NUM_NIX_INPX2(L) = 1-1; 
    end    
end

%CR_FCON_TIF_FCON_M1 = Tif_FCON -1; %14-bit, csr_TIF_FCON_M1
CR_FCON_NIF_FCON_M1 = NIF_FCON0-1; %15-bit, csr_NIF_FCON_M1;

CR_FCON_NOF1POF_FCON_M1 = NOF_FCON/POF_FCON-1; % 8-bit, csr_NOF1POF_FCON_M1
CR_FCON_ROI_TILES_FCON_M1 = NBX_FCON./Tbx_FCON-1; % 5-bit, csr_ROI_TILES_FCON_M1


% CR_FCON_TILE_MAC_FC_M1 = zeros(NUM_FCON,1); % 10-bit
% for L = 1:NUM_FCON
%     if NIF_FCON(L) <= FCWT_WORDS
%         CR_FCON_TILE_MAC_FC_M1(L) = NUM_TILE_DMA_pFCON(L)-1;
%     else
%         CR_FCON_TILE_MAC_FC_M1(L) = ceil(NOF_FCON(L)/POF_FCON)-1;
%     end
% end
% 
% CR_FCON_NIF_FC_M1 = NIF_FCON - 1; % 15-bit
% 
% CR_FCON_NOF_FC_M1 = NOF_FCON - 1; % 13-bit
% 
% CR_FCON_NIFGRP_FC_M1 = ceil(FCWT_WORDS./NIF_FCON) - 1; % 10-bit
% 
% CR_FCON_NIFGRP_POF_FC = POF_FCON*ceil(FCWT_WORDS./NIF_FCON); % 15-bit








%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of one DMA dpt transaction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% R_WORDS_RDPX_EWIS_M1 = NIX_EWIS0*8/DMA_WIDTH - 1; % TODO TODO TODO EWIS
% R_WORDS_RDPX_PLMX_M1 = CR_PLMX_RD_PX_dpt_bytes*8/DMA_WIDTH-1;
% CR_LAYER_WORDS_RDPX_M1 = zeros(NUM_LAYER,1); % 12-bit, W_WORDS_RDPX_M1 for Eltwise, Pool_max
% cnt_EWIS = 1;cnt_PLMX = 1;
% for L = 1:NUM_LAYER
%     if CR_LAYER_IS_EWIS(L) == 1
%         CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_EWIS_M1(cnt_EWIS);
%         cnt_EWIS = cnt_EWIS + 1;
%     end
%     if CR_LAYER_IS_PLMX(L) == 1
%         CR_LAYER_WORDS_RDPX_M1(L) = R_WORDS_RDPX_PLMX_M1(cnt_PLMX);
%         cnt_PLMX = cnt_PLMX + 1;
%     end
% end
% 
% R_WORDS_WRPX_CONV_M1 = CR_CONV_WR_PX_dpt_bytes*8/DMA_WIDTH - 1;
% R_WORDS_WRPX_PLMX_M1 = CR_PLMX_WR_PX_dpt_bytes*8/DMA_WIDTH - 1;
% R_WORDS_WRPX_DECV_M1 = CR_DECV_WR_PX_dpt_bytes*8/DMA_WIDTH - 1; % RSP: Deconv Updates
% CR_LAYER_WORDS_WRPX_M1 = zeros(NUM_LAYER,1); % 10-bit, W_WORDS_WRPX_M1
% cnt_CONV = 1;cnt_PLMX = 1;cnt_DECV = 1;
% for L = 1:NUM_LAYER
%     if CR_LAYER_IS_CONV(L) == 1
%         CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_CONV_M1(cnt_CONV);
%         cnt_CONV = cnt_CONV + 1;
%     end
%     if CR_LAYER_IS_PLMX(L) == 1
%         CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_PLMX_M1(cnt_PLMX);
%         cnt_PLMX = cnt_PLMX + 1;
%     end
%     if CR_LAYER_IS_DECV(L) == 1 % RSP: Deconv Updates
%         CR_LAYER_WORDS_WRPX_M1(L) = R_WORDS_WRPX_DECV_M1(cnt_DECV);
%         cnt_DECV = cnt_DECV + 1;
%     end
% end


%% DMA descriptors

% before DDR3_BADDR_IMAGE % NOT used
REG_DDR3_ENDADDR_WT = DDR3_BDEC_WT_FCON + (JTAG_WORDS*JTAG_WIDTH*(NUM_JTAG_MIF_FC-1)/8); % stored in csr register


%%  %%%%%%%%%%%%% CONV zero padding %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R_END_TOYPAD_CONV = mod(ceil(NOY_CONV0/POY),ceil(Toy_CONV/POY));
for L = 1:NUM_CONV
   if  R_END_TOYPAD_CONV(L) == 0
       R_END_TOYPAD_CONV(L) = ceil(Toy_CONV(L)/POY);
   end    
end
CR_CONV_END_TOYPAD_M1 = R_END_TOYPAD_CONV - 1; % 5-bit

CR_CONV_END_NOYPAD_M1 = ceil(NOY_CONV0./Toy_CONV) - 1; % 7-bit, W_END_NOYPAD_M1, TODO check

% For example, 
% NUM_PADS_POX_CONV = 2, => 2 MACs in POX axis have zero paddings as input
% R_PAD_E_CONV_loc = 4, start from the 4th MAC (e.g. MAC[3]) in the POX axis, the MACs have zero padding as inputs
% R_PAD_E_CONV_POX = 00001100 => MAC[2] and MAC[3] have zero paddings as inputs, others NO

NUM_PADS_POX_CONV = ceil(((NOX_CONV0-1).*STR_CONV+NKX_CONV - (NIX_CONV0+PAD_CONV))./STR_CONV); % # of MACs with padding inputs in POX axis
NUM_PADS_POY_CONV = ceil(((NOY_CONV0-1).*STR_CONV+NKY_CONV - (NIY_CONV0+PAD_CONV))./STR_CONV); % # of MACs with padding inputs in POY axis

R_PAD_E_CONV_loc = NOX_CONV0-floor(NOX_CONV0/POX)*POX; % the location of paddings in X axis
R_PAD_S_CONV_loc = NOY_CONV0-floor(NOY_CONV0/POY)*POY; % the location of paddings in Y axis
R_PAD_E_CONV_POX = repmat('0',NUM_CONV,POX); % East  pad only occurs at the end of tox, indicate which MAC has padding inputs
R_PAD_S_CONV_POY = repmat('0',NUM_CONV,POY); % South pad only occurs at the end of toy, indicate which MAC has padding inputs
for L = 1:NUM_CONV
    if PAD_CONV(L) ~= 0
        if R_PAD_E_CONV_loc(L) == 0
            R_PAD_E_CONV_loc(L) = POX;
        end
        if NUM_PADS_POX_CONV(L) > 0
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-0) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 1
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-1) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 2
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-2) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 3
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-3) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 4
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-4) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 5
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-5) = '1';
        end
        if NUM_PADS_POX_CONV(L) > 6
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-6) = '1';
        end 
        if NUM_PADS_POX_CONV(L) > 7
            R_PAD_E_CONV_POX(L,R_PAD_E_CONV_loc(L)-7) = '1';
        end 
    end
    R_PAD_E_CONV_POX(L,:) = fliplr(R_PAD_E_CONV_POX(L,:));
end
for L = 1:NUM_CONV
    if PAD_CONV(L) ~= 0
        if R_PAD_S_CONV_loc(L) == 0
            R_PAD_S_CONV_loc(L) = POY;
        end
        if NUM_PADS_POY_CONV(L) > 0
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-0) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 1
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-1) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 2
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-2) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 3
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-3) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 4
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-4) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 5
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-5) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 6
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-6) = '1';
        end
        if NUM_PADS_POY_CONV(L) > 7
            R_PAD_S_CONV_POY(L,R_PAD_S_CONV_loc(L)-7) = '1';
        end
    end
    R_PAD_S_CONV_POY(L,:) = fliplr(R_PAD_S_CONV_POY(L,:));
end

% Threshold (TH) for kx/ky counters.
% At E/S boundary, if kx/ky >= TH, the inputs of the selected MACs should be zeros.
R_PAD_E_CONV_TH1 = NKX_CONV - ((NOX_CONV0-1).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH1 = NKY_CONV - ((NOY_CONV0-1).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH2 = NKX_CONV - ((NOX_CONV0-2).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH2 = NKY_CONV - ((NOY_CONV0-2).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH3 = NKX_CONV - ((NOX_CONV0-3).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH3 = NKY_CONV - ((NOY_CONV0-3).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH4 = NKX_CONV - ((NOX_CONV0-4).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH4 = NKY_CONV - ((NOY_CONV0-4).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH5 = NKX_CONV - ((NOX_CONV0-5).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH5 = NKY_CONV - ((NOY_CONV0-5).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH6 = NKX_CONV - ((NOX_CONV0-6).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH6 = NKY_CONV - ((NOY_CONV0-6).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH7 = NKX_CONV - ((NOX_CONV0-7).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH7 = NKY_CONV - ((NOY_CONV0-7).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_TH8 = NKX_CONV - ((NOX_CONV0-8).*STR_CONV+NKX_CONV-(NIX_CONV0+PAD_CONV));
R_PAD_S_CONV_TH8 = NKY_CONV - ((NOY_CONV0-8).*STR_CONV+NKY_CONV-(NIY_CONV0+PAD_CONV));

R_PAD_E_CONV_NKX = ones(NUM_CONV,POX)*(2^WD_KXKY-1);
R_PAD_S_CONV_NKY = ones(NUM_CONV,POX)*(2^WD_KXKY-1);

for L = 1:NUM_CONV
    if NUM_PADS_POX_CONV(L) > 0
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-0) = R_PAD_E_CONV_TH1(L);
    end
    if NUM_PADS_POX_CONV(L) > 1
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-1) = R_PAD_E_CONV_TH2(L);
    end   
    if NUM_PADS_POX_CONV(L) > 2
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-2) = R_PAD_E_CONV_TH3(L);
    end       
    if NUM_PADS_POX_CONV(L) > 3
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-3) = R_PAD_E_CONV_TH4(L);
    end
    if NUM_PADS_POX_CONV(L) > 4
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-4) = R_PAD_E_CONV_TH5(L);
    end
    if NUM_PADS_POX_CONV(L) > 5
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-5) = R_PAD_E_CONV_TH6(L);
    end
    if NUM_PADS_POX_CONV(L) > 6
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-6) = R_PAD_E_CONV_TH7(L);
    end
    if NUM_PADS_POX_CONV(L) > 7
        R_PAD_E_CONV_NKX(L,R_PAD_E_CONV_loc(L)-7) = R_PAD_E_CONV_TH8(L);
    end
end
for L = 1:NUM_CONV
    if NUM_PADS_POY_CONV(L) > 0
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-0) = R_PAD_S_CONV_TH1(L);
    end
    if NUM_PADS_POY_CONV(L) > 1
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-1) = R_PAD_S_CONV_TH2(L);
    end   
    if NUM_PADS_POY_CONV(L) > 2
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-2) = R_PAD_S_CONV_TH3(L);
    end       
    if NUM_PADS_POY_CONV(L) > 3
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-3) = R_PAD_S_CONV_TH4(L);
    end
    if NUM_PADS_POY_CONV(L) > 4
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-4) = R_PAD_S_CONV_TH5(L);
    end
    if NUM_PADS_POY_CONV(L) > 5
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-5) = R_PAD_S_CONV_TH6(L);
    end
    if NUM_PADS_POY_CONV(L) > 6
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-6) = R_PAD_S_CONV_TH7(L);
    end
    if NUM_PADS_POY_CONV(L) > 7
        R_PAD_S_CONV_NKY(L,R_PAD_S_CONV_loc(L)-7) = R_PAD_S_CONV_TH8(L);
    end
end

CR_CONV_PAD_E_CV_POX = zeros(NUM_CONV,1); % POX-bit
CR_CONV_PAD_S_CV_POY = zeros(NUM_CONV,1); % POY-bit
for i = 1:NUM_CONV
    CR_CONV_PAD_E_CV_POX(i) = bin2dec(R_PAD_E_CONV_POX(i,:));
    CR_CONV_PAD_S_CV_POY(i) = bin2dec(R_PAD_S_CONV_POY(i,:));
end


if WD_KXKY < ceil(log2(max(max(R_PAD_E_CONV_NKX(:,:)))+1)) 
    fprintf('ERROR 31 @ Param.m : R_PAD_E_CONV_NKX\n') 
end
CR_CONV_PAD_E_CV_NKX = zeros(NUM_CONV,1); % POX*WD_KXKY = 8*4 = 32-bit
for i = 1:NUM_CONV
    R_PAD_CV_bin = '';
    for x = 1:POX
        bin_tmp = dec2bin(R_PAD_E_CONV_NKX(i,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    %%%%CR_CONV_PAD_E_CV_NKX(i) = bin2dec(R_PAD_CV_bin);
end


if WD_KXKY < ceil(log2(max(max(R_PAD_S_CONV_NKY(:,:)))+1)) 
    fprintf('ERROR 32 @ Param.m : R_PAD_S_CONV_NKY\n') 
end
CR_CONV_PAD_S_CV_NKY = zeros(NUM_CONV,1); % POY*WD_KXKY = 8*4 = 32-bit
for i = 1:NUM_CONV
    R_PAD_CV_bin = '';
    for x = 1:POY
        bin_tmp = dec2bin(R_PAD_S_CONV_NKY(i,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    %%%%CR_CONV_PAD_S_CV_NKY(i) = bin2dec(R_PAD_CV_bin);
end


PAD_N_POY = ceil(PAD_CONV./STR_CONV);
R_PAD_N_CONV_POY = repmat('0',NUM_CONV,POY);
for L = 1:NUM_CONV
    if PAD_CONV(L) ~= 0
        for P = 1:PAD_N_POY(L)
            R_PAD_N_CONV_POY(L,P) = '1';
        end
    end
    R_PAD_N_CONV_POY(L,:) = fliplr(R_PAD_N_CONV_POY(L,:));
end
CR_CONV_PAD_N_CV_POY = zeros(NUM_CONV,1); % POY = 8-bit
for L = 1:NUM_CONV
    CR_CONV_PAD_N_CV_POY(L,:) = bin2dec(R_PAD_N_CONV_POY(L,:));
end


%MAX_PAD1STR = 8; %support max(ceil(PAD_CONV/STR_CONV)) = 8
R_PAD_N_CONV_NKY = zeros(NUM_CONV,MAX_PAD1STR);
for L = 1:NUM_CONV
    for i = 1:MAX_PAD1STR
        if PAD_N_POY(L) >= i
            R_PAD_N_CONV_NKY(L,i) = PAD_CONV(L)-(i-1)*STR_CONV(L);
        end      
    end
end

if WD_KXKY < ceil(log2(max(max(R_PAD_N_CONV_NKY(:,:)))+2)) 
    fprintf('ERROR 33 @ Param.m : R_PAD_N_CONV_NKY\n') 
end
CR_CONV_PAD_N_CV_NKY = zeros(NUM_CONV,1); % WD_KXKY*MAX_PAD1STR = 4*8 = 32-bit, W_PAD_N_CV_NKY need to change bit width
for L = 1:NUM_CONV
    R_PAD_CV_bin = '';
    for x = 1:MAX_PAD1STR
        bin_tmp = dec2bin(R_PAD_N_CONV_NKY(L,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    CR_CONV_PAD_N_CV_NKY(L) = bin2dec(R_PAD_CV_bin);
end



%%  %%%%%%%%%%%%% DECV zero padding %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RSP: R_END_TOYPAD_DEDECV = mod(ceil(NOY_DECV0/POY),ceil(Toy_DECV/POY));
R_END_TOYPAD_DECV = mod(ceil(NOY_DECV0/POY),ceil(Toy_W_DECV/POY));
for L = 1:NUM_DECV
   if  R_END_TOYPAD_DECV(L) == 0
       R_END_TOYPAD_DECV(L) = ceil(Toy_W_DECV(L)/POY);
   end    
end
CR_DECV_END_TOYPAD_M1 = R_END_TOYPAD_DECV - 1; % 5-bit
% RSP: CR_DECV_END_NOYPAD_M1 = ceil(NOY_DECV0./Toy_DECV) - 1; % 7-bit, W_END_NOYPAD_M1, TODO check
CR_DECV_END_NOYPAD_M1 = ceil(NOY_DECV0./Toy_W_DECV) - 1; % 7-bit, W_END_NOYPAD_M1, TODO check

% RSP: NUM_PADS_POX_DECV = ceil(((NOX_DECV0-1).*STR_DECV+NKX_DECV - (NIX_DECV0+PAD_DECV))./STR_DECV); % # of paddings in X axis
% RSP: NUM_PADS_POY_DECV = ceil(((NOY_DECV0-1).*STR_DECV+NKY_DECV - (NIY_DECV0+PAD_DECV))./STR_DECV); % # of paddings in Y axis
NUM_PADS_POX_DECV = ceil(((NOX_W_DECV0-1).*STR_D+NKX_DECV - (NIX_W_DECV0+PAD_D))./STR_D); % # of paddings in X axis
NUM_PADS_POY_DECV = ceil(((NOY_W_DECV0-1).*STR_D+NKY_DECV - (NIY_W_DECV0+PAD_D))./STR_D); % # of paddings in Y axis

R_PAD_E_DECV_loc = NOX_W_DECV0-floor(NOX_W_DECV0/POX)*POX; % the location of paddings in X axis
R_PAD_S_DECV_loc = NOY_W_DECV0-floor(NOY_W_DECV0/POY)*POY; % the location of paddings in Y axis
R_PAD_E_DECV_POX = repmat('0',NUM_DECV,POX); % East  pad only occurs at the end of tox
R_PAD_S_DECV_POY = repmat('0',NUM_DECV,POY); % South pad only occurs at the end of toy
for L = 1:NUM_DECV
    if PAD_D(L) ~= 0
        if R_PAD_E_DECV_loc(L) == 0
            R_PAD_E_DECV_loc(L) = POX;
        end
        if NUM_PADS_POX_DECV(L) > 0
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-0) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 1
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-1) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 2
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-2) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 3
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-3) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 4
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-4) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 5
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-5) = '1';
        end
        if NUM_PADS_POX_DECV(L) > 6
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-6) = '1';
        end 
        if NUM_PADS_POX_DECV(L) > 7
            R_PAD_E_DECV_POX(L,R_PAD_E_DECV_loc(L)-7) = '1';
        end 
    end
    R_PAD_E_DECV_POX(L,:) = fliplr(R_PAD_E_DECV_POX(L,:));
end
for L = 1:NUM_DECV
    if PAD_D(L) ~= 0
        if R_PAD_S_DECV_loc(L) == 0
            R_PAD_S_DECV_loc(L) = POY;
        end
        if NUM_PADS_POY_DECV(L) > 0
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-0) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 1
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-1) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 2
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-2) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 3
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-3) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 4
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-4) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 5
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-5) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 6
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-6) = '1';
        end
        if NUM_PADS_POY_DECV(L) > 7
            R_PAD_S_DECV_POY(L,R_PAD_S_DECV_loc(L)-7) = '1';
        end
    end
    R_PAD_S_DECV_POY(L,:) = fliplr(R_PAD_S_DECV_POY(L,:));
end

R_PAD_E_DECV_TH1 = NKX_DECV - ((NOX_W_DECV0-1).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH1 = NKY_DECV - ((NOY_W_DECV0-1).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH2 = NKX_DECV - ((NOX_W_DECV0-2).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH2 = NKY_DECV - ((NOY_W_DECV0-2).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH3 = NKX_DECV - ((NOX_W_DECV0-3).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH3 = NKY_DECV - ((NOY_W_DECV0-3).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH4 = NKX_DECV - ((NOX_W_DECV0-4).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH4 = NKY_DECV - ((NOY_W_DECV0-4).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH5 = NKX_DECV - ((NOX_W_DECV0-5).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH5 = NKY_DECV - ((NOY_W_DECV0-5).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH6 = NKX_DECV - ((NOX_W_DECV0-6).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH6 = NKY_DECV - ((NOY_W_DECV0-6).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH7 = NKX_DECV - ((NOX_W_DECV0-7).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH7 = NKY_DECV - ((NOY_W_DECV0-7).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_TH8 = NKX_DECV - ((NOX_W_DECV0-8).*STR_D+NKX_DECV-(NIX_W_DECV0+PAD_D));
R_PAD_S_DECV_TH8 = NKY_DECV - ((NOY_W_DECV0-8).*STR_D+NKY_DECV-(NIY_W_DECV0+PAD_D));

R_PAD_E_DECV_NKX = ones(NUM_DECV,POX)*(2^4-1);
R_PAD_S_DECV_NKY = ones(NUM_DECV,POX)*(2^4-1);

for L = 1:NUM_DECV
    if NUM_PADS_POX_DECV(L) > 0
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-0) = R_PAD_E_DECV_TH1(L);
    end
    if NUM_PADS_POX_DECV(L) > 1
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-1) = R_PAD_E_DECV_TH2(L);
    end   
    if NUM_PADS_POX_DECV(L) > 2
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-2) = R_PAD_E_DECV_TH3(L);
    end       
    if NUM_PADS_POX_DECV(L) > 3
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-3) = R_PAD_E_DECV_TH4(L);
    end
    if NUM_PADS_POX_DECV(L) > 4
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-4) = R_PAD_E_DECV_TH5(L);
    end
    if NUM_PADS_POX_DECV(L) > 5
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-5) = R_PAD_E_DECV_TH6(L);
    end
    if NUM_PADS_POX_DECV(L) > 6
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-6) = R_PAD_E_DECV_TH7(L);
    end
    if NUM_PADS_POX_DECV(L) > 7
        R_PAD_E_DECV_NKX(L,R_PAD_E_DECV_loc(L)-7) = R_PAD_E_DECV_TH8(L);
    end
end
for L = 1:NUM_DECV
    if NUM_PADS_POY_DECV(L) > 0
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-0) = R_PAD_S_DECV_TH1(L);
    end
    if NUM_PADS_POY_DECV(L) > 1
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-1) = R_PAD_S_DECV_TH2(L);
    end   
    if NUM_PADS_POY_DECV(L) > 2
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-2) = R_PAD_S_DECV_TH3(L);
    end       
    if NUM_PADS_POY_DECV(L) > 3
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-3) = R_PAD_S_DECV_TH4(L);
    end
    if NUM_PADS_POY_DECV(L) > 4
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-4) = R_PAD_S_DECV_TH5(L);
    end
    if NUM_PADS_POY_DECV(L) > 5
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-5) = R_PAD_S_DECV_TH6(L);
    end
    if NUM_PADS_POY_DECV(L) > 6
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-6) = R_PAD_S_DECV_TH7(L);
    end
    if NUM_PADS_POY_DECV(L) > 7
        R_PAD_S_DECV_NKY(L,R_PAD_S_DECV_loc(L)-7) = R_PAD_S_DECV_TH8(L);
    end
end

CR_DECV_PAD_E_CV_POX = zeros(NUM_DECV,1); % POX-bit
CR_DECV_PAD_S_CV_POY = zeros(NUM_DECV,1); % POY-bit
for i = 1:NUM_DECV
    CR_DECV_PAD_E_CV_POX(i) = bin2dec(R_PAD_E_DECV_POX(i,:));
    CR_DECV_PAD_S_CV_POY(i) = bin2dec(R_PAD_S_DECV_POY(i,:));
end


if WD_KXKY < ceil(log2(max(max(R_PAD_E_DECV_NKX(:,:)))+1)) 
    fprintf('ERROR 31 @ Param.m : R_PAD_E_DECV_NKX\n') 
end
CR_DECV_PAD_E_CV_NKX = zeros(NUM_DECV,1); % WD_KXKY*POX = 4*8 = 32-bit
for i = 1:NUM_DECV
    R_PAD_CV_bin = '';
    for x = 1:POX
        bin_tmp = dec2bin(R_PAD_E_DECV_NKX(i,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    %%%%CR_DECV_PAD_E_CV_NKX(i) = bin2dec(R_PAD_CV_bin);
end


if WD_KXKY < ceil(log2(max(max(R_PAD_S_DECV_NKY(:,:)))+1)) 
    fprintf('ERROR 32 @ Param.m : R_PAD_S_DECV_NKY\n') 
end
CR_DECV_PAD_S_CV_NKY = zeros(NUM_DECV,1); % WD_KXKY*POY = 4*8 = 32-bit
for i = 1:NUM_DECV
    R_PAD_CV_bin = '';
    for x = 1:POY
        bin_tmp = dec2bin(R_PAD_S_DECV_NKY(i,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    CR_DECV_PAD_S_CV_NKY(i) = bin2dec(R_PAD_CV_bin);
end

PAD_N_POY = ceil(PAD_D./STR_D);
R_PAD_N_DECV_POY = repmat('0',NUM_DECV,POY);
for L = 1:NUM_DECV
    if PAD_D(L) ~= 0
        for P = 1:PAD_N_POY(L)
            R_PAD_N_DECV_POY(L,P) = '1';
        end
    end
    R_PAD_N_DECV_POY(L,:) = fliplr(R_PAD_N_DECV_POY(L,:));
end
CR_DECV_PAD_N_CV_POY = zeros(NUM_DECV,1); % POY = 8-bit
for L = 1:NUM_DECV
    CR_DECV_PAD_N_CV_POY(L,:) = bin2dec(R_PAD_N_DECV_POY(L,:));
end


%MAX_PAD1STR = 8; %support max(ceil(PAD_DECV/STR_DECV)) = 8
R_PAD_N_DECV_NKY = zeros(NUM_DECV,MAX_PAD1STR);
for L = 1:NUM_DECV
    for i = 1:MAX_PAD1STR
        if PAD_N_POY(L) >= i
            R_PAD_N_DECV_NKY(L,i) = PAD_D(L)-(i-1)*STR_D(L);
        end      
    end
end

if WD_KXKY < ceil(log2(max(max(R_PAD_N_DECV_NKY(:,:)))+2)) 
    fprintf('ERROR 33 @ Param.m : R_PAD_N_DECV_NKY\n') 
end
CR_DECV_PAD_N_CV_NKY = zeros(NUM_DECV,1); % WD_KXKY*MAX_PAD1STR = 4*8 = 32-bit, W_PAD_N_CV_NKY need to change bit width
for L = 1:NUM_DECV
    R_PAD_CV_bin = '';
    for x = 1:MAX_PAD1STR
        bin_tmp = dec2bin(R_PAD_N_DECV_NKY(L,x),WD_KXKY);
        R_PAD_CV_bin = [bin_tmp,R_PAD_CV_bin];
    end
    CR_DECV_PAD_N_CV_NKY(L) = bin2dec(R_PAD_CV_bin);
end




%%
fclose(fid);

fprintf('parameters.v generated \n\n');


