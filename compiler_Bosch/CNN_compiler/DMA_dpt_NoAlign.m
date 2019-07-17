
DDR3_BDEC_WT_BASE = 0; % will be set by software
DDR3_BDEC_IMAGE_BASE = 0;
DDR3_BDEC_LAYER_BASE = 0;

%DDR3_BDEC_WT_CONV = 0 + JTAG_WORDS*JTAG_WIDTH/8; % add margin address before conv weights

DDR3_BDEC_WT_CONV = DDR3_BDEC_WT_BASE;

% BUF_OUTFC_AB_BDEC = hex2dec('00000000'); %//32'h0020_0000 ~ 32'h0021_FFFF
% BUF_PX_AB_BDEC    = hex2dec('00100000'); %//32'h0010_0000 ~ 32'h001F_FFFF
% BUF_WT_AB_BDEC    = hex2dec('00000000'); %//32'h0000_0000 ~ 32'h000F_FFFF


%%

CR_CONV_num_toy = ceil(NOY_CONV./Toy_CONV);
CR_CONV_num_tof = ceil(NOF_CONV./Tof_CONV);
CONV_NUM_Tiles_1LAYER = CR_CONV_num_toy.*CR_CONV_num_tof;

CR_PLMX_num_toy = ceil(NOY_PLMX./Toy_PLMX);
CR_PLMX_num_tof = ceil(NOF_PLMX./Tof_PLMX);
PLMX_NUM_Tiles_1LAYER = CR_PLMX_num_toy.*CR_PLMX_num_tof;

CR_NEAR_num_toy = ceil(NOY_NEAR./Toy_NEAR);
CR_NEAR_num_tof = ceil(NOF_NEAR./Tof_NEAR);
NEAR_NUM_Tiles_1LAYER = CR_NEAR_num_toy.*CR_NEAR_num_tof;

% NOTE: GAPL inpx RD are divided into multiple tiles, but outpx are written to DRAM all together. 
CR_GAPL_num_tif = ceil(NIF_GAPL0./Tif_GAPL);
GAPL_NUM_Tiles_1LAYER = CR_GAPL_num_tif;

% RSP: Deconv Updates
CR_DECV_num_toy = ceil(NOY_DECV./Toy_W_DECV);
CR_DECV_num_tof = ceil(NOF_DECV./Tof_DECV);
DECV_NUM_Tiles_1LAYER = CR_DECV_num_toy.*CR_DECV_num_tof;

CR_EWIS_num_toy = ceil(NOY_EWIS./Toy_EWIS);

CR_ROIP_num_tof = ceil(NOF_ROIP./Tof_ROIP);

for L = 1:NUM_CONV
   if  CR_CONV_num_tof(L) > 1 && CR_CONV_num_toy(L) > 1
       fprintf('Warning 1 @ DMA_dpt.m: layer %d Not minimum DRAM access!!! \n',L)
   end
end
fprintf('\n')

%Tiles_RDpx_pEW = CONV_NUM_Tiles_1LAYER; %(NOY_CONV./Toy_CONV).*(NOF_CONV./Tof_CONV);

NUM_dpt_WRpx_pCVpTil = Tof_CONV;
NUM_dpt_RDpx_pCVpTil = NIF_CONV;
NUM_dpt_RDwt_pCVpTil = ones(NUM_CONV,1);
NUM_dpt_RDpx_pEWpTil = Tof_CONV; % TODO tiles in one layer have the same number of dpt
NUM_dpt_RDpx_pPLpTil = Tif_PLMX; % tiles in one layer have the same number of dpt
NUM_dpt_WRpx_pPLpTil = Tof_PLMX;
NUM_dpt_RDpx_pNNpTil = Tif_NEAR; % tiles in one layer have the same number of dpt
NUM_dpt_WRpx_pNNpTil = Tof_NEAR;
%RSP: Deconv Updates
NUM_dpt_WRpx_pDECVpTil = Tof_DECV;
NUM_dpt_RDpx_pDECVpTil = NIF_DECV;
if NUM_DECV>0
    NUM_dpt_RDwt_pDECVpTil = ones(NUM_DECV,1);
else
    NUM_dpt_RDwt_pDECVpTil = [];
end
%

%% DDR3 Base Addresses of Weights, Images, Pixels

% JTAG_WDWTCV = floor(JTAG_WIDTH/(WD_WT*POF)); 20180107
JTAG_WDWTCV_abs = JTAG_WIDTH/(WD_WT*POF);
if JTAG_WDWTCV_abs >= 1
    JTAG_WDWTCV = floor(JTAG_WDWTCV_abs); 
else
    JTAG_WDWTCV = 1/floor(1/JTAG_WDWTCV_abs);
end

% JTAG_WDWTFC = floor((WD_WT*POF_FCON)/JTAG_WIDTH);
JTAG_WDWTFC_abs = floor((WD_WT*POF_FCON)/JTAG_WIDTH); %NUM_JTAG_MIF_CV=60, NUM_JTAG_MIF_FC=64,NUM_JTAG_MIF_IMAGE=4
if JTAG_WDWTFC_abs >= 1
    JTAG_WDWTFC = floor(JTAG_WDWTFC_abs); 
else
    JTAG_WDWTFC = 1/floor(1/JTAG_WDWTFC_abs);
end

Depth_KN_cv = sum(ceil(NKI_CONV.*NKX_CONV.*NKY_CONV.*NOF_CONV/POF));

%Depth_KN_fc = sum(NIF_FC.*((NOF_FCON+24)./POF_FCON)); % add zero pads 
Depth_KN_pFC = zeros(NUM_FCON,1);
for L = 1 : NUM_FCON
    if  mod(log2(NOF_FCON(L)),1) == 0 % Depth to be power of 2
        Depth_KN_pFC(L) = NIF_FCON(L)*(NOF_FCON(L)/POF_FCON);  
    else
        Depth_KN_pFC(L) = NIF_FCON(L)*(2^ceil(log2(NOF_FCON(L)))/POF_FCON);  
    end
end
Depth_KN_fc = sum (Depth_KN_pFC);

% RSP: Deconv Updates
JTAG_WDWTDECV_abs = JTAG_WIDTH/(WD_WT*POF);
if JTAG_WDWTDECV_abs >= 1
    JTAG_WDWTDECV = floor(JTAG_WDWTDECV_abs); 
else
    JTAG_WDWTDECV = 1/floor(1/JTAG_WDWTDECV_abs);
end

Depth_KN_Decv = sum(ceil(NKI_DECV.*NKX_DECV.*NKY_DECV.*NOF_DECV/POF));
%

bits_PROP_outputs = 4096*128;  % MAX(ROI) = 4096

Depth_CVKN_RAM = ceil(Depth_KN_cv/JTAG_WDWTCV);
Depth_FCKN_RAM = Depth_KN_fc*JTAG_WDWTFC;
Depth_PROP_RAM = ceil(bits_PROP_outputs/JTAG_WIDTH); % store the anchors of proposal layer
Depth_DECVKN_RAM = ceil(Depth_KN_Decv/JTAG_WDWTDECV);

NUM_JTAG_MIF_CV = ceil(Depth_CVKN_RAM/JTAG_WORDS);
NUM_JTAG_MIF_FC = ceil(Depth_FCKN_RAM/JTAG_WORDS)+1;
NUM_JTAG_MIF_PR = ceil(Depth_PROP_RAM/JTAG_WORDS);
NUM_JTAG_MIF_DECV = ceil(Depth_DECVKN_RAM/JTAG_WORDS);

Bytes_WT_CONV = JTAG_WORDS*JTAG_WIDTH*NUM_JTAG_MIF_CV/8;
Bytes_WT_FCON = JTAG_WORDS*JTAG_WIDTH*NUM_JTAG_MIF_FC/8;
Bytes_PROP    = JTAG_WORDS*JTAG_WIDTH*NUM_JTAG_MIF_PR/8;
Bytes_WT_DECV = JTAG_WORDS*JTAG_WIDTH*NUM_JTAG_MIF_DECV/8;

Bytes_image = ceil((ceil(NIF_CONV(1))*NIX_CONV(1)*NIY_CONV(1)*WD_PX*FCT_DMA)/(JTAG_WIDTH*JTAG_WORDS))*(JTAG_WIDTH*JTAG_WORDS)/8;

% RSP: DDR3_BDEC_WT_FCON = DDR3_BDEC_WT_CONV + Bytes_WT_CONV;
% RSP: DDR3_BDEC_PROP    = DDR3_BDEC_WT_FCON + Bytes_WT_FCON;
% RSP: DDR3_BDEC_IMAGE   = DDR3_BDEC_PROP    + Bytes_PROP;
DDR3_BDEC_WT_DECV = DDR3_BDEC_WT_CONV + Bytes_WT_CONV;
DDR3_BDEC_WT_FCON = DDR3_BDEC_WT_DECV + Bytes_WT_DECV;
DDR3_BDEC_PROP    = DDR3_BDEC_WT_FCON + Bytes_WT_FCON;

%DDR3_BDEC_IMAGE   = DDR3_BDEC_PROP    + Bytes_PROP;
DDR3_BDEC_IMAGE   = DDR3_BDEC_IMAGE_BASE;

NUM_IMAGES = 2;
CR_LAYER_outpx_addr = zeros(NUM_LAYER,1); % Base (start) address of outputs of each LAYER in DDR
%CR_LAYER_outpx_addr(1) = DDR3_BDEC_IMAGE + NUM_IMAGES*Bytes_image;

CR_LAYER_outpx_addr(1) = DDR3_BDEC_LAYER_BASE;

cnt_CONV = 1; cnt_DECV = 1; cnt_NEAR = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1;
for L = 2 : NUM_LAYER % NO padding zeros stored in DDR!!
    if CR_LAYER_IS_CONV(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_CONV(cnt_CONV)*NOX_CONV(cnt_CONV)*NOY_CONV_WRpx(cnt_CONV))*(WD_PX/8)*FCT_DMA;
        cnt_CONV = cnt_CONV + 1;
    end
    if CR_LAYER_IS_DECV(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_DECV(cnt_DECV)*NOX_DECV(cnt_DECV)*NOY_DECV(cnt_DECV)     )*(WD_PX/8)*FCT_DMA; % TODO TODO
        cnt_DECV = cnt_DECV + 1;
    end
    if CR_LAYER_IS_NEAR(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_NEAR(cnt_NEAR)*NOX_NEAR(cnt_NEAR)*NOY_NEAR_WRpx(cnt_NEAR))*(WD_PX/8)*FCT_DMA;
        cnt_NEAR = cnt_NEAR + 1;
    end
    if CR_LAYER_IS_PLMX(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_PLMX(cnt_PLMX)*NOX_PLMX(cnt_PLMX)*NOY_PLMX_WRpx(cnt_PLMX))*(WD_PX/8)*FCT_DMA;
        cnt_PLMX = cnt_PLMX + 1;
    end
    if CR_LAYER_IS_GAPL(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + NOF_GAPL(cnt_GAPL)*(WD_PX/8);
        cnt_GAPL = cnt_GAPL + 1;
    end
    if CR_LAYER_IS_ROIP(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOX_ROIP(cnt_ROIP)*NOY_ROIP(cnt_ROIP)*NOF_ROIP(cnt_ROIP)*NBX_ROIP(cnt_ROIP))*(WD_PX/8);
        cnt_ROIP = cnt_ROIP + 1;
    end
    if CR_LAYER_IS_PROP(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) +  bits_PROP_outputs/8;
        cnt_PROP = cnt_PROP + 1;
    end
    if CR_LAYER_IS_EWIS(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_EWIS(cnt_EWIS)*NOX_EWIS(cnt_EWIS)*NOY_EWIS(cnt_EWIS))*(WD_PX/8)*FCT_DMA;
        cnt_EWIS = cnt_EWIS + 1;
    end
    if CR_LAYER_IS_FCON(L-1) == 1
        CR_LAYER_outpx_addr(L) = CR_LAYER_outpx_addr(L-1) + (NOF_FCON(cnt_FCON)*NBX_FCON(cnt_FCON))*(WD_PX/8);
        cnt_FCON = cnt_FCON + 1;
    end            
end

DDR3_BYTE_Last_LAYER = 64*1024;
for L = NUM_LAYER % NO padding zeros stored in DDR!!
    if CR_LAYER_IS_CONV(L) == 1
        DDR3_BYTE_Last_LAYER = (NOF_CONV(cnt_CONV)*NOX_CONV(cnt_CONV)*NOY_CONV_WRpx(cnt_CONV))*(WD_PX/8)*FCT_DMA;
    end
    if CR_LAYER_IS_DECV(L) == 1
        DDR3_BYTE_Last_LAYER = (NOF_DECV(cnt_DECV)*NOX_DECV(cnt_DECV)*NOY_DECV(cnt_DECV)     )*(WD_PX/8)*FCT_DMA; % TODO TODO
    end    
    if CR_LAYER_IS_PLMX(L) == 1
        DDR3_BYTE_Last_LAYER = (NOF_PLMX(cnt_PLMX)*NOX_PLMX(cnt_PLMX)*NOY_PLMX_WRpx(cnt_PLMX))*(WD_PX/8)*FCT_DMA;
    end
    if CR_LAYER_IS_GAPL(L) == 1
        DDR3_BYTE_Last_LAYER = NOF_GAPL(cnt_GAPL)*(WD_PX/8);
    end
    if CR_LAYER_IS_ROIP(L) == 1
        DDR3_BYTE_Last_LAYER = (NOX_ROIP(cnt_ROIP)*NOY_ROIP(cnt_ROIP)*NOF_ROIP(cnt_ROIP)*NBX_ROIP(cnt_ROIP))*(WD_PX/8);
    end
    if CR_LAYER_IS_PROP(L) == 1
        DDR3_BYTE_Last_LAYER = bits_PROP_outputs/8;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        DDR3_BYTE_Last_LAYER = (NOF_EWIS(cnt_EWIS)*NOX_EWIS(cnt_EWIS)*NOY_EWIS(cnt_EWIS))*(WD_PX/8)*FCT_DMA;
    end
    if CR_LAYER_IS_FCON(L) == 1
        DDR3_BYTE_Last_LAYER = (NOF_FCON(cnt_FCON)*NBX_FCON(cnt_FCON))*(WD_PX/8);
    end            
end

DDR3_BYTE_INWT  = Bytes_WT_CONV + Bytes_WT_DECV + Bytes_WT_FCON + 1024*64;
%DDR3_SIZE_IMAGE = Bytes_image*NUM_IMAGES;
DDR3_BYTE_OUTPX = (max(CR_LAYER_outpx_addr) + DDR3_BYTE_Last_LAYER) - DDR3_BDEC_LAYER_BASE + 1024*64; % add 64 KByte margin 



%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% size of one output feature map stored in DRAM
Bytes_1OutMap_DATA = NIX_CONV(1).*NIY_CONV0(1)   .*(WD_PX/8)*FCT_DMA; % NOT include zero pads
Bytes_1OutMap_CONV = Tox_CONV   .*NOY_CONV_WRpx  .*(WD_PX/8)*FCT_DMA; % NOT include zero pads
Bytes_1OutMap_DECV = Tox_DECV   .*NOY_W_DECV_WRpx.*(WD_PX/8)*FCT_DMA; % NOT include zero pads
Bytes_1OutMap_NEAR = NOX_NEAR   .*NOY_NEAR_WRpx  .*(WD_PX/8)*FCT_DMA;
Bytes_1OutMap_PLMX = Tox_PLMX   .*NOY_PLMX_WRpx  .*(WD_PX/8)*FCT_DMA; % NOT include zero pads
Bytes_1OutMap_GAPL = NOF_GAPL                    .*(WD_PX/8); % GAPL outputs are continuously stored in buffer and DRAM
Bytes_1OutMap_ROIP = 32*16                       .*(WD_PX/8)*FCT_DMA; % TODO
Bytes_1OutMap_EWIS = NOX_EWIS   .*NOY_EWIS       .*(WD_PX/8)*FCT_DMA;

Bytes_1OutMap_LAYER = zeros(NUM_LAYER,1); % size of one output feature map stored in DRAM
cnt_CONV = 1; cnt_DECV = 1; cnt_NEAR = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1;
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_CONV(cnt_CONV);
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_DECV(cnt_DECV);
        cnt_DECV = cnt_DECV+1;
    end    
    if CR_LAYER_IS_NEAR(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_NEAR(cnt_NEAR);
        cnt_NEAR = cnt_NEAR+1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_PLMX(cnt_PLMX);
        cnt_PLMX = cnt_PLMX+1;
    end
    if CR_LAYER_IS_GAPL(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_GAPL(cnt_GAPL);
        cnt_GAPL = cnt_GAPL+1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        % TODO
        cnt_ROIP = cnt_ROIP+1;
    end    
    if CR_LAYER_IS_EWIS(L) == 1
        Bytes_1OutMap_LAYER(L) = Bytes_1OutMap_EWIS(cnt_EWIS);
        cnt_EWIS = cnt_EWIS+1;
    end
end


% DDR address offset of DMA read
%(Tiy_CONV_DMA-NKY_CONV+STR_CONV) includes the overlap by kernel window
Bytes_Tiy_offset_CONV = NIX_CONV.*(Tiy_CONV_DMA-NKY_CONV+STR_CONV)*(WD_PX/8)*FCT_DMA; % Tiy_CONV_DMA does not include PAD_CONV
% RSP: Deconv update: Bytes_Tiy_offset_DECV = NIX_DECV.*(Tiy_DECV_DMA-NKY_DECV+   STD_D)*(WD_PX/8)*FCT_DMA; % TODO TODO TODO
Bytes_Tiy_offset_DECV = NIX_DECV.*(Tiy_R_DECV_DMA-NKY_DECV+ STR_D)*(WD_PX/8)*FCT_DMA; % 
Bytes_Tiy_offset_NEAR = NIX_NEAR.*(Tiy_NEAR                      )*(WD_PX/8)*FCT_DMA;
Bytes_Tiy_offset_PLMX = NIX_PLMX.*(Tiy_PLMX    -NKY_PLMX+STR_PLMX)*(WD_PX/8)*FCT_DMA; % Tiy_CONV NOT include kernel window overlap
Bytes_Tiy_offset_EWIS = NIX_EWIS.*(Tiy_EWIS                      )*(WD_PX/8)*FCT_DMA;

DDR3_offset_RDPX_pCVpM = zeros(NUM_CONV,16);
for C = 1:NUM_CONV
    ID = ID_global_CONV(C);
    for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            DDR3_offset_RDPX_pCVpM(C,ii) = Bytes_1OutMap_DATA;
        else
            DDR3_offset_RDPX_pCVpM(C,ii) = Bytes_1OutMap_LAYER(input_ID);
        end
    end
end

DDR3_offset_RDPX_pPLpM = zeros(NUM_PLMX,16);
for C = 1:NUM_PLMX
    ID = ID_global_PLMX(C);
    for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            DDR3_offset_RDPX_pPLpM(C,ii) = Bytes_1OutMap_DATA;
        else
            DDR3_offset_RDPX_pPLpM(C,ii) = Bytes_1OutMap_LAYER(input_ID);
        end
    end
end

DDR3_offset_RDPX_pDECVpM = zeros(NUM_DECV,16);
for C = 1:NUM_DECV
    ID = ID_global_DECV(C);
    for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            DDR3_offset_RDPX_pDECVpM(C,ii) = Bytes_1OutMap_DATA;
        else
            DDR3_offset_RDPX_pDECVpM(C,ii) = Bytes_1OutMap_LAYER(input_ID);
        end
    end
end

DDR3_offset_RDPX_pNNpM = zeros(NUM_NEAR,16);
for C = 1:NUM_NEAR
    ID = ID_global_NEAR(C);
    for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            DDR3_offset_RDPX_pNNpM(C,ii) = Bytes_1OutMap_DATA;
        else
            DDR3_offset_RDPX_pNNpM(C,ii) = Bytes_1OutMap_LAYER(input_ID);
        end
    end
end


%% Compute Descriptors 

% Be careful of CONV 64
% first read pixels from DRAM

CR_LAYER_inpx_num_layer = zeros(NUM_LAYER,1);
for L = 1:NUM_LAYER
    CR_LAYER_inpx_num_layer(L) = length(input_layers_ID{L});
end

% If input is CONCAT, then offset of dpt are different for different layers

%% CONV

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_CONV_inpx_cmd_size = NIX_CONV0.*Tiy_CONV_DMA*(WD_PX/8)*FCT_DMA; % 32-bit, csr_inpx_cmd_size, csr_W_LEN_RDWT_CV
%CR_CONV_inpx_cmd_size = ceil(CR_CONV_inpx_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDpx_pTil_Bytes_CV

%R_CONV_RD_PX_NUM_dpt = NUM_dpt_RDpx_pCVpTil;
CR_CONV_offset_tiy = Bytes_Tiy_offset_CONV; % offset of different tiles
CR_CONV_RD_PX_offset_dpts = DDR3_offset_RDPX_pCVpM; % offset of different descriptors in one tile

CR_CONV_RD_PX_LayerBase  = zeros(NUM_CONV,16); % the MAX number of input layers is 16
CR_CONV_inpx_addr_adjust = zeros(NUM_CONV,1);
for C = 1:NUM_CONV
    ID = ID_global_CONV(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            base_LAYER = DDR3_BDEC_IMAGE;
        else
            base_LAYER = CR_LAYER_outpx_addr(input_ID);
        end
        if NOY_CONV(C) == Toy_CONV(C) % Tiles_RDpx_pCV(L) == 1  
            CR_CONV_inpx_addr_adjust(C) = 0; % DMA does NOT read padding
        else
            CR_CONV_inpx_addr_adjust(C) = NIX_CONV(C)*PAD_CONV(C)*(WD_PX/8)*FCT_DMA; % DMA reads padding from DDR
        end           
        CR_CONV_RD_PX_LayerBase(C,ii) = base_LAYER - CR_CONV_inpx_addr_adjust(C);       
    end
end

CR_CONV_RD_PX_NUM_dpt_perInputLayer = zeros(NUM_CONV,16); % one dpt reads one input feature map
for C = 1:NUM_CONV
    ID = ID_global_CONV(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            CR_CONV_RD_PX_NUM_dpt_perInputLayer(C,ii) = NIF_CONV(ID);
        else
            CR_CONV_RD_PX_NUM_dpt_perInputLayer(C,ii) = NOF_LAYER0(input_ID);
        end
    end
end
for C = 1:NUM_CONV
    if sum(CR_CONV_RD_PX_NUM_dpt_perInputLayer(C,:)) ~= NIF_CONV(C)
        fprintf('Warninig: CR_CONV_RD_PX_NUM_dpt_perInputLayer(%d) may not be correct\n\n',C)
    end
end


% RTL module of dma_control needs to compute RTL_* online
% RTL_* are only used to manually check RTL generated dpt 
% RTL_CONV_RD_PX_TileBase(per tile, per InputLayer, per CONV)
RTL_CONV_RD_PX_TileBase = zeros(max(CONV_NUM_Tiles_1LAYER),16,NUM_CONV); % the MAX number of input layers is 16
for C = 1:NUM_CONV % = cnt_CONV in RTL
    ID = ID_global_CONV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_CONV_num_tof(C)
    for TY = 1: CR_CONV_num_toy(C)
        TT = TY+(TF-1)*CR_CONV_num_toy(C); % iterate all tiles in one CONV
        for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            if CR_CONV_num_toy(C) == 1 && TF > 1
                RTL_CONV_RD_PX_TileBase(TT,ii,C) = 1;  % This tile has no RD_PX dpt and only has RD_WT
            else
                RTL_CONV_RD_PX_TileBase(TT,ii,C) = CR_CONV_RD_PX_LayerBase(C,ii) + (TY-1)*CR_CONV_offset_tiy(C);
            end
        end
    end
    end
end

% RTL_CONV_RD_PX_dpt_addr(per tile, per dpt, per CONV), descriptors of reading CONV inputs
RTL_CONV_RD_PX_dpt_addr = zeros(max(CONV_NUM_Tiles_1LAYER),max(NUM_dpt_RDpx_pCVpTil),NUM_CONV); % = DDR3_BDEC_RDPX_pTLpDpCV
for C = 1:NUM_CONV % = cnt_CONV in RTL
    ID = ID_global_CONV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_CONV_num_tof(C)
    for TY = 1: CR_CONV_num_toy(C)
        TT = TY+(TF-1)*CR_CONV_num_toy(C); % iterate all tiles in one CONV
        cnt_dpt = 1;
        for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            if CR_CONV_num_toy(C) == 1 && TF > 1
                RTL_CONV_RD_PX_dpt_addr(TT,1,C) = 1;  % This tile has no RD_PX dpt and only has RD_WT
            else
                for DD = 1:CR_CONV_RD_PX_NUM_dpt_perInputLayer(C,ii)
                    RTL_CONV_RD_PX_dpt_addr(TT,cnt_dpt,C) = RTL_CONV_RD_PX_TileBase(TT,ii,C)+(DD-1)*CR_CONV_RD_PX_offset_dpts(C,ii);
                    cnt_dpt = cnt_dpt+1;
                end
            end
        end
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_CONV_inwt_cmd_size = (NKX_CONV.*NKY_CONV.*NKI_CONV.*Tof_CONV)*(WD_WT/8); % 32-bit, csr_inwt_cmd_size, csr_W_LEN_RDWT_CV
%CR_CONV_inwt_cmd_size = ceil(CR_CONV_inwt_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDwt_pTil_Bytes_CV

INwt_pFCON_Bytes          = (NKI_CONV.*NKX_CONV.*NKY_CONV.*NOF_CONV)*(WD_WT/8);
INwt_pFCON_Bytes          = ceil(INwt_pFCON_Bytes*8/DMA_WIDTH)*DMA_WIDTH/8;
CR_CONV_inwt_addr = zeros(NUM_CONV,1); % weight base address of each layer
CR_CONV_inwt_addr(1) = DDR3_BDEC_WT_CONV;
for L=2:NUM_CONV
       CR_CONV_inwt_addr(L) = CR_CONV_inwt_addr(L-1) + INwt_pFCON_Bytes(L-1);
end

CR_CONV_offset_wt_tof = CR_CONV_inwt_cmd_size; % offset of different tiles

% # of dpts of weights per tile is always 1 

% RTL module of dma_control needs to compute RTL_* online
RTL_CONV_RD_WT_TileBase = zeros(NUM_CONV,max(CONV_NUM_Tiles_1LAYER)); % = DDR3_BDEC_RDWT_pTLpCV
for C = 1:NUM_CONV
    for TF = 1: CR_CONV_num_tof(C)
    for TY = 1: CR_CONV_num_toy(C)
        TT = TY+(TF-1)*CR_CONV_num_toy(C); % iterate all tiles in one CONV
        if TY == 1
            RTL_CONV_RD_WT_TileBase(C,TT) = CR_CONV_inwt_addr(C) + (TF-1)*CR_CONV_offset_wt_tof(C);
        else
            RTL_CONV_RD_WT_TileBase(C,TT) = 1; % This tile has no RD_WT dpt
        end
    end
    end
end

% RTL_CONV_RD_WT_dpt(per CONV, per DPT)
RTL_CONV_RD_WT_dpt_addr = RTL_CONV_RD_WT_TileBase;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
% JH: Use actual Toy_CONV0 on write descriptor to avoid writing garbage
% JH: Still use POY padded R_WORDS_WRPX_CONV_M1 to make sure address jumping inside output buffer is correct
len_WRpx_pTil_Bytes_CONV_raw = NOX_CONV0.*Toy_CONV0*(WD_PX/8)*FCT_DMA;
len_WRpx_pTil_Bytes_CONV     = Tox_CONV.*Toy_CONV *(WD_PX/8)*FCT_DMA;
%len_WRpx_pTil_Bytes_CONV_raw = ceil(len_WRpx_pTil_Bytes_CONV_raw*8/DMA_WIDTH)*DMA_WIDTH/8;
len_WRpx_pTil_Bytes_CONV     = ceil(len_WRpx_pTil_Bytes_CONV    *8/DMA_WIDTH)*DMA_WIDTH/8;
CR_CONV_outpx_cmd_size = len_WRpx_pTil_Bytes_CONV_raw; % 24-bit, csr_W_LEN_WRPX_CV, TODO, check

CR_CONV_NUM_dpt_WRpx_pCVpTil = NUM_dpt_WRpx_pCVpTil;
CR_CONV_offset_toy = len_WRpx_pTil_Bytes_CONV; % for Tiles_RDwt_pCV = 1
CR_CONV_offset_tof = NUM_dpt_WRpx_pCVpTil.*Bytes_1OutMap_CONV; % for Tiles_RDwt_pCV > 1
CR_CONV_offset_of  = Bytes_1OutMap_CONV; % offset of different dpt in one tile

% RTL module of DMA_control needs to compute base_WRpx_CONV_tile online
% DDR3_BDEC_CONV can be derived from R_LAYER_base_WRpx
RTL_CONV_WR_PX_TileBase = zeros(NUM_CONV,max(CONV_NUM_Tiles_1LAYER));
for C = 1:NUM_CONV
    ID = ID_global_CONV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_CONV_num_tof(C)
    for TY = 1: CR_CONV_num_toy(C)
        TT = TY+(TF-1)*CR_CONV_num_toy(C); % iterate all tiles in one CONV
        RTL_CONV_WR_PX_TileBase(C,TT) = CR_LAYER_outpx_addr(ID) + (TY-1)*CR_CONV_offset_toy(C) + (TF-1)*CR_CONV_offset_tof(C);
    end
    end
end

% RTL_CONV_WR_PX_dpt_addr(per tile, per dpt, per CONV), descriptors of reading CONV inputs
RTL_CONV_WR_PX_dpt_addr = zeros(max(CONV_NUM_Tiles_1LAYER),max(NUM_dpt_WRpx_pCVpTil),NUM_CONV); % = DDR3_BDEC_WRPX_pTLpDpCV
for C = 1:NUM_CONV % = cnt_CONV in RTL
    for TF = 1: CR_CONV_num_tof(C)
    for TY = 1: CR_CONV_num_toy(C)
        TT = TY+(TF-1)*CR_CONV_num_toy(C); % iterate all tiles in one CONV
        for DD = 1: CR_CONV_NUM_dpt_WRpx_pCVpTil(C)
            RTL_CONV_WR_PX_dpt_addr(TT,DD,C) = RTL_CONV_WR_PX_TileBase(C,TT)+(DD-1)*CR_CONV_offset_of(C);
        end
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% DECONV

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECONV read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
% RSP: CR_DECV_inpx_cmd_size = NIX_DECV.*Tiy_DECV_DMA*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_RDWT_CV
CR_DECV_inpx_cmd_size = NIX_R_DECV.*Tiy_R_DECV_DMA*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_RDWT_CV
CR_DECV_inpx_cmd_size = ceil(CR_DECV_inpx_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDpx_pTil_Bytes_CV

%R_DECV_RD_PX_NUM_dpt = NUM_dpt_RDpx_pCVpTil;
CR_DECV_offset_tiy = Bytes_Tiy_offset_DECV; % offset of different tiles
CR_DECV_RD_PX_offset_dpts = DDR3_offset_RDPX_pDECVpM; % offset of different descriptors in one tile

CR_DECV_RD_PX_LayerBase  = zeros(NUM_DECV,16); % the MAX number of input layers is 16
CR_DECV_inpx_addr_adjust = zeros(NUM_DECV,1);
for C = 1:NUM_DECV
    ID = ID_global_DECV(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            base_LAYER = DDR3_BDEC_IMAGE;
        else
            base_LAYER = CR_LAYER_outpx_addr(input_ID);
        end
        if NOY_DECV(C) == Toy_W_DECV(C) % Tiles_RDpx_pCV(L) == 1  
            CR_DECV_inpx_addr_adjust(C) = 0; % DMA does NOT read padding
        else
            CR_DECV_inpx_addr_adjust(C) = NIX_R_DECV(C)*PAD_D(C)*(WD_PX/8)*FCT_DMA; % DMA reads padding from DDR
        end           
        CR_DECV_RD_PX_LayerBase(C,ii) = base_LAYER - CR_DECV_inpx_addr_adjust(C);
    end
end

CR_DECV_RD_PX_NUM_dpt_perInputLayer = zeros(NUM_DECV,16); % one dpt reads one input feature map
for C = 1:NUM_DECV
    ID = ID_global_DECV(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            CR_DECV_RD_PX_NUM_dpt_perInputLayer(C,ii) = NIF_DECV(ID);
        else
            CR_DECV_RD_PX_NUM_dpt_perInputLayer(C,ii) = NOF_LAYER0(input_ID);
        end
    end
end
for C = 1:NUM_DECV
    if sum(CR_DECV_RD_PX_NUM_dpt_perInputLayer(C,:)) ~= NIF_DECV(C)
        fprintf('Warninig: CR_DECV_RD_PX_NUM_dpt_perInputLayer(%d) may not be correct\n\n',C)
    end
end


% RTL module of dma_control needs to compute RTL_* online
% RTL_DECV_RD_PX_TileBase(per tile, per InputLayer, per DECV)
RTL_DECV_RD_PX_TileBase = zeros(max(DECV_NUM_Tiles_1LAYER),16,NUM_DECV); % the MAX number of input layers is 16
for C = 1:NUM_DECV % = cnt_DECV in RTL
    ID = ID_global_DECV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_DECV_num_tof(C)
    for TY = 1: CR_DECV_num_toy(C)
        TT = TY+(TF-1)*CR_DECV_num_toy(C); % iterate all tiles in one DECV
        for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            if CR_DECV_num_toy(C) == 1 && TF > 1
                RTL_DECV_RD_PX_TileBase(TT,ii,C) = 1;  % This tile has no RD_PX dpt and only has RD_WT
            else
                RTL_DECV_RD_PX_TileBase(TT,ii,C) = CR_DECV_RD_PX_LayerBase(C,ii) + (TY-1)*CR_DECV_offset_tiy(C);
            end
        end
    end
    end
end

% RTL_DECV_RD_PX_dpt_addr(per tile, per dpt, per DECV), descriptors of reading DECV inputs
RTL_DECV_RD_PX_dpt_addr = zeros(max(DECV_NUM_Tiles_1LAYER),max(NUM_dpt_RDpx_pDECVpTil),NUM_DECV); % = DDR3_BDEC_RDPX_pTLpDpCV
for C = 1:NUM_DECV % = cnt_DECV in RTL
    ID = ID_global_DECV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_DECV_num_tof(C)
    for TY = 1: CR_DECV_num_toy(C)
        TT = TY+(TF-1)*CR_DECV_num_toy(C); % iterate all tiles in one DECV
        cnt_dpt = 1;
        for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            if CR_DECV_num_toy(C) == 1 && TF > 1
                RTL_DECV_RD_PX_dpt_addr(TT,1,C) = 1;  % This tile has no RD_PX dpt and only has RD_WT
            else
                for DD = 1:CR_DECV_RD_PX_NUM_dpt_perInputLayer(C,ii)
                    RTL_DECV_RD_PX_dpt_addr(TT,cnt_dpt,C) = RTL_DECV_RD_PX_TileBase(TT,ii,C)+(DD-1)*CR_DECV_RD_PX_offset_dpts(C,ii);
                    cnt_dpt = cnt_dpt+1;
                end
            end
        end
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_DECV_inwt_cmd_size = (NKX_DECV.*NKY_DECV.*NKI_DECV.*Tof_DECV)*(WD_WT/8); % 24-bit, csr_W_LEN_RDWT_CV
CR_DECV_inwt_cmd_size = ceil(CR_DECV_inwt_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDwt_pTil_Bytes_CV

INwt_pFCON_Bytes          = (NKI_DECV.*NKX_DECV.*NKY_DECV.*NOF_DECV)*(WD_WT/8);
INwt_pFCON_Bytes          = ceil(INwt_pFCON_Bytes*8/DMA_WIDTH)*DMA_WIDTH/8;
CR_DECV_inwt_addr = zeros(NUM_DECV,1); % base address of each layer
CR_DECV_inwt_addr(1) = DDR3_BDEC_WT_DECV;
for L=2:NUM_DECV
       CR_DECV_inwt_addr(L) = CR_DECV_inwt_addr(L-1) + INwt_pFCON_Bytes(L-1);
end

CR_DECV_offset_wt_tof = CR_DECV_inwt_cmd_size; % offset of different tiles

% # of dpts of weights per tile is always 1 

% RTL module of dma_control needs to compute RTL_* online
RTL_DECV_RD_WT_TileBase = zeros(NUM_DECV,max(DECV_NUM_Tiles_1LAYER)); % = DDR3_BDEC_RDWT_pTLpCV
for C = 1:NUM_DECV
    for TF = 1: CR_DECV_num_tof(C)
    for TY = 1: CR_DECV_num_toy(C)
        TT = TY+(TF-1)*CR_DECV_num_toy(C); % iterate all tiles in one DECV
        if TY == 1
            RTL_DECV_RD_WT_TileBase(C,TT) = CR_DECV_inwt_addr(C) + (TF-1)*CR_DECV_offset_wt_tof(C);
        else
            RTL_DECV_RD_WT_TileBase(C,TT) = 1; % This tile has no RD_WT dpt
        end
    end
    end
end

% RTL_DECV_RD_WT_dpt(per DECV, per DPT)
RTL_DECV_RD_WT_dpt_addr = RTL_DECV_RD_WT_TileBase;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
% JH: Use actual Toy_DECV0 on write descriptor to avoid writing garbage
% JH: Still use POY padded R_WORDS_WRPX_DECV_M1 to make sure address jumping inside output buffer is correct
% RSP: Deconv Updates: len_WRpx_pTil_Bytes_DECV_raw = Tox_DECV.*Toy_DECV0*(WD_PX/8)*FCT_DMA;
% RSP: Deconv Updates: len_WRpx_pTil_Bytes_DECV     = Tox_DECV.*Toy_DECV *(WD_PX/8)*FCT_DMA;
% RSP: Deconv Updates: len_WRpx_pTil_Bytes_DECV_raw = ceil(len_WRpx_pTil_Bytes_DECV_raw*8/DMA_WIDTH)*DMA_WIDTH/8;
% RSP: Deconv Updates: len_WRpx_pTil_Bytes_DECV     = ceil(len_WRpx_pTil_Bytes_DECV*8/DMA_WIDTH)*DMA_WIDTH/8;
% RSP: Deconv Updates: CR_DECV_outpx_cmd_size = len_WRpx_pTil_Bytes_DECV_raw; % 24-bit, csr_W_LEN_WRPX_DECV, TODO, check
len_WRpx_pTil_Bytes_DECV_raw = Tox_W_DECV.*Toy_W_DECV0*(WD_PX/8)*FCT_DMA;
len_WRpx_pTil_Bytes_DECV     = Tox_W_DECV.*Toy_W_DECV *(WD_PX/8)*FCT_DMA;
len_WRpx_pTil_Bytes_DECV_raw = ceil(len_WRpx_pTil_Bytes_DECV_raw*8/DMA_WIDTH)*DMA_WIDTH/8;
len_WRpx_pTil_Bytes_DECV     = ceil(len_WRpx_pTil_Bytes_DECV*8/DMA_WIDTH)*DMA_WIDTH/8;
CR_DECV_outpx_cmd_size = len_WRpx_pTil_Bytes_DECV_raw; % 24-bit, csr_W_LEN_WRPX_DECV, TODO, check

CR_DECV_NUM_dpt_WRpx_pCVpTil    = NUM_dpt_WRpx_pDECVpTil;
CR_DECV_offset_toy = len_WRpx_pTil_Bytes_DECV; % for Tiles_RDwt_pCV = 1
CR_DECV_offset_tof = NUM_dpt_WRpx_pDECVpTil.*Bytes_1OutMap_DECV; % for Tiles_RDwt_pCV > 1
CR_DECV_offset_of  = Bytes_1OutMap_DECV; % offset of different dpt in one tile

% RTL module of DMA_control needs to compute base_WRpx_DECV_tile online
% DDR3_BDEC_DECV can be derived from R_LAYER_base_WRpx
RTL_DECV_WR_PX_TileBase = zeros(NUM_DECV,max(DECV_NUM_Tiles_1LAYER));
for C = 1:NUM_DECV
    ID = ID_global_DECV(C); % = cnt_LAYER in RTL
    for TF = 1: CR_DECV_num_tof(C)
    for TY = 1: CR_DECV_num_toy(C)
        TT = TY+(TF-1)*CR_DECV_num_toy(C); % iterate all tiles in one DECV
        RTL_DECV_WR_PX_TileBase(C,TT) = CR_LAYER_outpx_addr(ID) + (TY-1)*CR_DECV_offset_toy(C) + (TF-1)*CR_DECV_offset_tof(C);
    end
    end
end

% RTL_DECV_WR_PX_dpt_addr(per tile, per dpt, per DECV), descriptors of reading DECV inputs
RTL_DECV_WR_PX_dpt_addr = zeros(max(DECV_NUM_Tiles_1LAYER),max(NUM_dpt_WRpx_pCVpTil),NUM_DECV); % = DDR3_BDEC_WRPX_pTLpDpCV
for C = 1:NUM_DECV % = cnt_DECV in RTL
    for TF = 1: CR_DECV_num_tof(C)
    for TY = 1: CR_DECV_num_toy(C)
        TT = TY+(TF-1)*CR_DECV_num_toy(C); % iterate all tiles in one DECV
        for DD = 1: CR_DECV_NUM_dpt_WRpx_pCVpTil(C)
            RTL_DECV_WR_PX_dpt_addr(TT,DD,C) = RTL_DECV_WR_PX_TileBase(C,TT)+(DD-1)*CR_DECV_offset_of(C);
        end
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% NEAR

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEAR read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_NEAR_inpx_cmd_size = Tix_NEAR.*Tiy_NEAR*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_RDPX_PL
CR_NEAR_inpx_cmd_size = ceil(CR_NEAR_inpx_cmd_size*8/DMA_WIDTH)*DMA_WIDTH/8; % len_RDpx_pTil_Bytes_NEAR

CR_NEAR_offset_tiy = Bytes_Tiy_offset_NEAR;
CR_NEAR_RD_PX_offset_dpts = DDR3_offset_RDPX_pNNpM; % offset of different descriptors in one tile

CR_NEAR_RD_PX_LayerBase  = zeros(NUM_NEAR,16); % the MAX number of input layers is 16
CR_NEAR_inpx_addr_adjust = zeros(NUM_NEAR,1);
for C = 1:NUM_NEAR
    ID = ID_global_NEAR(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            base_LAYER = DDR3_BDEC_IMAGE;
        else
            base_LAYER = CR_LAYER_outpx_addr(input_ID);
        end
        CR_NEAR_inpx_addr_adjust(C) = 0; % DMA does NOT read padding
        CR_NEAR_RD_PX_LayerBase(C,ii) = base_LAYER - CR_NEAR_inpx_addr_adjust(C);
    end
end

CR_NEAR_RD_PX_NUM_dpt_perInputLayer = zeros(NUM_NEAR,16); % one dpt reads one input feature map
for C = 1:NUM_NEAR
    ID = ID_global_NEAR(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            CR_NEAR_RD_PX_NUM_dpt_perInputLayer(C,ii) = NIF_NEAR(ID);
        else
            CR_NEAR_RD_PX_NUM_dpt_perInputLayer(C,ii) = NOF_LAYER0(input_ID);
        end
    end
end
for C = 1:NUM_NEAR
    if sum(CR_NEAR_RD_PX_NUM_dpt_perInputLayer(C,:)) ~= NIF_NEAR0(C)
        fprintf('Warninig: CR_NEAR_RD_PX_NUM_dpt_perInputLayer(%d) may not be correct\n\n',C)
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEAR read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEAR write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_NEAR_outpx_cmd_size = Tox_NEAR.*Toy_NEAR_WRpx*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_WRPX_PL
CR_NEAR_outpx_cmd_size = ceil(CR_NEAR_outpx_cmd_size*8/DMA_WIDTH)*DMA_WIDTH/8; % len_WRpx_pTil_Bytes_NEAR

CR_NEAR_NUM_dpt_WRpx_pNNpTil = NUM_dpt_WRpx_pNNpTil;
CR_NEAR_offset_toy = CR_NEAR_outpx_cmd_size; % offset of different tiles
CR_NEAR_offset_tof = NUM_dpt_WRpx_pNNpTil.*Bytes_1OutMap_NEAR; % offset of different tiles
CR_NEAR_offset_of  = Bytes_1OutMap_NEAR; % offset of different dpt in one tile

% RTL module of DMA_control needs to compute base_WRpx_NEAR_tile online
% DDR3_BDEC_NEAR can be derived from R_LAYER_base_WRpx
RTL_NEAR_WR_PX_TileBase = zeros(NUM_NEAR,max(NEAR_NUM_Tiles_1LAYER));
for L = 1:NUM_NEAR
    ID = ID_global_NEAR(C); % = cnt_LAYER in RTL
    for TF = 1: CR_NEAR_num_tof(L)
    for TY = 1: CR_NEAR_num_toy(L)
        T = TY+(TF-1)*CR_NEAR_num_toy(L);
        RTL_NEAR_WR_PX_TileBase(L,T) = CR_LAYER_outpx_addr(ID) + (TY-1)*CR_NEAR_offset_toy(L) + (TF-1)*CR_NEAR_offset_tof(L);
    end
    end
end

% RTL_NEAR_WR_PX_dpt_addr(per tile, per dpt, per CONV), descriptors of writing NEAR outputs to DDR
RTL_NEAR_WR_PX_dpt_addr = zeros(max(NEAR_NUM_Tiles_1LAYER),max(NUM_dpt_WRpx_pNNpTil),NUM_NEAR); % = DDR3_BDEC_WRPX_pTLpDpPL
for C = 1:NUM_NEAR % = cnt_CONV in RTL
    for TF = 1: CR_NEAR_num_tof(C)
    for TY = 1: CR_NEAR_num_toy(C)
        TT = TY+(TF-1)*CR_NEAR_num_toy(C); % iterate all tiles in one NEAR
        for DD = 1: CR_NEAR_NUM_dpt_WRpx_pNNpTil(C)
            RTL_NEAR_WR_PX_dpt_addr(TT,DD,C) = RTL_NEAR_WR_PX_TileBase(C,TT)+(DD-1)*CR_NEAR_offset_of(C);
        end
    end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEAR write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% PLMX

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLMX read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_PLMX_inpx_cmd_size = NIX_PLMX0.*Tiy_PLMX*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_RDPX_PL
%CR_PLMX_inpx_cmd_size = ceil(CR_PLMX_inpx_cmd_size*8/DMA_WIDTH)*DMA_WIDTH/8; % len_RDpx_pTil_Bytes_PLMX

CR_PLMX_offset_tiy = Bytes_Tiy_offset_PLMX;
CR_PLMX_RD_PX_offset_dpts = DDR3_offset_RDPX_pPLpM; % offset of different descriptors in one tile

CR_PLMX_RD_PX_LayerBase  = zeros(NUM_PLMX,16); % the MAX number of input layers is 16
CR_PLMX_inpx_addr_adjust = zeros(NUM_PLMX,1);
for C = 1:NUM_PLMX
    ID = ID_global_PLMX(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            base_LAYER = DDR3_BDEC_IMAGE;
        else
            base_LAYER = CR_LAYER_outpx_addr(input_ID);
        end
        if NOY_PLMX(C) == Toy_PLMX(C)
            CR_PLMX_inpx_addr_adjust(C) = 0; % DMA does NOT read padding
        else
            CR_PLMX_inpx_addr_adjust(C) = NIX_PLMX(C)*PAD_PLMX(C)*(WD_PX/8)*FCT_DMA; % DMA reads padding from DDR
        end         
        CR_PLMX_RD_PX_LayerBase(C,ii) = base_LAYER - CR_PLMX_inpx_addr_adjust(C);
    end
end

CR_PLMX_RD_PX_NUM_dpt_perInputLayer = zeros(NUM_PLMX,16); % one dpt reads one input feature map
for C = 1:NUM_PLMX
    ID = ID_global_PLMX(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            CR_PLMX_RD_PX_NUM_dpt_perInputLayer(C,ii) = NIF_PLMX(ID);
        else
            CR_PLMX_RD_PX_NUM_dpt_perInputLayer(C,ii) = NOF_LAYER0(input_ID);
        end
    end
end
for C = 1:NUM_PLMX
    if sum(CR_PLMX_RD_PX_NUM_dpt_perInputLayer(C,:)) ~= NIF_PLMX0(C)
        fprintf('Warninig: CR_PLMX_RD_PX_NUM_dpt_perInputLayer(%d) may not be correct\n\n',C)
    end
end


% RTL module of dma_control needs to compute RTL_* online 
RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer = 2047*ones(NUM_PLMX,16); % one dpt reads one input feature map
for C = 1:NUM_PLMX
    ID = ID_global_PLMX(C); % = cnt_LAYER in RTL
    RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,1) = 0;
    for ii = 2:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,ii) = RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,ii-1)+CR_PLMX_RD_PX_NUM_dpt_perInputLayer(C,ii-1);
    end
end

% RTL_PLMX_RD_PX_dpt_addr(per tile, per dpt, per CONV), descriptors of reading PLMX inputs
RTL_PLMX_RD_PX_dpt_addr = zeros(max(PLMX_NUM_Tiles_1LAYER),max(NUM_dpt_RDpx_pPLpTil),NUM_PLMX); % = DDR3_BDEC_RDPX_pTLpDpPL
for C = 1:NUM_PLMX % = cnt_PLMX in RTL
    ID = ID_global_PLMX(C); % = cnt_LAYER in RTL
        cnt_ilayer = 1; % count the input layers of one PLMX
        cnt_imap   = 1; % count the input maps in one input layer of one PLMX
        cnt_nif    = 1; % count the input maps of one PLMX
        for TF = 1: CR_PLMX_num_tof(C) % iterate tiles along Tif
        for TY = 1: CR_PLMX_num_toy(C) % iterate tiles along Tiy
            TT = TY+(TF-1)*CR_PLMX_num_toy(C); % iterate all tiles in one PLMX
            for DD = 1: Tif_PLMX(C) % iterate input maps in one tile
                
                RTL_PLMX_RD_PX_dpt_addr(TT,DD,C) = CR_PLMX_RD_PX_LayerBase(C,cnt_ilayer) + (TY-1)*CR_PLMX_offset_tiy(C) + (cnt_imap-1)*CR_PLMX_RD_PX_offset_dpts(C,cnt_ilayer);
                %fprintf('cnt_PLMX = %d, TF = %02d, TY = %d, DD = %02d, cnt_nif = %03d, cnt_ilayer = %d, cnt_imap = %d\n',C,TF,TY,DD,cnt_nif,cnt_ilayer,cnt_imap); % uncomment this to check RTL control logic
                
                if DD == Tif_PLMX(C) && TY ~= CR_PLMX_num_toy(C)
                    cnt_nif = 1+(TF-1)*Tif_PLMX(C);
                elseif DD == Tif_PLMX(C) && TY == CR_PLMX_num_toy(C)
                    cnt_nif = 1+TF*Tif_PLMX(C);
                else
                    cnt_nif = cnt_nif +1;
                end
                
                % compute cnt_imap and cnt_ilayer from cnt_nif
                for iii = 1:16 % support max 16 input layers
                    if cnt_nif > RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,iii) && cnt_nif <= RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,iii+1)
                        cnt_ilayer = iii;
                    end
                end
                cnt_imap = cnt_nif - RTL_PLMX_RD_PX_ACCUMULATED_dpt_perInputLayer(C,cnt_ilayer);
                
            end
        end
        %fprintf('\n');
        end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLMX read input pixels (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLMX write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transfer length (bytes) per transaction (descriptor) 
CR_PLMX_outpx_cmd_size = NOX_PLMX0.*Toy_PLMX_WRpx*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_WRPX_PL
%CR_PLMX_outpx_cmd_size = ceil(CR_PLMX_outpx_cmd_size*8/DMA_WIDTH)*DMA_WIDTH/8; % len_WRpx_pTil_Bytes_PLMX

CR_PLMX_NUM_dpt_WRpx_pPLpTil = NUM_dpt_WRpx_pPLpTil;
CR_PLMX_offset_toy = CR_PLMX_outpx_cmd_size; % offset of different tiles
CR_PLMX_offset_tof = NUM_dpt_WRpx_pPLpTil.*Bytes_1OutMap_PLMX; % offset of different tiles
CR_PLMX_offset_of  = Bytes_1OutMap_PLMX; % offset of different dpt in one tile

% RTL module of DMA_control needs to compute base_WRpx_PLMX_tile online
% DDR3_BDEC_PLMX can be derived from R_LAYER_base_WRpx
RTL_PLMX_WR_PX_TileBase = zeros(NUM_PLMX,max(PLMX_NUM_Tiles_1LAYER));
for L = 1:NUM_PLMX
    ID = ID_global_PLMX(C); % = cnt_LAYER in RTL
    for TF = 1: CR_PLMX_num_tof(L)
    for TY = 1: CR_PLMX_num_toy(L)
        T = TY+(TF-1)*CR_PLMX_num_toy(L);
        RTL_PLMX_WR_PX_TileBase(L,T) = CR_LAYER_outpx_addr(ID) + (TY-1)*CR_PLMX_offset_toy(L) + (TF-1)*CR_PLMX_offset_tof(L);
    end
    end
end

% RTL_PLMX_WR_PX_dpt_addr(per tile, per dpt, per CONV), descriptors of writing PLMX outputs to DDR
RTL_PLMX_WR_PX_dpt_addr = zeros(max(PLMX_NUM_Tiles_1LAYER),max(NUM_dpt_WRpx_pPLpTil),NUM_PLMX); % = DDR3_BDEC_WRPX_pTLpDpPL
for C = 1:NUM_PLMX % = cnt_CONV in RTL
    for TF = 1: CR_PLMX_num_tof(C)
    for TY = 1: CR_PLMX_num_toy(C)
        TT = TY+(TF-1)*CR_PLMX_num_toy(C); % iterate all tiles in one PLMX
        for DD = 1: CR_PLMX_NUM_dpt_WRpx_pPLpTil(C)
            RTL_PLMX_WR_PX_dpt_addr(TT,DD,C) = RTL_PLMX_WR_PX_TileBase(C,TT)+(DD-1)*CR_PLMX_offset_of(C);
        end
    end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLMX write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% PROP

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROP DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PROP read bbox  as inpx,  one command reads one feature map from DDR
% PROP read score as inwt,  one command reads one feature map from DDR
% PROP write ROIs as outpx, one command writes all ROIs to DDR

% transfer length (bytes) per transaction (descriptor) 
CR_PROP_inpx_cmd_size = NIX_PROP.*NIY_PROP0*(WD_PX/8)*FCT_DMA; % 32-bit, csr_inpx_cmd_size
CR_PROP_inpx_cmd_size = ceil(CR_PROP_inpx_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDpx_pTil_Bytes_CV

CR_PROP_inwt_cmd_size  = CR_PROP_inpx_cmd_size; % 32-bit, csr_inwt_cmd_size, cmd_size of score

CR_PROP_outpx_cmd_size = (NBX_ROIP0*DMA_WIDTH/2)/8; % one DMA address (256-bit) holds 2 ROIs

CR_PROP_inwt_addr  = zeros(NUM_PROP,1); % LAYER start address of score (conv)
for C = 1:NUM_PROP
    CR_PROP_inwt_addr(C)  = CR_LAYER_outpx_addr(ID_PROP_score(C));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROP DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% ROIP

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROIP DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read ROIs/boxes from PROP as inwt
CR_ROIP_inwt_addr = zeros(NUM_ROIP,1);
for C = 1:NUM_ROIP
    ROIP_ID = ID_global_ROIP(C);
    for ii = 1:length(input_layers_ID{ROIP_ID})
        input_ID = input_layers_ID{ROIP_ID}(ii);
        if CR_LAYER_IS_PROP(input_ID) == 1
            CR_ROIP_inwt_addr(C) = CR_LAYER_outpx_addr(input_ID);
        end
    end
end
% one cmd reads all the ROIs
CR_ROIP_inwt_cmd_size = NBX_ROIP0*(128/8); % 16*4+8

% one cmd reads one input feature map
CR_ROIP_inpx_cmd_size = NIX_ROIP.*NIY_ROIP0*(WD_PX/8)*FCT_DMA; % 32-bit, csr_inpx_cmd_size
CR_ROIP_inpx_cmd_size = ceil(CR_ROIP_inpx_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; % previous len_RDpx_pTil_Bytes_CV

% one cmd writes one tiled ROI/box to ddr
CR_ROIP_outpx_cmd_size = NOX_ROIP.*NOY_ROIP0*Tof_ROIP*(WD_PX/8);
if mod(CR_ROIP_outpx_cmd_size*8/JTAG_WIDTH,1) ~= 0
     fprintf('Warning @ DMA_dpt.m: CR_ROIP_outpx_cmd_size not aligned with DMA bitwidth, change Tof_ROIP!\n\n')
end

% offset between two adjacent tiles
CR_ROIP_offset_toy = NOX_ROIP.*NOY_ROIP0*Tof_ROIP*(WD_PX/8);

% offset between two adjacent ROIs/boxes
CR_ROIP_offset_tof = NOX_ROIP.*NOY_ROIP0*NOF_ROIP*(WD_PX/8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROIP DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% EWIS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EWIS DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% one cmd read/write one input/output feature map
% transfer length (bytes) per transaction (descriptor) 
CR_EWIS_inpx_cmd_size = NIX_EWIS0.*Tiy_EWIS*(WD_PX/8)*FCT_DMA;
%CR_EWIS_inpx_cmd_size = ceil(CR_EWIS_inpx_cmd_size*8/JTAG_WIDTH)*JTAG_WIDTH/8; 

CR_EWIS_outpx_cmd_size = CR_EWIS_inpx_cmd_size;

CR_EWIS_offset_tiy = Bytes_Tiy_offset_EWIS; % offset of different tiles
CR_EWIS_offset_toy = Bytes_Tiy_offset_EWIS; % offset of different tiles

CR_EWIS_offset_of = Bytes_1OutMap_EWIS; % offset of different dpt in one tile

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EWIS DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% GAPL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAPL DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % transfer length (bytes) per transaction (descriptor) 
CR_GAPL_inpx_cmd_size = NIX_GAPL.*NIY_GAPL0*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_RDPX_GAPL
CR_GAPL_inpx_cmd_size = ceil(CR_GAPL_inpx_cmd_size*8/DMA_WIDTH)*DMA_WIDTH/8; % len_RDpx_pTil_Bytes_PLMX

% GAPL outputs are written to DRAM all at once
CR_GAPL_outpx_cmd_size = NOF_GAPL.*(WD_PX/8)*FCT_DMA; % 24-bit, csr_W_LEN_WRPX_GAPL

% CR_GAPL_offset_tiy = Bytes_Tiy_offset_PLMX;
% CR_GAPL_RD_PX_offset_dpts = DDR3_offset_RDPX_pPLpM; % offset of different descriptors in one tile

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAPL DMA RD/WR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% FCON

% Tof_FCON = POF_FCON;
% CR_CONV_num_toy = ceil(NOY_CONV./Toy_CONV);
% CR_CONV_num_tof = ceil(NOF_CONV./Tof_CONV);
% CONV_NUM_Tiles_1LAYER = CR_CONV_num_toy.*CR_CONV_num_tof; % CONV_NUM_Tiles_1LAYER


CR_FCON_num_tbx = NBX_FCON./Tbx_FCON; % read PX of multiple 64/32 boxes
CR_FCON_num_tif = ceil(NIF_FCON./Tif_FCON); % read WT
CR_FCON_num_tof = ceil(NOF_FCON./Tof_FCON); % read WT
FCON_NUM_Tiles_WT = CR_FCON_num_tif.*CR_FCON_num_tof;
FCON_NUM_Tiles_1LAYER = CR_FCON_num_tbx.*FCON_NUM_Tiles_WT;

NUM_Tiles_px_pFCON = NBX_FCON./Tbx_FCON;
NUM_Tiles_wt_pFCON = ceil(NOF_FCON./Tof_FCON).*ceil(NIF_FCON./Tif_FCON);
NUM_Tiles_wt_FCON = sum(NUM_Tiles_wt_pFCON);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CR_FCON_inwt_cmd_size = (Tof_FCON.*Tif_FCON)*(WD_WT/8); % TILING_BYTES_FC

Bytes_WT_1FCON = (NIF_FCON.*NOF_FCON)*(WD_WT/8);
Bytes_WT_1FCON = ceil(Bytes_WT_1FCON*8/DMA_WIDTH)*DMA_WIDTH/8;
CR_FCON_inwt_addr = zeros(NUM_FCON,1); % base address of each layer
CR_FCON_inwt_addr(1) = DDR3_BDEC_WT_FCON;
for L=2:NUM_FCON
       CR_FCON_inwt_addr(L) = CR_FCON_inwt_addr(L-1) + Bytes_WT_1FCON(L-1);
end

% # of dpts of weights per tile is always 1 

% RTL module of dma_control needs to compute RTL_* online
RTL_FCON_RD_WT_dpt_addr = zeros(NUM_FCON,max(FCON_NUM_Tiles_1LAYER));
for C = 1:NUM_FCON
    ID = ID_global_FCON(C); % = cnt_LAYER in RTL
    for TB = 1:CR_FCON_num_tbx(C)
        for TO = 1:CR_FCON_num_tof(C)
        for TI = 1:CR_FCON_num_tif(C)
            TT = TI+(TO-1)*CR_FCON_num_tif(C)+(TB-1)*FCON_NUM_Tiles_WT(C);
            %for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            ii = 1; % assume FCON only has one input layer
            RTL_FCON_RD_WT_dpt_addr(C,TT) = CR_FCON_inwt_addr(C) + (TT-1)*CR_FCON_inwt_cmd_size(C);
            %end
        end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON read weights (RD_WT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON read inputs (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CR_FCON_inpx_cmd_size = zeros(NUM_FCON,1); % one dpt reads the entire Tbx boxes of PX
R_WORDS_RDPX_FCON_M1  = zeros(NUM_FCON,1); 
for C = 1:NUM_FCON
    ID = ID_global_FCON(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if CR_LAYER_IS_FCON(input_ID)
            ID_Local = find(ID_global_FCON == input_ID);
            CR_FCON_inpx_cmd_size(C) =  NOF_FCON(ID_Local)*Tbx_FCON(ID_Local)*(WD_PX/8);
            R_WORDS_RDPX_FCON_M1 (C) = (NOF_FCON(ID_Local)*WD_PX)/DMA_WIDTH - 1;
        end
        if CR_LAYER_IS_ROIP(input_ID)
            ID_Local = find(ID_global_ROIP == input_ID);
            CR_FCON_inpx_cmd_size(C) =  NOY_ROIP(ID_Local)*NOX_ROIP(ID_Local)*NOF_ROIP(ID_Local)*Tbx_FCON(L)*(WD_PX/8);
            R_WORDS_RDPX_FCON_M1 (C) = (NOY_ROIP(ID_Local)*NOX_ROIP(ID_Local)*NOF_ROIP(ID_Local)*WD_PX)/DMA_WIDTH - 1;
        end
        if CR_LAYER_IS_CONV(input_ID) % NBX_FCON0 must be 1 and the inputs are only stored in the first input buffer. 
            ID_Local = find(ID_global_CONV == input_ID);
            CR_FCON_inpx_cmd_size(C) =  NOY_CONV(ID_Local)*NOX_CONV(ID_Local)*NOF_CONV(ID_Local)*(WD_PX/8);
            R_WORDS_RDPX_FCON_M1 (C) = (NOY_CONV(ID_Local)*NOX_CONV(ID_Local)*NOF_CONV(ID_Local)*WD_PX)/DMA_WIDTH - 1;
        end
        if CR_LAYER_IS_PLMX(input_ID) % NBX_FCON0 must be 1 and the inputs are only stored in the first input buffer.
            ID_Local = find(ID_global_PLMX == input_ID);
            CR_FCON_inpx_cmd_size(C) =  NOY_PLMX(ID_Local)*NOX_PLMX(ID_Local)*NOF_PLMX(ID_Local)*(WD_PX/8);
            R_WORDS_RDPX_FCON_M1 (C) = (NOY_PLMX(ID_Local)*NOX_PLMX(ID_Local)*NOF_PLMX(ID_Local)*WD_PX)/DMA_WIDTH - 1;
        end
        if CR_LAYER_IS_GAPL(input_ID) % NBX_FCON0 must be 1 and the inputs are only stored in the first input buffer.
            ID_Local = find(ID_global_GAPL == input_ID);
            CR_FCON_inpx_cmd_size(C) =  NOF_GAPL(ID_Local)*(WD_PX/8);
            R_WORDS_RDPX_FCON_M1 (C) = (NOF_GAPL(ID_Local)*WD_PX)/DMA_WIDTH - 1;
        end
    end
end

CR_FCON_RD_PX_LayerBase = zeros(NUM_FCON,16); % the MAX number of input layers is 16
for C = 1:NUM_FCON
    ID = ID_global_FCON(C);
    for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
        input_ID = input_layers_ID{ID}(ii);
        if input_ID == 0
            base_LAYER = hex2dec(DDR3_BDEC_IMAGE);
        else
            base_LAYER = CR_LAYER_outpx_addr(input_ID);
        end
        CR_FCON_RD_PX_LayerBase(C,ii) = base_LAYER;
    end
end

% RTL module of dma_control needs to compute RTL_* online
% RTL_FCON_RD_PX_TileBase(per tile, per CONV)
RTL_FCON_RD_PX_TileBase = zeros(max(FCON_NUM_Tiles_1LAYER),NUM_FCON); % assume FCON only has one input layer
for C = 1:NUM_FCON % = cnt_FCON in RTL
    ID = ID_global_FCON(C); % = cnt_LAYER in RTL
    for TB = 1:CR_FCON_num_tbx(C)
        for TO = 1:CR_FCON_num_tof(C)
        for TI = 1:CR_FCON_num_tif(C)
            TT = TI+(TO-1)*CR_FCON_num_tif(C)+(TB-1)*FCON_NUM_Tiles_WT(C);
            %for ii = 1:CR_LAYER_inpx_num_layer(ID) % iterate all input layers
            ii = 1; % assume FCON only has one input layer
                if TO == 1 && TI == 1
                    RTL_FCON_RD_PX_TileBase(TT,C) = CR_FCON_RD_PX_LayerBase(C,ii) + (TB-1)*CR_FCON_inpx_cmd_size(C); 
                else
                    RTL_FCON_RD_PX_TileBase(TT,C) = 1;  % This tile has no RD_PX dpt and only has RD_WT
                end
            %end
        end
        end
    end
end

RTL_FCON_RD_PX_dpt_addr = RTL_FCON_RD_PX_TileBase; % one dpt reads the entire tile of PX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON read inputs (RD_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%// in dma wr, 'roi' in fcon = 'map' in conv
%// csr_tof            : = NUM_FC_BOX
%// csr_offset_of      : addr offset of one        outpx roi /box
%// csr_offset_tof     : addr offset of NUM_FC_BOX outpx rois/boxes 
%// csr_outpx_cmd_size : one cmd wr one roi/box = csr_offset_of

% one cmd writes one entire box of PX to ddr
CR_FCON_outpx_cmd_size = NOF_FCON.*(WD_PX/8);
CR_FCON_offset_of      = NOF_FCON.*(WD_PX/8);
CR_FCON_offset_tof     = NOF_FCON.*(WD_PX/8).*Tbx_FCON;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON write outputs (WR_PX) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

NUM_dpt_1layer_RDpx_1CV = NUM_dpt_RDpx_pCVpTil.*CONV_NUM_Tiles_1LAYER;
NUM_dpt_1layer_RDwt_1CV = NUM_dpt_RDwt_pCVpTil.*CONV_NUM_Tiles_1LAYER;
NUM_dpt_1layer_WRpx_1CV = NUM_dpt_WRpx_pCVpTil.*CONV_NUM_Tiles_1LAYER;
NUM_dpt_1layer_RDpx_1PL = NUM_dpt_RDpx_pPLpTil.*PLMX_NUM_Tiles_1LAYER;
NUM_dpt_1layer_WRpx_1PL = NUM_dpt_WRpx_pPLpTil.*PLMX_NUM_Tiles_1LAYER;
NUM_dpt_1layer_RDpx_1NN = NUM_dpt_RDpx_pNNpTil.*NEAR_NUM_Tiles_1LAYER;
NUM_dpt_1layer_WRpx_1NN = NUM_dpt_WRpx_pNNpTil.*NEAR_NUM_Tiles_1LAYER;

NUM_dpt_max_RDpx_1CV = max(NUM_dpt_RDpx_pCVpTil.*CONV_NUM_Tiles_1LAYER);
NUM_dpt_max_RDwt_1CV = max(NUM_dpt_RDwt_pCVpTil.*CONV_NUM_Tiles_1LAYER);
NUM_dpt_max_WRpx_1CV = max(NUM_dpt_WRpx_pCVpTil.*CONV_NUM_Tiles_1LAYER);
NUM_dpt_max_RDpx_1PL = max(NUM_dpt_RDpx_pPLpTil.*PLMX_NUM_Tiles_1LAYER);
NUM_dpt_max_WRpx_1PL = max(NUM_dpt_WRpx_pPLpTil.*PLMX_NUM_Tiles_1LAYER);
NUM_dpt_max_RDpx_1NN = max(NUM_dpt_RDpx_pNNpTil.*NEAR_NUM_Tiles_1LAYER);
NUM_dpt_max_WRpx_1NN = max(NUM_dpt_WRpx_pNNpTil.*NEAR_NUM_Tiles_1LAYER);


%% For sim verification

% RD_PX: base (start) address per input feature map per LAYER
DDR3_BDEC_perLAYER_perInputMap = zeros(NUM_LAYER,max(NIF_LAYER0));  % NO zero pads
for L = 1:NUM_LAYER
    if CR_LAYER_IS_PROP(L) == 0 && CR_LAYER_IS_FCON(L) == 0 % PROP and FCON don't have the concept of feature map
        MAP = 1;
        for ii = 1:length(input_layers_ID{L}) % iterate all input layers
            input_ID = input_layers_ID{L}(ii);
            if input_ID == 0
                for M = 1 : NIF_LAYER0(ID_local_LAYER(L))
                        DDR3_BDEC_perLAYER_perInputMap(L,MAP) = DDR3_BDEC_IMAGE + (M-1)*Bytes_1OutMap_DATA;
                        MAP = MAP+1;
                end
            else
                for M = 1 : NOF_LAYER0(input_ID)
                    DDR3_BDEC_perLAYER_perInputMap(L,MAP) = CR_LAYER_outpx_addr(input_ID) + (M-1)*Bytes_1OutMap_LAYER(input_ID);
                    MAP = MAP+1;
                end
            end
        end
        if CR_LAYER_IS_EWIS(L) == 0
            if MAP-1 ~= NIF_LAYER0(L)
                fprintf("ERROR31 @ DMA_dpt.m: CONCAT DMA read error #LAYER = %d, MAP-1 = %d, NIF_LAYER0(%d) = %d \n\n",L,MAP-1,L,NIF_LAYER0(L));
            end
        else
            if MAP-1 ~= NIF_LAYER0(L)
                fprintf("ERROR31 @ DMA_dpt.m: EWIS DMA read error #LAYER = %d, MAP-1 = %d, NIF_LAYER0(%d) = %d \n\n",L,MAP-1,L,NIF_LAYER0(L));
            end            
        end
    end
end


% WR_PX: base (start) address per output feature map per LAYER
DDR3_BDEC_perLAYER_perOutputMap = zeros(NUM_LAYER,max(NOF_LAYER0));  % NO zero pads
for L = 1:NUM_LAYER
    if CR_LAYER_IS_PROP(L) == 0 && CR_LAYER_IS_FCON(L) == 0 % PROP and FCON don't have the concept of feature map
        for M = 1 : NOF_LAYER0(L)
            DDR3_BDEC_perLAYER_perOutputMap(L,M) = CR_LAYER_outpx_addr(L) + (M-1)*Bytes_1OutMap_LAYER(L);
        end
    end
end




%% Optional CONV

% DDR3_BDEC_RDWT_pTLpCV = zeros(max(CONV_NUM_Tiles_1LAYER),NUM_CONV);
% for L=1:NUM_CONV
%     for TF = 1 : CR_CONV_num_tof(L)
%     for TY = 1 : CR_CONV_num_toy(L)
%         if TY == 1
%             DDR3_BDEC_RDWT_pTLpCV(TY+(TF-1)*CR_CONV_num_toy(L),L) = DDR3_BDEC_RDWT_pCV(L) + (TF-1)*Bytes_RDwt_pTL(L);
%         end
%     end
%     end
% end
% 
% DDR3_BDEC_RDPX_pCVpM = zeros(NUM_CONV,ceil(max(NIF_CONV))); % addr of each input feature map % NO zero pads
% for C = 1:NUM_CONV
%     ID = ID_global_CONV(C);
%     MAP = 1;
%     for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
%         input_ID = input_layers_ID{ID}(ii);
%         if input_ID == 0
%             for M = 1: NIF_CONV(C)
%                 DDR3_BDEC_RDPX_pCVpM(C,MAP) = hex2dec(DDR3_BADDR_IMAGE) + (M-1)*Bytes_1OutMap_DATA;
%                 MAP = MAP+1;
%             end
%         else
%             for M = 1: NOF_LAYER0(input_ID)
%                 DDR3_BDEC_RDPX_pCVpM(C,MAP) = CR_LAYER_outpx_addr(input_ID) + (M-1)*Bytes_1OutMap_LAYER(input_ID);
%                 MAP = MAP+1;
%             end
%         end
%     end
%     if MAP-1 ~= NIF_CONV(C)
%         fprintf("ERROR31 @ DMA_dpt.m: CONCAT DMA read error #CONV = %d, MAP-1 = %d, NIF_CONV(C) = %d \n\n",C,MAP-1,NIF_CONV(C));
%     end
% end

% %DDR3_BDEC_RDPX_D_CV = cell(1,1,NUM_CONV);
% DDR3_BDEC_RDPX_pTLpDpCV = zeros( max(CONV_NUM_Tiles_1LAYER), max(NUM_dpt_RDpx_pCVpTil), NUM_CONV);
% for L = 1:NUM_CONV
%     for TF = 1: Tiles_RDwt_pCV(L)
%     for TY = 1: Tiles_RDpx_pCV(L)
%        if NOY_CONV(L) == Toy_CONV(L) % Tiles_RDpx_pCV(L) == 1    
%            for D = 1: NUM_dpt_RDpx_pCVpTil(L)
%                DDR3_BDEC_RDPX_pTLpDpCV(TY,D,L) = DDR3_BDEC_RDPX_pCVpM(L,D);
%            end
% %        elseif NOY_CONV(L) == Toy_CONV(L) && with_EltWise(L) == 1 % Tiles_RDpx_pCV(L) == 1
% %            for D = 1: NUM_dpt_RDpx_pCVpTil(L) % Buffer_INpx is overwritten by EltWise, need to RD px from DRAM again
% %                DDR3_BDEC_RDPX_pTLpDpCV(TY+(TF-1)*Tiles_RDpx_pCV(L),D,L) = DDR3_BDEC_RDPX_pCVpM(L,D);
% %            end           
% %            fprintf('Info#12: conv# = %d, Buffer_INpx is overwritten by EltWise, need to RD px from DRAM again\n',L);
%        else
%            for D = 1: NUM_dpt_RDpx_pCVpTil(L)
%                DDR3_BDEC_RDPX_pTLpDpCV(TY+(TF-1)*Tiles_RDpx_pCV(L),D,L) = DDR3_BDEC_RDPX_pCVpM(L,D) - NIX_CONV(L)*PAD_CONV(L)*(WD_PX/8)*FCT_DMA + (TY-1)*Bytes_Tiy_offset_CONV(L);
%            end
%        end
%     end
%     end
% end
% 
% 
% DDR3_BDEC_WRPX_pCVpM = zeros(NUM_CONV,ceil(max(NOF_CONV))); % addr of each output feature map
% for L = 1:NUM_CONV
%    for M = 1: ceil(NOF_CONV(L))
%        DDR3_BDEC_WRPX_pCVpM(L,M) = DDR3_BDEC_CONV(L) + (M-1)*Bytes_1OutMap_CONV(L);
%    end
% end
% 
% DDR3_BDEC_WRPX_pTLpDpCV = zeros(max(CONV_NUM_Tiles_1LAYER), max(NUM_dpt_WRpx_pCVpTil), NUM_CONV);
% for L = 1:NUM_CONV
%     for TF = 1: Tiles_RDwt_pCV(L)
%     for TY = 1: Tiles_RDpx_pCV(L)
%         for D = 1: NUM_dpt_WRpx_pCVpTil(L)    
%             DDR3_BDEC_WRPX_pTLpDpCV(TY+(TF-1)*Tiles_RDpx_pCV(L),D,L) = DDR3_BDEC_WRPX_pCVpM(L,D+(TF-1)*NUM_dpt_WRpx_pCVpTil(L)) + (TY-1)*len_WRpx_pTil_Bytes_CV(L);
%         end
%     end
%     end
% end


%% Optional PLMX
% 
% DDR3_BDEC_RDPX_pPLpM = zeros(NUM_PLMX,ceil(max(NOF_PLMX))); % addr of each input feature map % NO zero pads
% for C = 1:NUM_PLMX
%     ID = ID_global_PLMX(C);
%     MAP = 1;
%     for ii = 1:length(input_layers_ID{ID}) % iterate all input layers
%         input_ID = input_layers_ID{ID}(ii);
%         if input_ID == 0
%             for M = 1: NOF_PLMX(C)
%                 DDR3_BDEC_RDPX_pPLpM(C,MAP) = hex2dec(DDR3_BADDR_IMAGE) + (M-1)*Bytes_1OutMap_DATA;
%                 MAP = MAP+1;
%             end
%         else
%             for M = 1: NOF_LAYER0(input_ID)
%                 DDR3_BDEC_RDPX_pPLpM(C,MAP) = CR_LAYER_outpx_addr(input_ID) + (M-1)*Bytes_1OutMap_LAYER(input_ID);
%                 MAP = MAP+1;
%             end
%         end
%     end
%     if MAP-1 ~= NOF_PLMX(C)
%         fprintf("ERROR31 @ DMA_dpt.m: CONCAT DMA read error #PLMX = %d, MAP-1 = %d, NOF_PLMX(%d) = %d \n\n",C,MAP-1,C,NOF_PLMX(C));
%     end
% end
% 
% DDR3_BDEC_RDPX_pTLpDpPL = zeros(max(CR_PLMX_NUM_Tiles_1LAYER), max(NUM_dpt_RDpx_pPLpTil), NUM_PLMX);
% for L = 1:NUM_PLMX  
%     if Toy_PLMX(L) < NOY_PLMX(L)
%         for TY = 1: CR_PLMX_num_toy(L)
%             for TF = 1: CR_PLMX_num_tof(L)
%                 T = TY+(TF-1)*CR_PLMX_num_toy(L);
%                 for D = 1: NUM_dpt_RDpx_pPLpTil(L)   
%                     DDR3_BDEC_RDPX_pTLpDpPL(T,D,L) = DDR3_BDEC_RDPX_pPLpM(L,D+(TF-1)*NUM_dpt_RDpx_pPLpTil(L)) - NIX_PLMX(L)*PAD_PLMX(L)*(WD_PX/8)*FCT_DMA + (TY-1)*Bytes_Tiy_offset_PLMX(L);
%                 end
%             end
%         end
%     else % one tiling can cover a whole output feature map
%         for T = 1: CR_PLMX_NUM_Tiles_1LAYER(L)
%             for D = 1: NUM_dpt_RDpx_pPLpTil(L)    
%                 DDR3_BDEC_RDPX_pTLpDpPL(T,D,L) = DDR3_BDEC_RDPX_pPLpM(L,D+(T-1)*NUM_dpt_RDpx_pPLpTil(L));
%             end
%         end        
%     end
% end
% 
% DDR3_BDEC_WRPX_pPLpM = zeros(NUM_PLMX,ceil(max(NOF_PLMX))); % addr of each input feature map
% for L = 1:NUM_PLMX
%         for M = 1: ceil(NOF_PLMX(L))
%             DDR3_BDEC_WRPX_pPLpM(L,M) = DDR3_BDEC_PLMX(L) + (M-1)*Bytes_1OutMap_PLMX(L);
%         end
% end
% DDR3_BDEC_WRPX_pTLpDpPL = zeros(max(CR_PLMX_NUM_Tiles_1LAYER), max(NUM_dpt_RDpx_pPLpTil), NUM_PLMX);
% for L = 1:NUM_PLMX  
%     if Toy_PLMX(L) < NOY_PLMX(L)
%         for TY = 1: CR_PLMX_num_toy(L)
%             for TF = 1: CR_PLMX_num_tof(L)
%                 T = TY+(TF-1)*CR_PLMX_num_toy(L);
%                 for D = 1: NUM_dpt_RDpx_pPLpTil(L)   
%                     DDR3_BDEC_WRPX_pTLpDpPL(T,D,L) = DDR3_BDEC_WRPX_pPLpM(L,D+(TF-1)*NUM_dpt_RDpx_pPLpTil(L)) + (TY-1)*len_WRpx_pTil_Bytes_PLMX(L);
%                 end
%             end
%         end              
%     else % one tiling can cover a whole output feature map
%         for T = 1: CR_PLMX_NUM_Tiles_1LAYER(L)
%             for D = 1: NUM_dpt_RDpx_pPLpTil(L)    
%                 DDR3_BDEC_WRPX_pTLpDpPL(T,D,L) = DDR3_BDEC_WRPX_pPLpM(L,D+(T-1)*NUM_dpt_RDpx_pPLpTil(L));
%             end
%         end        
%     end
% end






