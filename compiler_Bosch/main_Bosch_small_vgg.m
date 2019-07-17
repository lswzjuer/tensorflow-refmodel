%Generated by Gen_MATLAB_CNNs_param.py

clear
clc

% % % workspace = getenv("WORKSPACE");
% % % global matlab_export_dir = strcat(workspace,"/verif/sim/");
% % % jsonlab_path = strcat(workspace,"/verif/compiler/utils/jsonlab-1.5");
% % % addpath(jsonlab_path);
% % % function r = write_json(filename, structure)
% % %     global matlab_export_dir;
% % %     filepath = strcat(matlab_export_dir,filename);
% % %     fidr = fopen(filepath,'w');
% % %     json_str = savejson('',structure);
% % %     fprintf(fidr,"%s", json_str);
% % %     fclose(fidr);
% % %     r=1;
% % % end


%% *************************************************************** CNN model parameters ***************************************************************
% Generated by Gen_MATLAB_DLA_param.py
% use: models/DLA2.0_prototxt/DLA2.0_Bosch_small_vgg.prototxt

NUM_CONV = 13; % CONV includes normal convolution and depthwise convolution
NUM_DECV = 0; % DEConVolution (deconv) or transposed convolution
NUM_NEAR = 0; % Nearest-neighbor interpolation
NUM_PLMX = 4; % PooLing MaX layer
NUM_GAPL = 0; % Global AVerage Pooling layer
NUM_ROIP = 0; % ROIPooling layer
NUM_PROP = 0; % PROPosal layer
NUM_EWIS = 0; % Element-WISe layer
NUM_FCON = 0; % Fully-CONnected layer
NUM_LAYER = NUM_CONV+NUM_DECV+NUM_NEAR+NUM_PLMX+NUM_GAPL+NUM_ROIP+NUM_PROP+NUM_EWIS+NUM_FCON;

%             1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,
NKX_CONV  = [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,]';
NKY_CONV  = [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,]';
PAD_CONV  = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]';
STR_CONV  = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]';
NKI_CONV0 = [ 3,16,16,32,32,64,64,64,128,128,128,128,128,]'; % # of input kernel maps
NIF_CONV0 = [ 3,16,16,32,32,64,64,64,128,128,128,128,128,]';
NOF_CONV0 = [16,16,32,32,64,64,64,128,128,128,128,128,128,]';
NIX_CONV0 = [1080,1080,540,540,270,270,270,135,135,135,68,68,68,]';
NIY_CONV0 = [1920,1920,960,960,480,480,480,240,240,240,120,120,120,]';
NOX_CONV0 = [1080,1080,540,540,270,270,270,135,135,135,68,68,68,]';
NOY_CONV0 = [1920,1920,960,960,480,480,480,240,240,240,120,120,120,]';

NKX_DECV  = []';
NKY_DECV  = []';
PAD_DECV  = []';
STR_DECV  = []'; % Deconv stride (Internal Zero Padding Size) 
NKI_DECV0 = []';
NIF_DECV0 = []';
NOF_DECV0 = []';
NIX_DECV0 = []';
NIY_DECV0 = []';
NOX_DECV0 = []';
NOY_DECV0 = []';

STR_NEAR  = []';
NIF_NEAR0 = []';
NOF_NEAR0 = []';
NIX_NEAR0 = []';
NIY_NEAR0 = []';
NOX_NEAR0 = []';
NOY_NEAR0 = []';

NKX_PLMX  = [ 2, 2, 2, 2,]';
NKY_PLMX  = [ 2, 2, 2, 2,]';
PAD_PLMX  = [ 0, 0, 0, 0,]';
STR_PLMX  = [ 2, 2, 2, 2,]';
NIF_PLMX0 = [16,32,64,128,]';
NOF_PLMX0 = [16,32,64,128,]';
NIX_PLMX0 = [1080,540,270,135,]';
NIY_PLMX0 = [1920,960,480,240,]';
NOX_PLMX0 = [540,270,135,68,]';
NOY_PLMX0 = [960,480,240,120,]';

NOX_ROIP0 = []';
NOY_ROIP0 = []';
NOF_ROIP0 = []';
NBX_ROIP0 = []'; % # of ROIs or anchor boxes


WD_DIV = 16;
NKX_GAPL  = []';
NKY_GAPL  = []';
NIX_GAPL0 = []';
NIY_GAPL0 = []';
NIF_GAPL0 = []';
NOF_GAPL0 = []';

NIF_EWIS0 = []';
NOF_EWIS0 = []';
NIX_EWIS0 = []';
NIY_EWIS0 = []';
NOX_EWIS0 = []';
NOY_EWIS0 = []';

NIF_FCON0 = []'; % Warning check NIF of 1st FCON
NOF_FCON0 = []';
NBX_FCON0 = []'; % # of ROIs or anchor boxes


%                    1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,
CR_LAYER_IS_CONV = [ 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,];
CR_LAYER_IS_DECV = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_NEAR = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_PLMX = [ 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,];
CR_LAYER_IS_ROIP = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_PROP = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_GAPL = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_EWIS = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_FCON = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];

CR_LAYER_IS_DWIS = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];

CR_CONV_with_ReLU = [1,1,1,1,1,1,1,1,1,1,1,1,1]';
CR_EWIS_with_ReLU = []';
CR_FCON_with_ReLU = []';
CR_CONV_with_Bias = ones(NUM_CONV,1);
CR_FCON_with_Bias = ones(NUM_FCON,1);

CR_CONV_MACOUT_RSBIT_CV = 7*ones(NUM_CONV,1);
CR_DECV_MACOUT_RSBIT_CV = 7*ones(NUM_DECV,1);
CR_FCON_MACOUT_RSBIT_FC = 7*ones(NUM_FCON,1); % FCON MAC output right shift bits


% Support MAX number of input layers of 16! 
input_layers_ID{001} = [0,]; % Assume 1st layer only read from image
input_layers_ID{002} = [1,];
input_layers_ID{003} = [2,];
input_layers_ID{004} = [3,];
input_layers_ID{005} = [4,];
input_layers_ID{006} = [5,];
input_layers_ID{007} = [6,];
input_layers_ID{008} = [7,];
input_layers_ID{009} = [8,];
input_layers_ID{010} = [9,];
input_layers_ID{011} = [10,];
input_layers_ID{012} = [11,];
input_layers_ID{013} = [12,];
input_layers_ID{014} = [13,];
input_layers_ID{015} = [14,];
input_layers_ID{016} = [15,];
input_layers_ID{017} = [16,];


% ------------------------------ PROP parameters ------------------------------
BIT_frac_ANCHOR      = 4;
num_anchors          = 0*ones(NUM_PROP,1);
CR_PROP_ANCHOR_CTR_X = 4.5*2^BIT_frac_ANCHOR*ones(NUM_PROP,1); % 16-bit
CR_PROP_ANCHOR_CTR_Y = 4.5*2^BIT_frac_ANCHOR*ones(NUM_PROP,1); % 16-bit
CR_PROP_ANCHOR_W     = zeros(NUM_PROP,64);     % 64*16-bit
CR_PROP_ANCHOR_H     = zeros(NUM_PROP,64);     % 64*16-bit
%CR_PROP_ANCHOR_W(1,1:num_anchors(1)) = [  9, 18, 36, 54, 96,192,288, 10, 21, 42, 63,112,224,336, 12, 24, 48, 72,128,256,384, 15, 30, 60, 90,160,320,480, 18, 36, 72,108,192,384,576, 21, 42, 84,126,224,448,672, 27, 54,108,162,288,576,864]*2^BIT_frac_ANCHOR;
%CR_PROP_ANCHOR_H(1,1:num_anchors(1)) = [ 25, 51,102,153,272,544,816, 21, 42, 84,126,224,448,672, 18, 36, 72,108,192,384,576, 15, 30, 60, 90,160,320,480, 12, 24, 48, 72,128,256,384, 10, 21, 42, 63,112,224,336,  9, 18, 36, 54, 96,192,288]*2^BIT_frac_ANCHOR;
% ------------------------------ PROP parameters ------------------------------



%% *************************************************************** Loop Tiling Sizes for On-chip Buffer ***************************************************************

             %  01;  02; 03; 04; 05; 06; 07; 08; 09; 10; 11; 12; 13;
%NOY_CONV0 = [1920,1920,960,960,480,480,480,240,240,240,120,120,120,]';
Toy_CONV0  = [   8,   8,  8, 8,  8,   8,  8,  8,  8,  8,  8,  8,  8,]';
Toy_CONV0  = min(Toy_CONV0, NOY_CONV0);

             %01;02;03;04;05;06;07; 08; 09; 10; 11; 12; 13;
%NOF_CONV0= [16,16,32,32,64,64,64,128,128,128,128,128,128,]';
Tof_CONV0 = [16,16,32,32,64,64,64,128,128,128,128,128,128,]';
Tof_CONV0 = min(Tof_CONV0, NOF_CONV0);

Toy_DECV0 = []; %8*ones(NUM_DECV,1);
Tof_DECV0 = NOF_DECV0;

Toy_NEAR = []; %8*ones(NUM_DECV,1);
Tif_NEAR = [];

%NIF_PLMX0 = [ 16, 32, 64,128,]';
Tif_PLMX   = [  8,  8,  8,  8,]'; % if FCON is after PLMX, Tif_PL = NIF_PL % NIF_PL0 = [96;192;96;];
%NOY_PLMX0 = [960,480,240,120,]';
Toy_PLMX   = [ 20, 20, 20, 20,]'; % NOTE, for stride=1, pad=1 pooling, TOY_PL = NOY_PL, NOY_PLMX0 = [128;64;32;16;08;04;02;];

%NOY_EWIS0=[];
Toy_EWIS = [];

Tif_GAPL = [32;];
Tof_GAPL = Tif_GAPL;

Tif_PROP = [1;]; 
Toy_PROP = [1;];

Tof_ROIP = [64;]; 

NUM_FC_BOX = 1; % # of parallel computed boxes for FCON
Tbx_FCON = []; %NUM_FC_BOX*ones(NUM_FCON,1);
Tif_FCON = NIF_FCON0; %Tof_FCON = POF_FCON;


%%
run ./CNN_compiler/DLA_size.m
run ./CNN_compiler/Variants.m
run ./CNN_compiler/DMA_dpt_NoAlign.m % run 1st
run ./CNN_compiler/Param.m % run 2nd
run ./CNN_compiler/Param_LAYER.m % generate LAYER based configuration parameters
run ./CNN_compiler/Performance_Bosch.m
run ./CNN_compiler/csv_layer.m

% % % run ./CNN_compiler/cnn_emu_api.m

% save ("DDR3_BDEC_WRPX_pCVpM","DDR3_BDEC_WRPX_pCVpM");
% save ("DDR3_BDEC_WRPX_pPLpM","DDR3_BDEC_WRPX_pPLpM");
% save ("./CNN_compiler/DDR3_BDEC_WRPX_pCVpM","DDR3_BDEC_WRPX_pCVpM");
% save ("./CNN_compiler/DDR3_BDEC_WRPX_pPLpM","DDR3_BDEC_WRPX_pPLpM");

% load('./CNN_compiler/PVANET_ob_KN_BS_INT.mat')
% load('./CNN_compiler/FT_data_6images.mat')

%run ./CNN_compiler/Bias_Kernel_Image_sim.m

%run ./CNN_compiler/RAM_init.m

close('all')

%% ************************* Layer Information *************************

% LAYER(001) is CONV(001), conv1_1
% LAYER(002) is CONV(002), conv1_2
% LAYER(003) is PLMX(001), pool1
% LAYER(004) is CONV(003), conv2_1
% LAYER(005) is CONV(004), conv2_2
% LAYER(006) is PLMX(002), pool2
% LAYER(007) is CONV(005), conv3_1
% LAYER(008) is CONV(006), conv3_2
% LAYER(009) is CONV(007), conv3_3
% LAYER(010) is PLMX(003), pool3
% LAYER(011) is CONV(008), conv4_1
% LAYER(012) is CONV(009), conv4_2
% LAYER(013) is CONV(010), conv4_3
% LAYER(014) is PLMX(004), pool4
% LAYER(015) is CONV(011), conv5_1
% LAYER(016) is CONV(012), conv5_2
% LAYER(017) is CONV(013), conv5_3



