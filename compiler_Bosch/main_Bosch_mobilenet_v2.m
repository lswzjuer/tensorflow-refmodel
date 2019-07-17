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
% use: models/DLA2.0_prototxt/DLA2.0_Bosch_mobilenet_v2.prototxt

NUM_CONV = 39; % CONV includes normal convolution and depthwise convolution
NUM_DECV = 0; % DEConVolution (deconv) or transposed convolution
NUM_NEAR = 0; % Nearest-neighbor interpolation
NUM_PLMX = 0; % PooLing MaX layer
NUM_GAPL = 0; % Global AVerage Pooling layer
NUM_ROIP = 0; % ROIPooling layer
NUM_PROP = 0; % PROPosal layer
NUM_EWIS = 8; % Element-WISe layer
NUM_FCON = 0; % Fully-CONnected layer
NUM_LAYER = NUM_CONV+NUM_DECV+NUM_NEAR+NUM_PLMX+NUM_GAPL+NUM_ROIP+NUM_PROP+NUM_EWIS+NUM_FCON;

%             1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
NKX_CONV  = [ 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,]';
NKY_CONV  = [ 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,]';
PAD_CONV  = [ 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,]';
STR_CONV  = [ 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]';
NKI_CONV0 = [ 3, 1,32,16, 1,96,24, 1,144,24, 1,144,32, 1,192,32, 1,192,32, 1,192,64, 1,384,64, 1,384,64, 1,384,64, 1,384,96, 1,576,96, 1,576,]'; % # of input kernel maps
NIF_CONV0 = [ 3,32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,]';
NOF_CONV0 = [32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,96,]';
NIX_CONV0 = [1080,540,540,540,540,270,270,270,270,270,270,135,135,135,135,135,135,135,135,135,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,]';
NIY_CONV0 = [1920,960,960,960,960,480,480,480,480,480,480,240,240,240,240,240,240,240,240,240,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,]';
NOX_CONV0 = [540,540,540,540,270,270,270,270,270,270,135,135,135,135,135,135,135,135,135,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,]';
NOY_CONV0 = [960,960,960,960,480,480,480,480,480,480,240,240,240,240,240,240,240,240,240,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,]';

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

NKX_PLMX  = []';
NKY_PLMX  = []';
PAD_PLMX  = []';
STR_PLMX  = []';
NIF_PLMX0 = []';
NOF_PLMX0 = []';
NIX_PLMX0 = []';
NIY_PLMX0 = []';
NOX_PLMX0 = []';
NOY_PLMX0 = []';

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

NIF_EWIS0 = [48,64,64,128,128,128,192,192,]';
NOF_EWIS0 = [24,32,32,64,64,64,96,96,]';
NIX_EWIS0 = [270,135,135,68,68,68,68,68,]';
NIY_EWIS0 = [480,240,240,120,120,120,120,120,]';
NOX_EWIS0 = [270,135,135,68,68,68,68,68,]';
NOY_EWIS0 = [480,240,240,120,120,120,120,120,]';

NIF_FCON0 = []'; % Warning check NIF of 1st FCON
NOF_FCON0 = []';
NBX_FCON0 = []'; % # of ROIs or anchor boxes


%                    1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
CR_LAYER_IS_CONV = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,];
CR_LAYER_IS_DECV = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_NEAR = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_PLMX = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_ROIP = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_PROP = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_GAPL = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];
CR_LAYER_IS_EWIS = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,];
CR_LAYER_IS_FCON = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,];

CR_CONV_with_ReLU = [0,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0]';
CR_EWIS_with_ReLU = [0,0,0,0,0,0,0,0]';
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
input_layers_ID{010} = [9,6,];
input_layers_ID{011} = [10,];
input_layers_ID{012} = [11,];
input_layers_ID{013} = [12,];
input_layers_ID{014} = [13,];
input_layers_ID{015} = [14,];
input_layers_ID{016} = [15,];
input_layers_ID{017} = [16,13,];
input_layers_ID{018} = [17,];
input_layers_ID{019} = [18,];
input_layers_ID{020} = [19,];
input_layers_ID{021} = [20,17,];
input_layers_ID{022} = [21,];
input_layers_ID{023} = [22,];
input_layers_ID{024} = [23,];
input_layers_ID{025} = [24,];
input_layers_ID{026} = [25,];
input_layers_ID{027} = [26,];
input_layers_ID{028} = [27,24,];
input_layers_ID{029} = [28,];
input_layers_ID{030} = [29,];
input_layers_ID{031} = [30,];
input_layers_ID{032} = [31,28,];
input_layers_ID{033} = [32,];
input_layers_ID{034} = [33,];
input_layers_ID{035} = [34,];
input_layers_ID{036} = [35,32,];
input_layers_ID{037} = [36,];
input_layers_ID{038} = [37,];
input_layers_ID{039} = [38,];
input_layers_ID{040} = [39,];
input_layers_ID{041} = [40,];
input_layers_ID{042} = [41,];
input_layers_ID{043} = [42,39,];
input_layers_ID{044} = [43,];
input_layers_ID{045} = [44,];
input_layers_ID{046} = [45,];
input_layers_ID{047} = [46,43,];



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

             %  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
%NOY_CONV0 = [960,960,960,960,480,480,480,480,480,480,240,240,240,240,240,240,240,240,240,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,]';
Toy_CONV0  = [  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,]';
Toy_CONV0  = min(Toy_CONV0, NOY_CONV0);

            % 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,11, 12, 13,14, 15, 16,17, 18, 19,20, 21, 22,23, 24, 25,
%NOF_CONV0= [32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,96,]';
Tof_CONV0 = [32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,96,]';
Tof_CONV0 = min(Tof_CONV0, NOF_CONV0);

Toy_DECV0 = []; %8*ones(NUM_DECV,1);
Tof_DECV0 = NOF_DECV0;

Toy_NEAR = []; %8*ones(NUM_DECV,1);
Tif_NEAR = [];

%NIF_PLMX0 = [ 64,128,256,]';
Tif_PLMX   = []; % if FCON is after PLMX, Tif_PL = NIF_PL % NIF_PL0 = [96;192;96;];
%NOY_PLMX0 = [480,240,120,]';
Toy_PLMX   = []; % NOTE, for stride=1, pad=1 pooling, TOY_PL = NOY_PL, NOY_PLMX0 = [128;64;32;16;08;04;02;];

%NOY_EWIS0=[270,135,135,68,68,68,68,68,]';
Toy_EWIS = [ 30, 27, 27,34,34,34,34,34,]';

Tif_GAPL = [32;];
Tof_GAPL = Tif_GAPL;

Tif_PROP = [1;]; 
Toy_PROP = [1;];

Tof_ROIP = [64;]; 

NUM_FC_BOX = 1; % # of parallel computed boxes for FCON
Tbx_FCON = []; %NUM_FC_BOX*ones(NUM_FCON,1);
Tif_FCON = NIF_FCON0; %Tof_FCON = POF_FCON;


%%
run ./CNN_compiler/DLA_size_x16y08f16.m
run ./CNN_compiler/Variants.m
run ./CNN_compiler/DMA_dpt_NoAlign.m % run 1st
run ./CNN_compiler/Param.m % run 2nd
run ./CNN_compiler/Param_LAYER.m % generate LAYER based configuration parameters
run ./CNN_compiler/Performance.m
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

% LAYER(001) is CONV(001), conv1
% LAYER(002) is CONV(002), bneck_conv_dw_0
% LAYER(003) is CONV(003), conv_pw_lin_0
% LAYER(004) is CONV(004), conv_exp_1
% LAYER(005) is CONV(005), bneck_conv_dw_1
% LAYER(006) is CONV(006), conv_pw_lin_1
% LAYER(007) is CONV(007), conv_exp_2
% LAYER(008) is CONV(008), bneck_conv_dw_2
% LAYER(009) is CONV(009), conv_pw_lin_2
% LAYER(010) is EWIS(001), residual_2
% LAYER(011) is CONV(010), conv_exp_3
% LAYER(012) is CONV(011), bneck_conv_dw_3
% LAYER(013) is CONV(012), conv_pw_lin_3
% LAYER(014) is CONV(013), conv_exp_4
% LAYER(015) is CONV(014), bneck_conv_dw_4
% LAYER(016) is CONV(015), conv_pw_lin_4
% LAYER(017) is EWIS(002), residual_4
% LAYER(018) is CONV(016), conv_exp_5
% LAYER(019) is CONV(017), bneck_conv_dw_5
% LAYER(020) is CONV(018), conv_pw_lin_5
% LAYER(021) is EWIS(003), residual_5
% LAYER(022) is CONV(019), conv_exp_6
% LAYER(023) is CONV(020), bneck_conv_dw_6
% LAYER(024) is CONV(021), conv_pw_lin_6
% LAYER(025) is CONV(022), conv_exp_7
% LAYER(026) is CONV(023), bneck_conv_dw_7
% LAYER(027) is CONV(024), conv_pw_lin_7
% LAYER(028) is EWIS(004), residual_7
% LAYER(029) is CONV(025), conv_exp_8
% LAYER(030) is CONV(026), bneck_conv_dw_8
% LAYER(031) is CONV(027), conv_pw_lin_8
% LAYER(032) is EWIS(005), residual_8
% LAYER(033) is CONV(028), conv_exp_9
% LAYER(034) is CONV(029), bneck_conv_dw_9
% LAYER(035) is CONV(030), conv_pw_lin_9
% LAYER(036) is EWIS(006), residual_9
% LAYER(037) is CONV(031), conv_exp_10
% LAYER(038) is CONV(032), bneck_conv_dw_10
% LAYER(039) is CONV(033), conv_pw_lin_10
% LAYER(040) is CONV(034), conv_exp_11
% LAYER(041) is CONV(035), bneck_conv_dw_11
% LAYER(042) is CONV(036), conv_pw_lin_11
% LAYER(043) is EWIS(007), residual_11
% LAYER(044) is CONV(037), conv_exp_12
% LAYER(045) is CONV(038), bneck_conv_dw_12
% LAYER(046) is CONV(039), conv_pw_lin_12
% LAYER(047) is EWIS(008), residual_12

