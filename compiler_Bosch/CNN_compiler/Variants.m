
%% Variants of parameters and design variables

% NOTE: if a variable name is affiliated with 0, it means this variable represents the original network size.  

cnt_CONV = 1; cnt_DECV = 1; cnt_NEAR = 1; cnt_PLMX = 1; cnt_GAPL = 1; cnt_ROIP = 1; cnt_PROP = 1; cnt_EWIS = 1; cnt_FCON = 1;
ID_global_CONV = zeros(NUM_CONV,1);
ID_global_DECV = zeros(NUM_DECV,1);
ID_global_NEAR = zeros(NUM_NEAR,1);
ID_global_PLMX = zeros(NUM_PLMX,1);
ID_global_GAPL = zeros(NUM_GAPL,1);
ID_global_ROIP = zeros(NUM_ROIP,1);
ID_global_PROP = zeros(NUM_PROP,1);
ID_global_EWIS = zeros(NUM_EWIS,1);
ID_global_FCON = zeros(NUM_FCON,1);
ID_local_LAYER = zeros(NUM_LAYER,1);
NOX_LAYER0     = zeros(NUM_LAYER,1);
NOY_LAYER0     = zeros(NUM_LAYER,1);
NIF_LAYER0     = zeros(NUM_LAYER,1);
NOF_LAYER0     = zeros(NUM_LAYER,1);
NIX_LAYER0     = zeros(NUM_LAYER,1);
for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) == 1
        ID_local_LAYER(L) = cnt_CONV;
        ID_global_CONV(cnt_CONV) = L;
        NOX_LAYER0(L) = NOX_CONV0(cnt_CONV);
        NOY_LAYER0(L) = NOY_CONV0(cnt_CONV);
        NIF_LAYER0(L) = NIF_CONV0(cnt_CONV);
        NOF_LAYER0(L) = NOF_CONV0(cnt_CONV);
        NIX_LAYER0(L) = NIX_CONV0(cnt_CONV);
        cnt_CONV = cnt_CONV+1;
    end
    if CR_LAYER_IS_DECV(L) == 1
        ID_local_LAYER(L) = cnt_DECV;
        ID_global_DECV(cnt_DECV) = L;
        NOX_LAYER0(L) = NOX_DECV0(cnt_DECV);
        NOY_LAYER0(L) = NOY_DECV0(cnt_DECV);
        NIF_LAYER0(L) = NIF_DECV0(cnt_DECV);
        NOF_LAYER0(L) = NOF_DECV0(cnt_DECV);
        NIX_LAYER0(L) = NIX_DECV0(cnt_DECV);
        cnt_DECV = cnt_DECV+1;
    end
    if CR_LAYER_IS_NEAR(L) == 1
        ID_local_LAYER(L) = cnt_NEAR;
        ID_global_NEAR(cnt_NEAR) = L;
        NOX_LAYER0(L) = NOX_NEAR0(cnt_NEAR);
        NOY_LAYER0(L) = NOY_NEAR0(cnt_NEAR);
        NIF_LAYER0(L) = NIF_NEAR0(cnt_NEAR);
        NOF_LAYER0(L) = NOF_NEAR0(cnt_NEAR);
        NIX_LAYER0(L) = NIX_NEAR0(cnt_NEAR);
        cnt_NEAR = cnt_NEAR+1;
    end
    if CR_LAYER_IS_PLMX(L) == 1
        ID_local_LAYER(L) = cnt_PLMX;
        ID_global_PLMX(cnt_PLMX) = L;
        NOX_LAYER0(L) = NOX_PLMX0(cnt_PLMX);
        NOY_LAYER0(L) = NOY_PLMX0(cnt_PLMX);
        NIF_LAYER0(L) = NIF_PLMX0(cnt_PLMX);
        NOF_LAYER0(L) = NOF_PLMX0(cnt_PLMX);
        NIX_LAYER0(L) = NIX_PLMX0(cnt_PLMX);
        cnt_PLMX = cnt_PLMX+1;
    end
    if CR_LAYER_IS_GAPL(L) == 1
        ID_local_LAYER(L) = cnt_GAPL;
        ID_global_GAPL(cnt_GAPL) = L;
        NOX_LAYER0(L) = 1;
        NOY_LAYER0(L) = 1;
        NIF_LAYER0(L) = NIF_GAPL0(cnt_GAPL);
        NOF_LAYER0(L) = NOF_GAPL0(cnt_GAPL);
        NIX_LAYER0(L) = NIX_GAPL0(cnt_GAPL);
        cnt_GAPL = cnt_GAPL+1;
    end
    if CR_LAYER_IS_ROIP(L) == 1
        ID_local_LAYER(L) = cnt_ROIP;
        ID_global_ROIP(cnt_ROIP) = L;
        NOX_LAYER0(L) = NOX_ROIP0(cnt_ROIP);
        NOY_LAYER0(L) = NOY_ROIP0(cnt_ROIP);
        NIF_LAYER0(L) = NOF_ROIP0(cnt_ROIP);
        NOF_LAYER0(L) = NOF_ROIP0(cnt_ROIP);
        cnt_ROIP = cnt_ROIP+1;
    end
    if CR_LAYER_IS_PROP(L) == 1
        ID_local_LAYER(L) = cnt_PROP;
        ID_global_PROP(cnt_PROP) = L;
        cnt_PROP = cnt_PROP+1;
    end
    if CR_LAYER_IS_EWIS(L) == 1
        ID_local_LAYER(L) = cnt_EWIS;
        ID_global_EWIS(cnt_EWIS) = L;
        NOX_LAYER0(L) = NOX_EWIS0(cnt_EWIS);
        NOY_LAYER0(L) = NOY_EWIS0(cnt_EWIS);
        NIF_LAYER0(L) = NIF_EWIS0(cnt_EWIS);
        NOF_LAYER0(L) = NOF_EWIS0(cnt_EWIS);
        NIX_LAYER0(L) = NIX_EWIS0(cnt_EWIS);
        cnt_EWIS = cnt_EWIS+1;
    end
    if CR_LAYER_IS_FCON(L) == 1
        ID_local_LAYER(L) = cnt_FCON;
        ID_global_FCON(cnt_FCON) = L;
        NOX_LAYER0(L) = 1;
        NOY_LAYER0(L) = 1;
        NIF_LAYER0(L) = NIF_FCON0(cnt_FCON);
        NOF_LAYER0(L) = NOF_FCON0(cnt_FCON);
        cnt_FCON = cnt_FCON+1;
    end       
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONV Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NIF_CONV = NIF_CONV0;
NKI_CONV = NIF_CONV; % in DLA2.0, zeros are filled in kernels
for L = 1:NUM_CONV
    while NKX_CONV(L)*NKY_CONV(L)*NKI_CONV(L) < POF1OUPX*POY+6 % need to add redundant zero kernel maps
       NIF_CONV(L) = NIF_CONV(L)+1; 
       NKI_CONV(L) = NKI_CONV(L)+1;
       fprintf('Warning @ Variants: L=%d, NIK = %d, NKX_CONV*NKY_CONV*NIK = %d must > POF1OUPX*POY+6 = %d\n',L, NKI_CONV(L), NKX_CONV(L)*NKY_CONV(L)*NKI_CONV(L),(POF1OUPX*POY+6));
    end
end
fprintf('\n');

Toy_CONV = ceil(Toy_CONV0/POY)*POY;
Tof_CONV = ceil(Tof_CONV0/POF)*POF;

NOF_CONV = ceil(NOF_CONV0./Tof_CONV).*Tof_CONV;
NOY_CONV = ceil(NOY_CONV0./POY)*POY;
NIY_CONV = max((NOY_CONV-1).*STR_CONV+NKX_CONV-2*PAD_CONV, NIY_CONV0);
NOX_CONV = ceil(NOX_CONV0./PX_AD).*PX_AD;
NIX_CONV = ceil(NIX_CONV0./PX_AD).*PX_AD;

Tox_CONV  = NOX_CONV; % store at least one-row of feature map
Tix_CONV = (Tox_CONV-1).*STR_CONV+NKX_CONV; % include PAD_CONV
Tiy_CONV = (Toy_CONV-1).*STR_CONV+NKY_CONV;
Tiy_CONV = min(Tiy_CONV,NIY_CONV+2*PAD_CONV);

Tiy_CONV_DMA = min(Tiy_CONV,NIY_CONV);
Tiy_CONV_DMA_PAD = zeros(NUM_CONV,1);
for L = 1:NUM_CONV
   if NOY_CONV(L) == Toy_CONV(L) % Tiy_CONV_DMA does not include PAD_CONV
       Tiy_CONV_DMA_PAD(L) = Tiy_CONV_DMA(L)+PAD_CONV(L);
   else % Tiy_DMA includes PAD_CONV
       Tiy_CONV_DMA_PAD(L) = Tiy_CONV_DMA(L)       ;
   end
end

NOY_CONV_WRpx = ceil(NOY_CONV./Toy_CONV).*Toy_CONV;

NUM_TILE_pCONV = ceil(NOY_CONV./Toy_CONV).*ceil(NOF_CONV./Tof_CONV);
NUM_TILE_CONV = sum(NUM_TILE_pCONV);



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECV Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NIF_DECV = NIF_DECV0;
NKI_DECV = NIF_DECV; % in DLA2.0, zeros are filled in kernels
for L = 1:NUM_DECV
    while NKX_DECV(L)*NKY_DECV(L)*NKI_DECV(L) < POF1OUPX*POY+3 % need to add redundant zero kernel maps
       NIF_DECV(L) = NIF_DECV(L)+1; 
       NKI_DECV(L) = NKI_DECV(L)+1;
       fprintf('Warning @ Variants: L=%d, NIK = %d, NKX_DECV*NKY_DECV*NIK = %d must > POF1OUPX*POY+3 = %d\n',L, NKI_DECV(L), NKX_DECV(L)*NKY_DECV(L)*NKI_DECV(L),(POF1OUPX*POY+3));
    end
end
fprintf('\n');

if NUM_DECV > 0
    PAD_D = zeros(NUM_DECV,1);
    STR_D = zeros(NUM_DECV,1);
    for L = 1:NUM_DECV
      PAD_D(L) = NKX_DECV(L) - PAD_DECV(L) - 1;
      STR_D(L) = 1;
    end
else
    PAD_D = [];
    STR_D = [];
end

%       Original  Read Side   Compute/Write Side
% NIX   56        56          111
% NIY   90        90          179
% NOX   112       56          112
% NOY   180       90          180
% TOX   NOX=112   56          112
% TOY   TOY=32    32          (32-1)*2+1 = 63

Toy_DECV    = ceil(Toy_DECV0/POY)*POY;
Tof_DECV    = ceil(Tof_DECV0/POF)*POF;
NOF_DECV    = ceil(NOF_DECV0./Tof_DECV).*Tof_DECV;
NOY_DECV    = ceil(NOY_DECV0./POY)*POY;

Toy_W_DECV0 = (Toy_DECV0-1).*STR_DECV + 1;
Toy_R_DECV  = Toy_DECV;
Toy_W_DECV  = ceil(Toy_W_DECV0/POY)*POY;

NIY_DECV    = max((NOY_DECV-1).*STR_D+NKY_DECV-2*PAD_D, NIY_DECV0); % max((NOY_DECV + 2*PAD_DECV - NKY_DECV)./STR_DECV + 1, NIY_DECV0); % for DECV: NOX = (NIX-1)*STR+NKX-2*PAD;
NOX_DECV    = ceil(NOX_DECV0./PX_AD).*PX_AD;
NIX_DECV    = ceil(NIX_DECV0./PX_AD).*PX_AD;

Tox_DECV    = NOX_DECV; % store at least one-row of feature map
% YM Formula: Tix_DECV    = (Tox_DECV + 2*PAD_D- NKX_DECV)./STR_D+ 1; % include external padding
% Ym Formula: Tiy_DECV    = (Toy_DECV + 2*PAD_D- NKY_DECV)./STR_D+ 1; % include external padding
Tix_DECV = (Tox_DECV-1).*STR_D+NKX_DECV; % include PAD_DECV
Tiy_DECV = (Toy_DECV-1).*STR_D+NKY_DECV;
Tiy_DECV = min(Tiy_DECV,NIY_DECV+2*PAD_D);

% Above parameters are good for convolution

% Below parameters are needed for DMA reads
% For deconv: Since Intermediate output after 0 ppadding = (Input - 1)*OG_Stride + 1 . Therefore, to find input reverse the formula
% Input = ceil((Intermediate Output - 1)/OG_Stride) + 1
%       = ceil((33 - 1)/2) + 1
%        = 17
Tiy_R_DECV   = ceil((Tiy_DECV - 1)./STR_DECV) + 1; 
Tiy_DECV_DMA = min(Tiy_R_DECV,NIY_DECV);

if NUM_DECV > 0
    Tiy_DECV_DMA_PAD = zeros(NUM_DECV,1);
    for L = 1:NUM_DECV
       if NOY_DECV(L) == Toy_DECV(L) % Tiy_DECV_DMA does not include PAD_DECV
           Tiy_DECV_DMA_PAD(L) = Tiy_DECV_DMA(L)+PAD_D(L);
       else % Tiy_DMA includes PAD_CONV
           Tiy_DECV_DMA_PAD(L) = Tiy_DECV_DMA(L)       ;
       end
    end
else
    Tiy_DECV_DMA_PAD = [];
end

NIX_R_DECV0 = NIX_DECV0;
NIY_R_DECV0 = NIY_DECV0;
NOX_R_DECV0 = NIX_DECV0;
NOY_R_DECV0 = NIY_DECV0;
NOY_R_DECV  = ceil(NOY_R_DECV0./POY)*POY;
NIY_R_DECV  = max((NOY_R_DECV-1).*STR_D+NKY_DECV-2*PAD_D, NIY_R_DECV0); % max((NOY_R_DECV + 2*PAD_D -NKY_DECV)./STR_D + 1, NIY_R_DECV0);
NOX_R_DECV  = ceil(NOX_R_DECV0./PX_AD).*PX_AD;
NIX_R_DECV  = ceil(NIX_R_DECV0./PX_AD).*PX_AD;
Tox_R_DECV  = NOX_R_DECV;
%Tix_R_DECV  = (Tox_R_DECV + 2*PAD_D - NKY_DECV)./STR_D + 1;   % TODO: Check if we should use this instead: (TOX_R_DECV-1).*STR_D + NKX;
Tix_R_DECV  = (Tox_R_DECV-1).*STR_D + NKX_DECV;
NIX_W_DECV0 = (NIX_DECV0-1).*STR_DECV + 1;
NIY_W_DECV0 = (NIY_DECV0-1).*STR_DECV + 1;
NOX_W_DECV0 = NOX_DECV0;
NOY_W_DECV0 = NOY_DECV0;
NOY_W_DECV  = ceil(NOY_W_DECV0./POY)*POY;
NIY_W_DECV  = max((NOY_W_DECV-1).*STR_D+NKX_DECV-2*PAD_D, NIY_W_DECV0); % max((NOY_W_DECV + 2*PAD_D -NKY_DECV)./STR_D + 1, NIY_W_DECV0);
NOX_W_DECV  = ceil(NOX_W_DECV0./PX_AD).*PX_AD;
NIX_W_DECV  = ceil(NIX_W_DECV0./PX_AD).*PX_AD;
Tox_W_DECV  = NOX_W_DECV;
%Tix_W_DECV  = (Tox_W_DECV + 2*PAD_D - NKY_DECV)./STR_D + 1;   % TODO: Check if we should use this instead: (TOX_W_DECV-1).*STR_D + 1;
Tix_W_DECV  = (Tox_W_DECV-1).*STR_D + NKY_DECV;
%Tiy_W_DECV  = (Toy_W_DECV + 2*PAD_D - NKY_DECV)./STR_D + 1;   % TODO: Check if we should use this instead: (TOY_W_DECV-1).*STR_D + 1;
Tiy_W_DECV = (Toy_W_DECV-1).*STR_D + NKY_DECV;
Tiy_W_DECV = min(Tiy_W_DECV,NIY_W_DECV+2*PAD_D);

% RSP: 
Tiy_R_DECV_DMA = ceil(Tiy_R_DECV./ceil(PX_AD./Tix_R_DECV)).*ceil(PX_AD./Tix_R_DECV);
Tiy_R_DECV_DMA = min(Tiy_R_DECV_DMA, NIY_R_DECV);
Tiy_R_DECV_DMA_PAD  = zeros(NUM_DECV,1);

for L = 1:NUM_DECV
   if NOY_R_DECV(L) == Toy_R_DECV(L) % Tiy_DECV_DMA does not include PAD_DECV
       Tiy_R_DECV_DMA_PAD(L) = Tiy_R_DECV_DMA(L)+PAD_D(L);
   else % Tiy_DMA includes PAD_CONV
       Tiy_R_DECV_DMA_PAD(L) = Tiy_R_DECV_DMA(L);
   end
end

Tiy_W_DECV_DMA = min(Tiy_W_DECV, NIY_W_DECV);
if NUM_DECV > 0
    Tiy_W_DECV_DMA_PAD  = zeros(NUM_DECV,1);
    for L = 1:NUM_DECV
       if NOY_W_DECV(L) == Toy_W_DECV(L) % Tiy_DECV_DMA does not include PAD_DECV
           Tiy_W_DECV_DMA_PAD(L) = Tiy_W_DECV_DMA(L)+PAD_D(L);
       else % Tiy_DMA includes PAD_CONV
           Tiy_W_DECV_DMA_PAD(L) = Tiy_W_DECV_DMA(L);
       end
    end
else
    Tiy_W_DECV_DMA_PAD = [];
end

NOY_W_DECV_WRpx   = ceil(NOY_W_DECV./Toy_W_DECV).*Toy_W_DECV;
NUM_TILE_pDECV    = ceil(NOY_W_DECV./Toy_W_DECV).*ceil(NOF_DECV./Tof_DECV);
NUM_TILE_DECV     = sum(NUM_TILE_pDECV);



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLMX Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%NIX_PLMX = (NOX_PLMX-1).*STR_PLMX+NKX_PLMX-PAD_PLMX-PAD_E_PLMX;
%NIX_PLMX = ceil(NIX_PLMX./PX_AD).*PX_AD;
NIX_PLMX = ceil(NIX_PLMX0./PX_AD).*PX_AD;
NOX_PLMX = ceil(NOX_PLMX0./PX_AD).*PX_AD;
NIY_PLMX = ceil(NIY_PLMX0./POY  ).*POY;
NOY_PLMX = ceil(NOY_PLMX0./Toy_PLMX).*Toy_PLMX;
Tif_PLMX = ceil(Tif_PLMX ./POF_PLMX).*POF_PLMX;
NOF_PLMX = ceil(NOF_PLMX0./Tif_PLMX).*Tif_PLMX;


Tof_PLMX = Tif_PLMX;
Tox_PLMX = NOX_PLMX;
Tix_PLMX = NIX_PLMX;
Tiy_PLMX = min((Toy_PLMX-1).*STR_PLMX+NKY_PLMX, NIY_PLMX);

Toy_PLMX_WRpx = Toy_PLMX;
NOY_PLMX_WRpx = NOY_PLMX;

if NUM_PLMX > 0
    NUM_TILE_pPLMX = ceil(NOY_PLMX./Toy_PLMX).*ceil(NOF_PLMX./Tof_PLMX);
else
    NUM_TILE_pPLMX = 0;
end
NUM_TILE_PLMX = sum(NUM_TILE_pPLMX); % POOL_MAX only


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEAR Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NIX_NEAR = ceil(NIX_NEAR0./PX_AD).*PX_AD;
NOX_NEAR = ceil(NOX_NEAR0./PX_AD).*PX_AD;
NIY_NEAR = ceil(NIY_NEAR0./POY  ).*POY;
NOY_NEAR = ceil(NOY_NEAR0./Toy_NEAR).*Toy_NEAR;
Tif_NEAR = ceil(Tif_NEAR ./POF_NEAR).*POF_NEAR;
NOF_NEAR = ceil(NOF_NEAR0./Tif_NEAR).*Tif_NEAR;


Tof_NEAR = Tif_NEAR;
Tox_NEAR = NOX_NEAR;
Tix_NEAR = NIX_NEAR;
Tiy_NEAR = min(Toy_NEAR/STR_NEAR, NIY_NEAR);

Toy_NEAR_WRpx = Toy_NEAR;
NOY_NEAR_WRpx = NOY_NEAR;

if NUM_NEAR > 0
    NUM_TILE_pNEAR = ceil(NOY_NEAR./Toy_NEAR).*ceil(NOF_NEAR./Tof_NEAR);
else
    NUM_TILE_pNEAR = 0;
end
NUM_TILE_NEAR = sum(NUM_TILE_pNEAR);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAPL Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NOF_GAPL = ceil(NOF_GAPL0./PX_AD).*PX_AD;
NIX_GAPL = ceil(NIX_GAPL0./PX_AD).*PX_AD;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROP Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NIX_PROP0 = zeros(NUM_PROP,1); % Input CONV (bbox/score) Feature Map Width
NIY_PROP0 = zeros(NUM_PROP,1); % Input CONV (bbox/score) Feature Map Height

ID_PROP_bbox  = zeros(NUM_PROP,1);
ID_PROP_score = zeros(NUM_PROP,1);
%num_anchors   = zeros(NUM_PROP,1);
for L = 1:NUM_PROP
    
     input1_ID = input_layers_ID{ID_global_PROP(L)}(1);
     input2_ID = input_layers_ID{ID_global_PROP(L)}(2);
     
     if NOX_LAYER0(input1_ID) == NOX_LAYER0(input2_ID) && NOY_LAYER0(input1_ID) == NOY_LAYER0(input2_ID)
         NIX_PROP0(L) = NOX_LAYER0(input1_ID);
         NIY_PROP0(L) = NOY_LAYER0(input1_ID);
     else
         fprintf('Warning @ Variants.m: bbox and score sizes of PROP do NOT match! \n\n')
     end

     %num_anchors(L) = min(NOF_LAYER0(input1_ID), NOF_LAYER0(input2_ID));
     if NOF_LAYER0(input1_ID) > NOF_LAYER0(input2_ID)
         ID_PROP_bbox(L)  = input1_ID;
         ID_PROP_score(L) = input2_ID;
         if NOF_LAYER0(input1_ID)/NOF_LAYER0(input2_ID) ~= 4
             fprintf('Warning @ Variants.m: num_anchors in bbox and score do NOT match! \n\n')
         end
     end
     if NOF_LAYER0(input1_ID) <= NOF_LAYER0(input2_ID)
         ID_PROP_bbox(L)  = input2_ID;
         ID_PROP_score(L) = input1_ID;
         if NOF_LAYER0(input2_ID)/NOF_LAYER0(input1_ID) ~= 4
             fprintf('Warning @ Variants.m: num_anchors in bbox and score do NOT match! \n\n')
         end
     end   
end

NIX_PROP = ceil(NIX_PROP0./PX_AD).*PX_AD;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% EWIS Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NIX_EWIS = ceil(NIX_EWIS0./PX_AD).*PX_AD;
NOX_EWIS = ceil(NOX_EWIS0./PX_AD).*PX_AD;
NOY_EWIS = ceil(NOY_EWIS0./Toy_EWIS).*Toy_EWIS;
NOF_EWIS = ceil(NOF_EWIS0/(BUF_INPX_1BK/2))*(BUF_INPX_1BK/2);

Tif_EWIS = NIF_EWIS0;
Tof_EWIS = NOF_EWIS0;
Tiy_EWIS = Toy_EWIS;
%Tix_EWIS = NIX_EWIS;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROIP Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Toy_ROIP = NOY_ROIP0; % ROIP alwyas read one whole feature map
Tbx_ROIP = NBX_ROIP0; % ROIP always processes NBX_ROIP0 boxes in one tile

NOX_ROIP = ceil(NOX_ROIP0./POX).*POX;
NOY_ROIP = NOY_ROIP0;
NOF_ROIP = ceil(NOF_ROIP0./Tof_ROIP).*Tof_ROIP;
NBX_ROIP = ceil(NBX_ROIP0./Tbx_ROIP).*Tbx_ROIP;
for L = 1:NUM_ROIP
    if NOX_ROIP(L) > 2*POX
        fprintf('Warning @ Variants.m: DLA cannot support NOX_ROIP(L) > 2*POX for the following FCON\n\n')
    end
    if mod(NOX_ROIP(L)*NOY_ROIP(L)*Tof_ROIP(L)/PX_AD,1)~=0
        fprintf('Error @ Variants.m: Tof_ROIP need to align the ROIP output to DMA_width\n\n')
        Error
    end
end

NIX_ROIP0 = zeros(NUM_ROIP,1);
NIY_ROIP0 = zeros(NUM_ROIP,1);
for C = 1:NUM_ROIP
    ROIP_ID = ID_global_ROIP(C);
    for ii = 1:length(input_layers_ID{ROIP_ID})
        input_ID = input_layers_ID{ROIP_ID}(ii);
        if CR_LAYER_IS_PROP(input_ID) ~= 1
            NIX_ROIP0(C) = NOX_LAYER0(input_ID);
            NIY_ROIP0(C) = NOY_LAYER0(input_ID);
        end
    end
end
NIX_ROIP = ceil(NIX_ROIP0./PX_AD).*PX_AD;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% FCON Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Tof_FCON = POF_FCON;
if NUM_FCON > 0
    for L = 1:NUM_FCON
        if Tif_FCON(L) > (BUF_WEIGHT_WORDS*4)/(Tof_FCON/POF)
            Tif_FCON(L) = BUF_WEIGHT_WORDS*4;
            fprintf('Warning @Variants.m: Tif_FCON(%d) is larger than the weight buffer depth! \n\n',L)
        end
%         if Tif_FCON(L) < 64
%             Tif_FCON(L) = 64;
%             fprintf('Warning @Variants.m: Tif_FCON(%d) and NIF_FCON is too small and set to 64! \n\n',L)            
%         end
    end
else
    Tif_FCON = [];
end

NIF_FCON = ceil(NIF_FCON0./Tif_FCON).*Tif_FCON;
NOF_FCON = ceil(NOF_FCON0./POF_FCON).*POF_FCON;
NBX_FCON = ceil(NBX_FCON0./Tbx_FCON).*Tbx_FCON;


%% Buffer Utilization

% %%%%%%%%%%%%%%%%%%%%%%%%% CONV %%%%%%%%%%%%%%%%%%%%%%%%%
BUF_INPUT_width_CONV = POX; 
BUF_INPUT_num_CONV   = BUF_INPX_ALL;
BUF_INPUT_depth_CONV = (NIX_CONV/BUF_INPUT_width_CONV).*ceil((ceil(Tiy_CONV_DMA_PAD./STR_CONV)/BUF_INPUT_num_CONV)).*STR_CONV.*ceil(NIF_CONV);
depth_INPUT_CONV = max(BUF_INPUT_depth_CONV)/RAM_BK;
for LL = 1:NUM_CONV
    if BUF_INPUT_depth_CONV(LL)/RAM_BK > BUF_INPUT_WORDS
        fprintf('Error: @CONV(%d) the input buffer is NOT large enough for CONV (BUF_INPUT_WORDS)\n\n',LL)
        %Error
    end
end

BUF_OUTPUT_width_CONV = POX;
BUF_OUTPUT_num_CONV   = BUF_OUPX_1BK;
BUF_OUTPUT_depth_CONV = ceil((Tox_CONV.*Toy_CONV)/BUF_OUTPUT_width_CONV).*ceil(Tof_CONV/BUF_OUTPUT_num_CONV);
depth_OUTPUT_CONV = max(BUF_OUTPUT_depth_CONV)/RAM_BK;
for LL = 1:NUM_CONV
    if BUF_OUTPUT_depth_CONV(LL)/RAM_BK > BUF_OUTPUT_WORDS
        fprintf('Error: @CONV(%d) the output buffer is NOT large enough for CONV (BUF_OUTPUT_WORDS)\n\n',LL)
        Error
    end
end

BUF_WEIGHT_width_CONV = POF;
BUF_WEIGHT_num_CONV   = 1;
BUF_WEIGHT_depth_CONV = NIF_CONV.*NKX_CONV.*NKY_CONV.*ceil(Tof_CONV/BUF_WEIGHT_width_CONV);
depth_WEIGHT_CONV = max(BUF_WEIGHT_depth_CONV)/RAM_BK;
for LL = 1:NUM_CONV
    if BUF_WEIGHT_depth_CONV(LL)/RAM_BK > BUF_WEIGHT_WORDS
        fprintf('Error: @CONV(%d) the weight buffer is NOT large enough for CONV (BUF_WEIGHT_WORDS)\n\n',LL)
        %Error
    end
end


% %%%%%%%%%%%%%%%%%%%%%%%%% DECV %%%%%%%%%%%%%%%%%%%%%%%%%
BUF_INPUT_width_DECV = POX; 
BUF_INPUT_num_DECV   = BUF_INPX_ALL;
BUF_INPUT_depth_DECV = (NIX_DECV/BUF_INPUT_width_DECV).*ceil((ceil(Tiy_DECV_DMA_PAD/1)/BUF_INPUT_num_DECV)).*1.*ceil(NIF_DECV);
depth_INPUT_DECV = max(BUF_INPUT_depth_DECV)/RAM_BK;
if depth_INPUT_DECV > BUF_INPUT_WORDS
    fprintf('Error: the input buffer is NOT large enough for DECV (BUF_INPUT_WORDS)\n\n')
    Error
end

BUF_OUTPUT_width_DECV = POX;
BUF_OUTPUT_num_DECV   = BUF_OUPX_1BK;
BUF_OUTPUT_depth_DECV = ceil((Tox_DECV.*Toy_DECV)/BUF_OUTPUT_width_DECV).*ceil(Tof_DECV/BUF_OUTPUT_num_DECV);
depth_OUTPUT_DECV = max(BUF_OUTPUT_depth_DECV)/RAM_BK;
if depth_OUTPUT_DECV > BUF_OUTPUT_WORDS
    fprintf('Error: the output buffer is NOT large enough for DECV (BUF_OUTPUT_WORDS)\n\n')
    Error
end

BUF_WEIGHT_width_DECV = POF;
BUF_WEIGHT_num_DECV   = 1;
BUF_WEIGHT_depth_DECV = NIF_DECV.*NKX_DECV.*NKY_DECV.*ceil(Tof_DECV/BUF_WEIGHT_width_DECV);
depth_WEIGHT_DECV = max(BUF_WEIGHT_depth_DECV)/RAM_BK;
if depth_WEIGHT_DECV > BUF_WEIGHT_WORDS
    fprintf('Error: the weight buffer is NOT large enough for DECV (BUF_WEIGHT_WORDS)\n\n')
    Error
end


% %%%%%%%%%%%%%%%%%%%%%%%%% PLMX %%%%%%%%%%%%%%%%%%%%%%%%%
BUF_INPUT_width_PLMX = POX;
BUF_INPUT_num_PLMX   = POF_PLMX;
BUF_INPUT_depth_PLMX = ceil(Tix_PLMX./BUF_INPUT_width_PLMX).*Tiy_PLMX.*ceil(Tif_PLMX/BUF_INPUT_num_PLMX); 
depth_INPUT_PLMX = max(BUF_INPUT_depth_PLMX)/RAM_BK;
if depth_INPUT_PLMX > BUF_INPUT_WORDS
    fprintf('Error: the input buffer is NOT large enough for PLMX (BUF_INPUT_WORDS)\n\n')
    Error
end

BUF_OUTPUT_width_PLMX = POX;
BUF_OUTPUT_num_PLMX   = POF_PLMX;
BUF_OUTPUT_depth_PLMX = ceil(Tox_PLMX./BUF_OUTPUT_width_PLMX).*ceil(Tof_PLMX/BUF_OUTPUT_num_PLMX).*Toy_PLMX_WRpx;
depth_OUTPUT_PLMX = max(BUF_OUTPUT_depth_PLMX)/RAM_BK;
if depth_OUTPUT_PLMX > BUF_OUTPUT_WORDS
    fprintf('Error: the output buffer is NOT large enough for PLMX (BUF_OUTPUT_WORDS)\n\n')
    Error
end


% %%%%%%%%%%%%%%%%%%%%%%%%% EWIS %%%%%%%%%%%%%%%%%%%%%%%%%
BUF_INPUT_width_EWIS = POX;
BUF_INPUT_num_EWIS   = BUF_INPX_ALL/2;
BUF_INPUT_depth_EWIS = ceil(NIX_EWIS./BUF_INPUT_width_EWIS).*Toy_EWIS.*ceil(Tif_EWIS/BUF_INPUT_num_EWIS); 
depth_INPUT_EWIS = max(BUF_INPUT_depth_EWIS)/RAM_BK;
if depth_INPUT_EWIS > BUF_INPUT_WORDS
    fprintf('Error: the input buffer is NOT large enough for EWIS (BUF_INPUT_WORDS)\n\n')
    %Error
end


% %%%%%%%%%%%%%%%%%%%%%%%%% ROIP %%%%%%%%%%%%%%%%%%%%%%%%%
BUF_INPUT_width_ROIP = POX;
BUF_INPUT_num_ROIP   = BUF_INPX_ALL;
BUF_INPUT_depth_ROIP = ceil(NIX_ROIP./BUF_INPUT_width_ROIP).*NIY_ROIP0.*ceil(Tof_ROIP/BUF_INPUT_num_ROIP); 
depth_INPUT_ROIP = max(BUF_INPUT_depth_ROIP)/RAM_BK;
if depth_INPUT_ROIP > BUF_INPUT_WORDS
    fprintf('Error: the input buffer is NOT large enough for ROIP (BUF_INPUT_WORDS)\n\n')
    Error
end

BUF_OUTPUT_width_ROIP = POX;
BUF_OUTPUT_num_ROIP   = BUF_OUPX_ALL;
BUF_OUTPUT_depth_ROIP = ceil(NOX_ROIP./BUF_OUTPUT_width_ROIP).*NOY_ROIP0.*ceil(NBX_ROIP0/BUF_OUTPUT_num_ROIP).*Tof_ROIP; 
depth_OUTPUT_ROIP = max(BUF_OUTPUT_depth_ROIP)/RAM_BK;
if depth_OUTPUT_ROIP > BUF_OUTPUT_WORDS
    fprintf('Error: the output buffer is NOT large enough for ROIP (BUF_OUTPUT_WORDS)\n\n')
    Error
end


% %%%%%%%%%%%%%%%%%%%%%%%%% FCON read from ROIP %%%%%%%%%%%%%%%%%%%%%%%%%
if NUM_ROIP > 0
    BUF_INPUT_width_FCON = POX;
    BUF_INPUT_num_FCON   = BUF_INPX_ALL;
    BUF_INPUT_depth_FCON = ceil(NOX_ROIP.*NOY_ROIP.*NOF_ROIP./BUF_INPUT_width_FCON).*ceil(NUM_FC_BOX/BUF_INPUT_num_FCON); 
    depth_INPUT_FCON = max(BUF_INPUT_depth_FCON)/RAM_BK;
    if depth_INPUT_FCON > 2*BUF_INPUT_WORDS % use input buffer in combined mode
        fprintf('Error: the input buffer is NOT large enough for FCON with inputs from ROIP (BUF_INPUT_WORDS)\n\n')
        Error
    end
end


if NUM_CONV > 0
    fprintf('The input  buffer depth requirement of CONV is %d <= %d.\n', depth_INPUT_CONV, BUF_INPUT_WORDS);
    fprintf('The output buffer depth requirement of CONV is %d <= %d.\n', depth_OUTPUT_CONV,BUF_OUTPUT_WORDS);
    fprintf('The weight buffer depth requirement of CONV is %d <= %d.\n', depth_WEIGHT_CONV,BUF_WEIGHT_WORDS);
end
if NUM_DECV > 0
    fprintf('The input  buffer depth requirement of DECV is %d <= %d.\n', depth_INPUT_DECV, BUF_INPUT_WORDS);
    fprintf('The output buffer depth requirement of DECV is %d <= %d.\n', depth_OUTPUT_DECV,BUF_OUTPUT_WORDS);
    fprintf('The weight buffer depth requirement of DECV is %d <= %d.\n', depth_WEIGHT_DECV,BUF_WEIGHT_WORDS);
end
if NUM_PLMX > 0
    fprintf('The input  buffer depth requirement of PLMX is %d <= %d.\n', depth_INPUT_PLMX, BUF_INPUT_WORDS);
    fprintf('The output buffer depth requirement of PLMX is %d <= %d.\n', depth_OUTPUT_PLMX,BUF_OUTPUT_WORDS);
end
if NUM_EWIS > 0
    fprintf('The input  buffer depth requirement of EWIS is %d <= %d.\n', depth_INPUT_EWIS, BUF_INPUT_WORDS);
    %fprintf('The output buffer depth requirement of EWIS is %d <= %d.\n', depth_INPUT_EWIS, BUF_OUTPUT_WORDS);
end
if NUM_ROIP > 0
    fprintf('The input  buffer depth requirement of ROIP is %d <= %d.\n', depth_INPUT_ROIP, BUF_INPUT_WORDS);
    fprintf('The output buffer depth requirement of ROIP is %d <= %d.\n', depth_OUTPUT_ROIP, BUF_OUTPUT_WORDS);
end

fprintf('\n');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Check if the CNN sizes and design variables voilates the DLA limiations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (BUF_INPX_ALL < BUF_OUPX_ALL)
    fprintf('Error @ Variants: Eltwise input buffer bank not engouth !\n\n');
    Error
end

for L = 1:NUM_LAYER
    if CR_LAYER_IS_CONV(L) + CR_LAYER_IS_DECV(L) + CR_LAYER_IS_NEAR(L) + CR_LAYER_IS_PLMX(L) + CR_LAYER_IS_GAPL(L) + CR_LAYER_IS_ROIP(L) + CR_LAYER_IS_PROP(L) + CR_LAYER_IS_EWIS(L) + CR_LAYER_IS_FCON(L) ~= 1
        fprintf('Error @ Variants: the layer type of layer %d is NOT correct!\n\n',L);
        Error
    end
end

if mod(log2(DMA_WIDTH/(POF*WD_WT)), 1)~= 0
    fprintf('Error @ Variants: DMA_WIDTH/(POF*WD_WT) or (POF*WD_WT)/DMA_WIDTH must be power of 2 !\n\n');
    Error
end

if NUM_CONV>0
    need_bias_CONV_words = ceil(sum(NOF_CONV.*CR_CONV_with_Bias)/BUF_OUPX_ALL);
    if need_bias_CONV_words > BUF_BIAS_CONV_WORDS
        fprintf('Error @Variants.m: BUF_BIAS_CONV_WORDS is NOT large enough store all the CONV biases! \n\n')
        %Error   
    end
end
if NUM_FCON>0
    need_bias_FCON_words = ceil(sum(NOF_FCON.*CR_FCON_with_Bias)/POX);
    if need_bias_FCON_words > BUF_BIAS_FCON_WORDS
        fprintf('Error @Variants.m: BUF_BIAS_FCON_WORDS is NOT large enough store all the FCON biases! \n\n')
        %Error   
    end
end

for L = 1:NUM_PLMX
    if STR_PLMX(L) == 1
        if PAD_PLMX(L) ~= 1
            fprintf('Error @Variants.m: if STR_PLMX = 1, PAD_PLMX must be 1, while PAD_PLMX(%d) = %d \n\n',L,PAD_PLMX(L))
            Error             
        end
        if NKX_PLMX(L) > 4 || NKY_PLMX(L) > 4
            fprintf('Error @Variants.m: NKX_PLMX and NKY_PLMX must <= 4 !\n\n')
            Error             
        end
        if Toy_PLMX(L) < NOY_PLMX(L)
            fprintf('Error @Variants.m: if STR_PLMX = 1, Toy_PLMX must equal to NOY_PLMX, while Toy_PLMX(%d) = %d with NOY_PLMX(%d) = %d.\n\n', L,Toy_PLMX(L),L,NOY_PLMX(L))
            %Error             
        end
    elseif STR_PLMX(L) == 2
        if PAD_PLMX(L) ~= 0
            fprintf('Error @Variants.m: if STR_PLMX = 2, PAD_PLMX must be 0, while PAD_PLMX(%d) = %d \n\n',L,PAD_PLMX(L))
            Error             
        end
    else
            fprintf('Error @Variants.m: STR_PLMX must be 1 or 2 \n\n')
            Error    
    end
end

if NKI_CONV0 ~= NIF_CONV0
    fprintf('Error @Variants.m: NKI_CONV0 ~= NIF_CONV0, group of CONV must be 1! \n\n')
    Error  
end

if NUM_PROP > 1
    fprintf('Warnining: DLA can only support one proposal layer in one CNN model! \n\n')
end





