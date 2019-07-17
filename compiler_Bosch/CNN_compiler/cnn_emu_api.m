

r = write_json("Tof.json", Tof_CONV);
% % r = write_json("DDR3_BDEC_KN_CONV1.json", DDR3_BDEC_KN_CONV1);
% % r = write_json("DDR3_BADDR_KN_FC6.json", hex2dec(DDR3_BADDR_KN_FC6));
% % r = write_json("DDR3_BADDR_PROP.json", hex2dec(DDR3_BADDR_PROP));
% % r = write_json("DDR3_BADDR_IMAGE.json", hex2dec(DDR3_BADDR_IMAGE));
% % r = write_json("NOY_WRPX_BBOX.json", NOY_WRPX_BBOX);
% % r = write_json("TILING_BYTES_FC.json", TILING_BYTES_FC);
% % 
r = write_json("IS_CONV.json", CR_LAYER_IS_CONV);
r = write_json("IS_PLMX.json", CR_LAYER_IS_PLMX);

r = write_json("LAYER_local_ID.json", ID_local_LAYER);

r = write_json("NUM_TILE_pCONV.json", NUM_TILE_pCONV);
r = write_json("NUM_TILE_pPLMX.json", NUM_TILE_pPLMX);
r = write_json("NKX.json", NKX_CONV);

r = write_json("NIX_LAYER0.json", NIX_LAYER0);

r = write_json("NIX.json", NIX_CONV);
r = write_json("NIY.json", NIY_CONV);
r = write_json("NOX.json", NOX_CONV);
r = write_json("NOY.json", NOY_CONV);
r = write_json("NOF.json", NOF_CONV);
r = write_json("NIF.json", NIF_CONV);

r = write_json("Toy.json", Toy_CONV);
r = write_json("Toy_PL.json", Toy_PLMX);

r = write_json("NIX_PL.json", NIX_PLMX);
r = write_json("NIY_PL.json", NIY_PLMX);
r = write_json("NOX_PL.json", NOX_PLMX);
r = write_json("NOY_PL.json", NOY_PLMX);
r = write_json("NOF_PL.json", NOF_PLMX);
r = write_json("NIF_PL.json", NIF_PLMX0);

r = write_json("lut_nif.json", lut_nif);
r = write_json("lut_inpx_addr.json", lut_inpx_addr);
r = write_json("lut_offset_if.json", lut_offset_if);
r = write_json("inpx_num_layer.json", CR_LAYER_inpx_num_layer);
r = write_json("inpx_cmd_size.json", CR_LAYER_inpx_cmd_size);
r = write_json("offset_tiy.json", CR_LAYER_offset_tiy);

r = write_json("r_inpx_layer_id.json", R_inpx_layer_id);
r = write_json("inwt_addr.json", CR_LAYER_inwt_addr);
r = write_json("inwt_cmd_size.json", CR_LAYER_inwt_cmd_size);
r = write_json("offset_wt_tof.json", CR_LAYER_offset_wt_tof);

r = write_json("outpx_addr.json", CR_LAYER_outpx_addr);
r = write_json("outpx_cmd_size.json", CR_LAYER_outpx_cmd_size);
r = write_json("offset_toy.json", CR_LAYER_offset_toy);
r = write_json("offset_of.json", CR_LAYER_offset_of);

r = write_json("tof.json", CR_LAYER_tof);
r = write_json("num_tof.json", CR_LAYER_num_tof);
r = write_json("num_toy.json", CR_LAYER_num_toy);

r = write_json("Tiy.json", Tiy_CONV);
r = write_json("Tiy_PL.json", Tiy_PLMX);
r = write_json("Tof.json", Tof_CONV);
r = write_json("Tof_PL.json", Tof_PLMX);
r = write_json("STR.json", STR_CONV);
r = write_json("STR_PL.json", STR_PLMX);
r = write_json("PAD.json", PAD_CONV);
r = write_json("PAD_PL.json", PAD_PLMX);


% % r = write_json("DDR3_BDEC_RDWT_pCV.json", DDR3_BDEC_RDWT_pCV);
% % r = write_json("DDR3_BDEC_RDWT_pCV.json",DDR3_BDEC_RDWT_pCV);

% % r = write_json("DDR3_BDEC_RDPX_pCVpM.json",DDR3_BDEC_RDPX_pCVpM);
% % r = write_json("DDR3_BDEC_RDPX_pPLpM.json",DDR3_BDEC_RDPX_pPLpM);

% % r = write_json("DDR3_BDEC_WRPX_pCVpM.json",DDR3_BDEC_WRPX_pCVpM);
% % r = write_json("DDR3_BDEC_WRPX_pPLpM.json",DDR3_BDEC_WRPX_pPLpM);

% % r = write_json("DDR3_BDEC_RDPX_pTL.json",DDR3_BDEC_RDPX_pTL);

% % r = write_json("DDR3_BDEC_RDWT_pTLpCV.json",DDR3_BDEC_RDWT_pTLpCV);

% % r = write_json("OUpxMap_pCV_Bytes.json",OUpxMap_pCV_Bytes);

% % r = write_json("R_LEN_RDPX_CV.json",R_LEN_RDPX_CV);
% % r = write_json("R_LEN_RDWT_CV.json",R_LEN_RDWT_CV);
% % r = write_json("R_LEN_WRPX_CV.json",R_LEN_WRPX_CV);
% % r = write_json("R_LEN_RDPX_PL.json",R_LEN_RDPX_PL);
% % r = write_json("R_LEN_WRPX_PL.json",R_LEN_WRPX_PL);

% % r = write_json("NUM_dpt_RD_pTL_CV.json",NUM_dpt_RD_pTL_CV);
% % r = write_json("NUM_dpt_RDwt_pTL.json",NUM_dpt_RDwt_pTL);
% % r = write_json("NUM_dpt_WR_pTL_CV.json",NUM_dpt_WR_pTL_CV);
% % r = write_json("NUM_dpt_RDpx_pTL_PL.json",NUM_dpt_RDpx_pTL_PL);
% % r = write_json("NUM_dpt_WRpx_pTL_PL.json",NUM_dpt_WRpx_pTL_PL);

% % r = write_json("RDaddr_pDPT_RD_CV.json",RDaddr_pDPT_RD_CV);
% % r = write_json("RDaddr_pDPT_RD_PL.json",RDaddr_pDPT_RD_PL);
% % r = write_json("WRaddr_pDPT_WR_CV.json",WRaddr_pDPT_WR_CV);
% % r = write_json("WRaddr_pDPT_WR_PL.json",WRaddr_pDPT_WR_PL);

% % r = write_json("NUM_TILE_DMA_pFCON.json",NUM_TILE_DMA_pFCON);

% % %r = write_json("DDR3_BDEC_RDPX_pTLpDpCV.json",DDR3_BDEC_RDPX_pTLpDpCV);

% % r = write_json("Bytes_RDpx_pTL.json",Bytes_RDpx_pTL);
% % r = write_json("Bytes_RDwt_pTL.json",Bytes_RDwt_pTL);
% % r = write_json("Bytes_RDpx_pTL_PL.json",Bytes_RDpx_pTL_PL);

% % r = write_json("NUM_ANCHOR.json",NUM_ANCHOR);

r = write_json("DDR3_ENDADDR_WT.json",REG_DDR3_ENDADDR_WT);
