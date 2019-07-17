%Formated descriptor RAM dumps for ASIC

%Bank_depth_DMA_descriptor_WR_CV = 8192;
%Bank_depth_DMA_descriptor_RD_CV = 8192;
%Bank_depth_DMA_descriptor_WR_PL = 4096;
%Bank_depth_DMA_descriptor_RD_PL = 4096;
%Bank_depth_PTILE_descriptor_size_LUT = 4096;
%
%%Create directory if it doesn't exist
%if ~exist('RAM_initialzation', 'dir')
%  mkdir('RAM_initialization');
%end
%%Clear previous results
%delete('./RAM_initialization/*');
%
%count = 0;
%bank_count = 0; %Word counter within each bank
%bank_id = 0; %Bank counter
%file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_WR_CV_bank%d.bin',bank_id);
%fid = fopen(file_name,'w');
%for i=1:NUM_DPT_WR_CONV
%  count = count + 1;
%  fprintf(fid,DPT_WR_CV_WRaddr{count});
%  fprintf(fid,'\n');
%  if (bank_count == Bank_depth_DMA_descriptor_WR_CV - 1)
%    fprintf('RAM_DMA_descriptor_WR_CV_bank%d.bin generated\n\n',bank_id);
%    bank_id = bank_id + 1;
%    bank_count = 0;
%    fclose(fid);
%    file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_WR_CV_bank%d.bin',bank_id);
%    fid = fopen(file_name,'w');
%  else
%    bank_count = bank_count + 1;
%  end
%end
%fclose(fid);
%
%count = 0;
%bank_count = 0; %Word counter within each bank
%bank_id = 0; %Bank counter
%file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_RD_CV_bank%d.bin',bank_id);
%fid = fopen(file_name,'w');
%for i=1:NUM_DPT_RD_CONV
%  count = count + 1;
%  fprintf(fid,DPT_RD_CV_RDaddr{count});
%  fprintf(fid,'\n');
%  if (bank_count == Bank_depth_DMA_descriptor_RD_CV - 1)
%    fprintf('RAM_DMA_descriptor_RD_CV_bank%d.bin generated\n\n',bank_id);
%    bank_id = bank_id + 1;
%    bank_count = 0;
%    fclose(fid);
%    file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_RD_CV_bank%d.bin',bank_id);
%    fid = fopen(file_name,'w');
%  else
%    bank_count = bank_count + 1;
%  end
%end
%fclose(fid);
%
%count = 0;
%bank_count = 0; %Word counter within each bank
%bank_id = 0; %Bank counter
%file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_WR_PL_bank%d.bin',bank_id);
%fid = fopen(file_name,'w');
%for i=1:NUM_DPT_WR_PLMX
%  count = count + 1;
%  fprintf(fid,DPT_WR_PLMX_WRaddr{count});
%  fprintf(fid,'\n');
%  if (bank_count == Bank_depth_DMA_descriptor_WR_PL - 1)
%    fprintf('RAM_DMA_descriptor_WR_PL_bank%d.bin generated\n\n',bank_id);
%    bank_id = bank_id + 1;
%    bank_count = 0;
%    fclose(fid);
%    file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_WR_PL_bank%d.bin',bank_id);
%    fid = fopen(file_name,'w');
%  else
%    bank_count = bank_count + 1;
%  end
%end
%fclose(fid);
%
%count = 0;
%bank_count = 0; %Word counter within each bank
%bank_id = 0; %Bank counter
%file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_RD_PL_bank%d.bin',bank_id);
%fid = fopen(file_name,'w');
%for i=1:NUM_DPT_RD_PLMX
%  count = count + 1;
%  fprintf(fid,DPT_RD_PLMX_RDaddr{count});
%  fprintf(fid,'\n');
%  if (bank_count == Bank_depth_DMA_descriptor_RD_PL - 1)
%    fprintf('RAM_DMA_descriptor_RD_PL_bank%d.bin generated\n\n',bank_id);
%    bank_id = bank_id + 1;
%    bank_count = 0;
%    fclose(fid);
%    file_name = sprintf('./RAM_initialization/RAM_DMA_descriptor_RD_PL_bank%d.bin',bank_id);
%    fid = fopen(file_name,'w');
%  else
%    bank_count = bank_count + 1;
%  end
%end
%fclose(fid);
%
%%Formated descriptor size RAM dumps for ASIC
%fid = fopen('./RAM_initialization/RAM_PTILE_WR_CV_descriptor_size_LUT.bin','w');
%%JH: Remove first layer 
%for t = 1:NUM_TILE_CONV
%    fprintf(fid,dec2bin(NUM_dpt_WR_pTL_CV(t)));
%    fprintf(fid,'\n');
%end
%fclose(fid);
%
%fid = fopen('./RAM_initialization/RAM_PTILE_RD_CV_descriptor_size_LUT.bin','w');
%for t = 1:NUM_TILE_CONV
%    fprintf(fid,dec2bin(NUM_dpt_RD_pTL_CV(t)));
%    fprintf(fid,'\n');
%end
%fclose(fid);
%
%fid = fopen('./RAM_initialization/RAM_PTILE_WR_PL_descriptor_size_LUT.bin','w');
%for t = 1:NUM_TILE_PLMX
%    fprintf(fid,dec2bin(NUM_dpt_WRpx_pTL_PLMX(t)));
%    fprintf(fid,'\n');
%end
%fclose(fid);
%
%fid = fopen('./RAM_initialization/RAM_PTILE_RD_PL_descriptor_size_LUT.bin','w');
%for t = 1:NUM_TILE_PLMX
%    fprintf(fid,dec2bin(NUM_dpt_RDpx_pTL_PLMX(t)));
%    fprintf(fid,'\n');
%end
%fclose(fid);
%
%%Generate CONV bias binary file
%fid = fopen('./RAM_initialization/bias_CONV.bin','w');
%for i=1:row_BSCV
%        bin_tmp='';
%        for ii=1:col_BSCV
%            bin_tmp = [dec2bin(conv_BS_2s(i,ii),WD_BSCV),bin_tmp];
%        end
%        fprintf(fid,bin_tmp);
%        fprintf(fid,'\n');
%end
%fclose(fid);  
%
%
%%Generate FC bias binary file
%for L=1:NUM_FCON
%    file_name = sprintf('./RAM_initialization/bias_FCON%d.bin',L);
%    fid = fopen(file_name,'w');
%    for j=1:NOF_FCON0(L)
%        %Convert to 2's complement
%        if BS_fc{L}(j)<0
%          BS_fc_2s = BS_fc{L}(j) + 2^WD_BSFC;
%        else
%          BS_fc_2s = BS_fc{L}(j);
%        end
%        fprintf(fid,dec2bin(BS_fc_2s));
%        fprintf(fid,'\n');
%    end
%    fclose(fid);
%end
%
fid = fopen('./DMA_LUT.bin','w');
bin_tmp="";
for L=1:NUM_LAYER
    bin_tmp=strcat(dec2bin(lut_offset_if(L),32),dec2bin(lut_inpx_addr(L),32));
    bin_tmp=strcat(dec2bin(lut_nif(L),16),bin_tmp);
    fprintf(fid,bin_tmp);
    fprintf(fid,'\n');
end
fclose(fid)
