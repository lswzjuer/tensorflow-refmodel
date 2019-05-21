import numpy as np

def read_show(path_,output_path):
    npz_file=np.load(path_,allow_pickle=True)
    print(npz_file.items())
    new_dict={}
    for lay_name in npz_file:
        sample_dict={}
        weight=npz_file[lay_name][()]['weights']
        bias=npz_file[lay_name][()]['biases']
        sample_dict['weights']=weight
        sample_dict['biases']=bias
        new_dict[lay_name]=sample_dict
        print('name: {}  weights hsape: {}  biases shape: {}'.format(lay_name,weight.shape,bias.shape))
    # print(new_dict)
    with open(output_path,'wb') as f :
        np.save(f,new_dict)
    new_filesnpy=np.load(output_path)
    print(new_filesnpy)

if __name__=="__main__":
    ssd_path=r'G:\codeing\Fabu\tf-reference-model\lib\model\ssd\0515_obj_params.npz'
    out_path=r'G:\codeing\Fabu\tf-reference-model\lib\model\ssd\obj_params_change.npz'
    #data_path=r'G:\codeing\Fabu\Bosch\Bosch_code\03_benchmark\03_code\preprocess\ssd\obj_params.npz'
    #out_path=r'G:\codeing\Fabu\Bosch\Bosch_code\03_benchmark\03_code\preprocess\ssd\obj_params_change.npz'
    fcn8path=r'G:\codeing\Fabu\tf-reference-model\lib\model\fcn8_seg\seg_params.npz'
    fcn8path_quantized=r'G:\codeing\Fabu\tf-reference-model\lib\model\fcn8_seg\seg_params_change.npz'

    act_=r'G:\codeing\Fabu\tf-reference-model\lib\model\rcnn\act_params.npz'
    act_quantized=r'G:\codeing\Fabu\tf-reference-model\lib\model\rcnn\act_params_change.npz'
    read_show(act_,act_quantized)


