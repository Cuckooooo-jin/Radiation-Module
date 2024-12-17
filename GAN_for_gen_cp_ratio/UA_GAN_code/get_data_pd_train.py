import pandas as pd
import numpy as np
from enum import Enum
import pickle
##pudong index:

class OutputType(Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"

class Normalization(Enum):
    ZERO_ONE = "ZERO_ONE"
    MINUSONE_ONE = "MINUSONE_ONE"

class Output(object):
    def __init__(self, type_, dim, normalization=None, is_gen_flag=False):
        self.type_ = type_
        self.dim = dim
        self.normalization = normalization
        self.is_gen_flag = is_gen_flag

        if type_ == OutputType.CONTINUOUS and normalization is None:
            raise Exception("normalization must be set for continuous output")

def get_cloudtype():
    ori_df = pd.read_csv("D:/yanjiusheng/pythonProject/pudong1620.csv")
    dni = ori_df.loc[:,["DNI"]].values
    cldni = ori_df.loc[:,["Clearsky DNI"]].values
    cp =ori_df.loc[:,["Cloud Type"]].values
    all_days = cp.shape[0]//24
    print(all_days)
    ratio_list = []
    for i in range(cldni.shape[0]):
        if cldni[i]==0:
            ratio = 0
            ratio_list.append(ratio)
        else:
            ratio = dni[i]/cldni[i]
            ratio_list.append(ratio)
    cp_ratio = np.concatenate([cp,np.array(ratio_list).reshape(cp.shape[0],1)],axis = 1)
    list_24 = []

    for i in range(all_days):
        list_24.append(cp_ratio[i*24:(i+1)*24,:])

    cp_24_mean = []
    cp_24_std= []

    for xx in list_24:
        cp_24_mean.append(np.mean(xx[:,0]))
        cp_24_std.append(np.std(xx[:,0]))

    type_1_fea = []
    type_2_fea = []
    type_3_fea = []
    type_4_fea = []
    type_5_fea = []
    type_6_fea = []
    type_7_fea = []
    type_8_fea = []

    type_1_att = []
    type_2_att = []
    type_3_att = []
    type_4_att = []
    type_5_att = []
    type_6_att = []
    type_7_att = []
    type_8_att = []

    #gen_flag.append(1)
    attr_one_hot_codes = np.eye(8)
    feat_one_hot_codes = np.eye(10)
    cp_std_50 =np.percentile(np.array(cp_24_std),50),##1.8252663428174591
    cp_mean_25 =np.percentile(np.array(cp_24_mean),25),##1.5833
    cp_mean_50 =np.percentile(np.array(cp_24_mean),50),##3.4583
    cp_mean_75 =np.percentile(np.array(cp_24_mean),75)

    for i in range(len(cp_24_mean)):

        if cp_24_mean[i]<= cp_mean_25 and cp_24_std[i] <=cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_1_fea.append(np.array(sub_list))
            type_1_att.append(attr_one_hot_codes[0,:])

        if cp_24_mean[i]<= cp_mean_25 and cp_24_std[i] >cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_2_fea.append(np.array(sub_list))
            type_2_att.append(attr_one_hot_codes[1,:])

        if cp_24_mean[i]> cp_mean_25 and cp_24_mean[i]<=cp_mean_50 and cp_24_std[i] <=cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_3_fea.append(np.array(sub_list))
            type_3_att.append(attr_one_hot_codes[2,:])

        if cp_24_mean[i]> cp_mean_25 and cp_24_mean[i]<=cp_mean_50 and cp_24_std[i] > cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_4_fea.append(np.array(sub_list))
            type_4_att.append(attr_one_hot_codes[3,:])

        if cp_24_mean[i]> cp_mean_50 and cp_24_mean[i]<=cp_mean_75 and cp_24_std[i] <= cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_5_fea.append(np.array(sub_list))
            type_5_att.append(attr_one_hot_codes[4,:])

        if cp_24_mean[i]> cp_mean_50 and cp_24_mean[i]<=cp_mean_75 and cp_24_std[i] > cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_6_fea.append(np.array(sub_list))
            type_6_att.append(attr_one_hot_codes[5,:])
        
        if cp_24_mean[i]> cp_mean_75 and cp_24_std[i] <=cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_7_fea.append(np.array(sub_list))
            type_7_att.append(attr_one_hot_codes[6,:])
        
        if cp_24_mean[i]> cp_mean_75 and cp_24_std[i] > cp_std_50:
            daily_seq = list_24[i]
            sub_list = []
            for row in daily_seq:
                sub_list.append(np.append(feat_one_hot_codes[row[0],:],row[1]))
            type_8_fea.append(np.array(sub_list))
            type_8_att.append(attr_one_hot_codes[7,:])
    

    fea_type_list = [type_1_fea,type_2_fea,type_3_fea,type_4_fea,type_5_fea,type_6_fea,type_7_fea,type_8_fea]
    att_type_list = [type_1_att,type_2_att,type_3_att,type_4_att,type_5_att,type_6_att,type_7_att,type_8_att]

    fea_array = None
    attr_array = None


    for i in range(8):
        fea = fea_type_list[i]
        attr = att_type_list[i]

        if fea_array is None:
            fea_array = fea
        else:
            fea_array = np.concatenate([fea_array,fea],axis=0)
        
        if attr_array is None:
            attr_array = attr
        else:
            attr_array = np.concatenate([attr_array,attr],axis=0)
        
        
    print(fea_array.shape,attr_array.shape)


    data_feature_output = [
        Output(type_=OutputType.DISCRETE, dim=10, normalization=None, is_gen_flag=False),
        Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)]
    data_attribute_output = [
        Output(type_=OutputType.DISCRETE, dim=8, normalization=None, is_gen_flag=False)]
    return fea_array, attr_array,data_attribute_output,data_feature_output

fea_array, attr_array,data_attribute_output,data_feature_output = get_cloudtype()

'''
with open('D:/yanjiusheng/pythonProject/cloutype/data_train.npz', 'wb') as file:
    np.savez(file, arr_0 = fea_array, arr_1 = attr_array, arr_2 = np.array(gen_flag_list))

with open('/remote-home/21310019/2024/tem_rh/data_feature_output.pkl', 'wb') as file:
    pickle.dump(data_feature_output, file)

with open('/remote-home/21310019/2024/tem_rh/data_attribute_output.pkl', 'wb') as file:
    pickle.dump(data_attribute_output, file)
'''