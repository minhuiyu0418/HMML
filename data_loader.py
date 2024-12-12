import os
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf


def read_mat_file(filepath):
    mat = scipy.io.loadmat(filepath)
    scan = mat["IMG"][:,:,:]
    return scan

def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    img_new = img[11:-10,20:200,0:-21]
    return img_new

def process_scan(path):
    # Read scan
    volume = read_mat_file(path)
    # Normalize
    volume = normalize(volume)
    # Crop width, height and depth
    volume = resize_volume(volume)
    # Add an extra dimension
    volume = np.reshape(volume,(volume.shape[0],volume.shape[1],volume.shape[2],1))
    return volume



def get_training_data_info():
    '''1. read input images, label, and scores
       2. check for the presence of PET images and using synthetic PET for the subjects that have missing PET'''
    
    zero_files = os.listdir('E:/Dataset/ADNI_1_and_2/ADNI_AD_CN/CN_orig_processed')
    one_files = os.listdir('E:/Dataset/ADNI_1_and_2/ADNI_AD_CN/AD_orig_processed')
    all_PET_files = os.listdir('E:/Dataset/ADNI_ALL/ADNI_info/data/PET_orig_processed')
    all_PET_syn = os.listdir('E:/Dataset/ADNI_ALL/ADNI_info/data/PET_synthesized_orig_processed')

    zero_file_valid = []
    one_file_valid = []
    zero_file_syn = []
    one_file_syn = []

    for filename in zero_files:
        if filename in all_PET_files:
            zero_file_valid.append(filename)
        elif filename in all_PET_syn:
            zero_file_syn.append(filename)

    for filename in one_files:
        if filename in all_PET_files:
            one_file_valid.append(filename)
        elif filename in all_PET_syn:
            one_file_syn.append(filename)
            
    zero_paths = ['E:/Dataset/ADNI_1_and_2/ADNI_AD_CN/CN_orig_processed/'+filename for filename in zero_file_valid+zero_file_syn]
    zero_paths_PET_real = ['E:/Dataset/ADNI_ALL/ADNI_info/data/PET_orig_processed/'+filename for filename in zero_file_valid]
    zero_paths_PET_syn = ['E:/Dataset/ADNI_ALL/ADNI_info/data/PET_synthesized_orig_processed/'+filename for filename in zero_file_syn]

    one_paths = ['E:/Dataset/ADNI_1_and_2/ADNI_AD_CN/AD_orig_processed/'+filename for filename in one_file_valid+one_file_syn]
    one_paths_PET_real = ['E:/Dataset/ADNI_ALL/ADNI_info/data/PET_orig_processed/'+filename for filename in one_file_valid]
    one_paths_PET_syn = ['E:/Dataset/ADNI_ALL/ADNI_info/data/PET_synthesized_orig_processed/'+filename for filename in one_file_syn]

    data_MMSE = pd.read_excel('E:/Dataset/ADNI info extract/MMSE.xlsx')
    data_GDS = pd.read_excel('E:/Dataset/ADNI info extract/GDSCALE.xlsx') 

    one_paths_PET = one_paths_PET_real + one_paths_PET_syn
    zero_paths_PET = zero_paths_PET_real + zero_paths_PET_syn

    print("MRI with CN: " + str(len(zero_paths_PET)))
    print("MRI with AD: " + str(len(one_paths_PET)))

    comined_MRI_subject = one_file_valid + one_file_syn + zero_file_valid + zero_file_syn
    combined_file_path_MRI = one_paths + zero_paths
    combined_file_path_PET = one_paths_PET + zero_paths_PET
    combined_label = [1 for _ in range(len(one_paths))] + [0 for _ in range(len(zero_paths))]
    
    train_mmse_ADNI = []
    train_gds_ADNI = []


    for j in range (len(comined_MRI_subject)):
        flag = 0
        for i in range (len(data_MMSE)):
            if int(comined_MRI_subject[j][6:10]) == int(data_MMSE['RID'][i]) and data_MMSE['VISCODE2'][i] == 'sc':
                flag = 1
                if data_MMSE['MMSCORE'].values[i] >= 0:
                    train_mmse_ADNI.append(data_MMSE['MMSCORE'].values[i])
                else:
                    train_mmse_ADNI.append(-1)
        if flag == 0:
            train_mmse_ADNI.append(-1)
    print(len(train_mmse_ADNI))

    for j in range (len(comined_MRI_subject)):
        flag = 0
        for i in range (len(data_GDS)):
            if int(comined_MRI_subject[j][6:10]) == int(data_GDS['RID'][i]) and data_GDS['VISCODE2'][i] == 'sc':
                flag = 1
                if data_GDS['GDTOTAL'].values[i] >= 0:
                    train_gds_ADNI.append(data_GDS['GDTOTAL'].values[i])
                else:
                    train_gds_ADNI.append(-1)
        if flag == 0:
            train_gds_ADNI.append(-1)
    print(len(train_gds_ADNI))
                
    FileName_Label_List = np.array(list(zip(combined_file_path_MRI,combined_file_path_PET,combined_label,train_gds_ADNI,train_mmse_ADNI)))
    return FileName_Label_List


def create_file_list(FileName_Label_List,len_training_path,batch_size):
    '''Control the label of each batch data to ensure that it contains both positive and negative label'''
    
    np.random.shuffle(FileName_Label_List)

    zeros_index = np.where(FileName_Label_List[:,2]=='0')[0]
    ones_index = np.where(FileName_Label_List[:,2]=='1')[0]
    combined_index = np.concatenate((zeros_index,ones_index))
    if len(combined_index)%batch_size!= 0:
        combined_index = np.concatenate(combined_index,ones_index[:len(combined_index)%batch_size])

    batch_num = int(len(combined_index)/batch_size)

    column_subject_info_all = []
    for column in range(batch_size):
        column_subject_info_all.append(np.expand_dims(FileName_Label_List[combined_index[batch_num*column:batch_num*(column+1)]],1))

    paired_files = np.reshape(np.hstack(column_subject_info_all),(len_training_path,5))
    return paired_files
            
            
            
            
def load_data(len_training_path,batch_size,epochs):
    '''define data loaders'''

    FileName_Label_List = get_training_data_info()
    paired_files_all_epoches = create_file_list(FileName_Label_List,len_training_path,batch_size)
    for i in range (epochs-1):
        paired_files_all_epoches = np.vstack((paired_files_all_epoches,create_file_list(FileName_Label_List,len_training_path,batch_size)))
        
    def imageLoader():
        for file in paired_files_all_epoches:
    #         print('\nProcessing data...')
            X_MRI = (tf.convert_to_tensor(process_scan(file[0])),
                    tf.convert_to_tensor(process_scan(file[1])))
            Y = (tf.convert_to_tensor(file[2], dtype=np.float32),
                tf.convert_to_tensor(file[2], dtype=np.float32),
                tf.convert_to_tensor(file[2], dtype=np.float32),
                tf.convert_to_tensor(file[3], dtype=np.float32),
                tf.convert_to_tensor(file[4], dtype=np.float32))
    #         print('\nLabel loading finished',Y[0],Y[4])
            yield X_MRI,Y

    types = ((tf.float32,tf.float32),(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))
    shapes = (([160,180,160,1],[160,180,160,1]),([],[],[],[],[]))

    # define the training data for each round
    dataset = tf.data.Dataset.from_generator(imageLoader,
                                            output_types=types,
                                            output_shapes=shapes)
    
    return dataset
