import os
import math
import random   
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import nibabel as nib
import scipy
from scipy import ndimage,io
from tensorflow.keras import backend as K
from tensorflow.keras import layers

repeating = 5
run_time_all = ['round1','round2','round3','round4','round5']

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

zero_files = os.listdir('../../Dataset/ADNI_1_and_2/CN')
one_files = os.listdir('../../Dataset/ADNI_1_and_2/AD')
all_PET_files = os.listdir('../../Dataset/ADNI_PET')
all_PET_syn = os.listdir('../../Dataset/ADNI_PET_synthesized')


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
        
zero_paths = ['../../Dataset/ADNI_1_and_2/CN/'+filename for filename in zero_file_valid+zero_file_syn]
zero_paths_PET_real = ['../../Dataset/ADNI_PET/'+filename for filename in zero_file_valid]
zero_paths_PET_syn = ['../../Dataset/ADNI_PET_synthesized/'+filename for filename in zero_file_syn]

one_paths = ['../../Dataset/ADNI_1_and_2/AD/'+filename for filename in one_file_valid+one_file_syn]
one_paths_PET_real = ['../../Dataset/ADNI_PET/'+filename for filename in one_file_valid]
one_paths_PET_syn = ['../../Dataset/ADNI_PET_synthesized/'+filename for filename in one_file_syn]

data_MMSE = pd.read_excel('../../Dataset/ADNI info extract/MMSE.xlsx')
data_GDS = pd.read_excel('../../Dataset/ADNI info extract/GDSCALE.xlsx') 

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


# Define data loaders.
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

epochs = 50
batch_size = 5
len_training_path = 795

# Control the label of each batch data to ensure that it contains both positive and negative label
def create_file_list(FileName_Label_List):
    np.random.shuffle(FileName_Label_List)

    zeros_index = np.where(FileName_Label_List[:,2]=='0')[0]
    ones_index = np.where(FileName_Label_List[:,2]=='1')[0]

    first_column_subject_info = FileName_Label_List[zeros_index[:159]]
    second_column_subject_info = FileName_Label_List[zeros_index[159:318]]
    third_column_subject_info = FileName_Label_List[np.concatenate([zeros_index[318:],ones_index[:41]])]
    fourth_column_subject_info = FileName_Label_List[ones_index[41:200]]
    fifth_column_subject_info = FileName_Label_List[ones_index[200:]]

    paired_files = np.reshape(np.concatenate([np.expand_dims(first_column_subject_info,1),
                                                  np.expand_dims(second_column_subject_info,1),
                                                  np.expand_dims(third_column_subject_info,1),
                                                  np.expand_dims(fourth_column_subject_info,1),
                                                  np.expand_dims(fifth_column_subject_info,1)],
                                                 axis=1),
                                  (len_training_path,5))
    return paired_files

# Define the model modules
class Patches(layers.Layer):
    def __init__(self):
        super(Patches, self).__init__()

    def call(self, cnn_layer): #cnn(None,4,6,4,128)
        batch_size = tf.shape(cnn_layer)[0]
        patch_dims = cnn_layer.shape[-1]
        patches = tf.reshape(cnn_layer, [batch_size, 96, patch_dims]) #(None,96,128)
        patches = tf.transpose(patches, perm=[0, 2, 1]) #(None,128,96)
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=128, projection_dim=64):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def VoxCNN(inputs):
    
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x1 = layers.Reshape((5,5,5,7,8,7,32))(x)
    x2 = layers.AveragePooling3D(pool_size=(5,5,5))(x)
    x = layers.Multiply()([x1, x2])
    x  = layers.Reshape((35,40,35,32))(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    
    return x

# Define the model
def get_model(width=160, height=180, depth=160):

    inputs1 = layers.Input((width, height, depth, 1),name="input_MRI")
    x = layers.Conv3D(filters=8, kernel_size=5, activation="relu")(inputs1)
    x = layers.MaxPool3D(pool_size=2)(x)
    cnn_out1 = VoxCNN(x)

    GAP_MRI = layers.GlobalAveragePooling3D(data_format='channels_last')(cnn_out1)
    GAP_MRI_s = tf.keras.layers.Softmax()(GAP_MRI)
    L2_normalize_embeddings_MRI = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),name='l2_layer_MRI')(GAP_MRI_s)
    
    inputs2 = layers.Input((width, height, depth, 1),name="input_PET")
    xx = layers.Conv3D(filters=8, kernel_size=5, activation="relu")(inputs2)
    xx = layers.MaxPool3D(pool_size=2)(xx)
    cnn_out2 = VoxCNN(xx)
    
    GAP_PET = layers.GlobalAveragePooling3D(data_format='channels_last')(cnn_out2)
    GAP_PET_s = tf.keras.layers.Softmax()(GAP_PET)
    L2_normalize_embeddings_PET = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),name='l2_layer_PET')(GAP_PET_s)

    # Modality attention
    attention_1 = layers.Dense(1, activation="sigmoid")(GAP_MRI)
    attention_2 = layers.Dense(1, activation="sigmoid")(GAP_PET)
    cnn_out1 = layers.Multiply()([attention_1, cnn_out1])
    cnn_out2 = layers.Multiply()([attention_2, cnn_out2])
    cnn_cancated = layers.Concatenate(axis=-1)([cnn_out1, cnn_out2])

    # Transformer layers
    feature = Patches()(cnn_cancated)
    encoded_features = PatchEncoder()(feature)
    xxx1 = layers.BatchNormalization()(encoded_features)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.1
    )(xxx1, xxx1)

    xxx2 = layers.Add()([attention_output, encoded_features])
    xxx3 = layers.BatchNormalization()(xxx2)
    xxx3 = mlp(xxx3, hidden_units=[128,64], dropout_rate=0.1)
    encoded_features = layers.Add()([xxx3, xxx2])
    representation = layers.BatchNormalization()(encoded_features)
    representation = tf.math.reduce_mean(representation , axis=1)

    # Classification.
    representation_classification = mlp(representation, hidden_units=[64,64], dropout_rate=0.3)
    outputs_classification = layers.Dense(1, activation="sigmoid",name="outputs_classification")(representation_classification)

    # Regression GDS
    representation_regression_gds = mlp(representation, hidden_units=[64,64], dropout_rate=0.3)
    outputs_regression_gds = layers.Dense(1, activation="linear",name="outputs_regression_gds")(representation_regression_gds)

    # Regression MMSE
    representation_regression_mmse = mlp(representation, hidden_units=[64,64], dropout_rate=0.3)
    outputs_regression_mmse = layers.Dense(1, activation="linear",name="outputs_regression_mmse")(representation_regression_mmse)

    # Define the model.
    model = tf.keras.Model([inputs1,inputs2], [L2_normalize_embeddings_MRI, L2_normalize_embeddings_PET, outputs_classification, outputs_regression_gds, outputs_regression_mmse], name="3dVGG")
    return model

# Run
for i in range (repeating):
    run_time = run_time_all[i]
    model = None
    
    #define the image loader
    paired_files_all_epoches = create_file_list(FileName_Label_List)
    for i in range (epochs-1):
        paired_files_all_epoches = np.vstack((paired_files_all_epoches,create_file_list(FileName_Label_List)))

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
                                            output_shapes=shapes
                                            )

    model = get_model()
    model.summary()

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=math.ceil(len_training_path/batch_size), decay_rate=0.96, staircase=True
    )
    model.compile(
        loss={"l2_layer_MRI": tfa.losses.TripletSemiHardLoss(),
              "l2_layer_PET": tfa.losses.TripletSemiHardLoss(),
              "outputs_classification": "binary_crossentropy",
              "outputs_regression_gds": "mean_squared_error",
              "outputs_regression_mmse": "mean_squared_error"},
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics={'outputs_classification': [tf.keras.metrics.AUC(name='auc'),"acc"]},
        loss_weights={'l2_layer_MRI':0.001,'l2_layer_PET':0.001,'outputs_classification':1,'outputs_regression_gds':1,'outputs_regression_mmse':0.01}
    )

    model.fit(
        iter(dataset.batch(batch_size)),
        epochs=epochs,
        batch_size = batch_size,
        steps_per_epoch=math.ceil(len_training_path/batch_size),
        shuffle=True
    )

