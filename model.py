import tensorflow as tf
from tensorflow.keras import layers



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

def CNN(inputs):
    
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
def get_model_HMML(width=160, height=180, depth=160):

    inputs1 = layers.Input((width, height, depth, 1),name="input_MRI")
    x = layers.Conv3D(filters=8, kernel_size=5, activation="relu")(inputs1)
    x = layers.MaxPool3D(pool_size=2)(x)
    cnn_out1 = CNN(x)

    GAP_MRI = layers.GlobalAveragePooling3D(data_format='channels_last')(cnn_out1)
    GAP_MRI_s = tf.keras.layers.Softmax()(GAP_MRI)
    L2_normalize_embeddings_MRI = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),name='l2_layer_MRI')(GAP_MRI_s)
    
    inputs2 = layers.Input((width, height, depth, 1),name="input_PET")
    xx = layers.Conv3D(filters=8, kernel_size=5, activation="relu")(inputs2)
    xx = layers.MaxPool3D(pool_size=2)(xx)
    cnn_out2 = CNN(xx)
    
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