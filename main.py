import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from data_loader import load_data
from util import print_gpu
from model import get_model_HMML

print_gpu()

# repeat the running for 5 times
repeating = 5
run_time_all = ['round1','round2','round3','round4','round5']

len_training_path = 795
epochs = 50
batch_size = 5


# Run
for i in range (repeating):
    run_time = run_time_all[i]
    model = None
    
    # load the image name, label and score
    dataset = load_data(len_training_path,batch_size,epochs)
    model = get_model_HMML()
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
    
    checkpoint_filepath = './'+run_time+'/{epoch:02d}-checkpoint'
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='loss', mode='min', save_best_only=False, save_freq='epoch')

    model.fit(
        iter(dataset.batch(batch_size)),
        epochs=epochs,
        batch_size = batch_size,
        steps_per_epoch=math.ceil(len_training_path/batch_size),
        shuffle=True,
        callbacks=[model_checkpoint_cb]
    )

