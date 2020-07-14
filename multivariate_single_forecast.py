# INSERT COPYRIGHT HERE

# python modules
import matplotlib.pyplot as plt
import tensorflow as tf
from ta.utils import dropna
import pyfinancialdata
from datetime import datetime as dt 

# local modules
from market import EquityData
from models.lstm import split, split_multivariate, show_plot
from utils import plot_train_history
from config import BATCH_SIZE, BUFFER_SIZE, \
                   EPOCHS, CLASSIFICATION, \
                   HISTORY_SIZE, TARGET_DIS, \
                   STEP, FEATURES, DATA_DIR, \
                   DATA_SYM

# use seed 
tf.random.set_seed(42)

def get_lstm():
  """
  Keras LSTM Architecture 
  """
  ssm = tf.keras.models.Sequential()

  ssm.add(tf.keras.layers.LSTM(256, return_sequences=True,
                                input_shape=xt.shape[-2:]))

  ssm.add(tf.keras.layers.LSTM(256, return_sequences=True))

  ssm.add(tf.keras.layers.LSTM(128))

  ssm.add(tf.keras.layers.Dense(64))

  ssm.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  ssm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])
  return ssm

if __name__ == "__main__":
  # construct data and get additional info 
  e = EquityData(DATA_DIR, DATA_SYM)
  data = pyfinancialdata.get_multi_year(provider='histdata', instrument='SPXUSD', years=[2016, 2017, 2018], time_group='10min')
  e.data = data

  # add MACD
  e.data['MA_long'] = e.data['close'].rolling(window=52).mean()
  e.data['MA_short'] = e.data['close'].rolling(window=7).mean()

  # evaluation interval
  window = int(e.data.shape[0]/BATCH_SIZE) * 1
  
  # pick selected features
  features = e.data[FEATURES]
  
  # cleanse
  features.index = e.data.index
  features = features.dropna()
  features = features[26:]

  # to numpy
  dataset = features.values

  # get validation and training data
  xt, yt, xv, yv = split_multivariate(dataset, 
                                      HISTORY_SIZE, 
                                      TARGET_DIS,
                                      STEP, 
                                      single_step=True, 
                                      classification=CLASSIFICATION)

  # construct datasets
  t_ds = tf.data.Dataset.from_tensor_slices((xt, yt))
  t_ds = t_ds.cache().shuffle(
      BUFFER_SIZE).batch(BATCH_SIZE).repeat()
  v_ds = tf.data.Dataset.from_tensor_slices((xv, yv))
  v_ds = v_ds.batch(BATCH_SIZE).repeat()

  # validation callback
  v_cb = tf.keras.callbacks.ModelCheckpoint(
      'checkpoints/multivariate_single_model', monitor='val_accuracy', verbose=1, save_best_only=True,
      save_weights_only=False, mode='auto', save_freq='epoch'
  )
  
  # tensorboard callback
  logdir = "logs/scalars/{}".format(dt.today())
  ts_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  # get model
  ssm = get_lstm()

  # run trial
  history = ssm.fit(t_ds, epochs=EPOCHS,
                    steps_per_epoch=window,
                    validation_data=v_ds,
                    validation_steps=50, 
                    callbacks=[v_cb, ts_cb])

  # TODO plot results
  # plot_train_history(single_step_history,
  #                   'Single Step Training and validation loss')

  # for x, y in val_data_single.take(10):
  #   y_pred = ssm.predict(x)[0]
  #   print(f"prediction: {y_pred}")
  #   if CLASSIFICATION:
  #     if y_pred >= 0.5:
  #       y_pred = 1
  #     else:
  #       y_pred = 0
  #   plot = show_plot([x[0][:, 0].numpy(), y[0].numpy(),
  #                     y_pred], TARGET_DIS,
  #                   'Single Step Prediction')
  #   plot.show()
