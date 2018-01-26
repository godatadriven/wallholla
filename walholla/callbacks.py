import keras.callbacks as callbacks

early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='min')
base_logger = callbacks.BaseLogger()
lr_scheduler = callbacks.LearningRateScheduler(schedule=lambda x: 1/(1+x))
increasing_lr_scheduler = callbacks.LearningRateScheduler(schedule=lambda x: x * 0.01)
plateau_reduces = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
