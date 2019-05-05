
# Twitch Image Classifier
Originally, I did a project in the data analysis class in which we classified twitch images according to the shown game. Back then, we used SVMs and transfered learning.
Now, I want to do this project again - this time building my own network architecture. My goal is to get at least 80% accuracy.

## Setup
Please add the twitch images to the data directory. If you do not have any preview images, then you can use my twitch preview image crawler, which is also available on my github.

The following directory structure should be present

data/training/Game1/...

data/training/Game2/...

data/training/Game...

data/validation/Game1/...

data/validation/Game2/...

data/validataion/Game...

## Code

### Check for networkx version
For hyperas networkx version 1.11 is required. Thus, we need to check this first!


```python
import networkx

assert networkx.__version__ == '1.11', 'You need to install networkx version 1.11, otherwise hyperas does not work'
```

### Check directory structure
In the first step we'll just check, if the right directory structure is present. This will hopefully save some time, if I work on this later again and I need to setup the environment again.


```python
import os
import glob

# First check if data directory is present. This should be the case anyways, since this is set in the git repository,
# but we'll check :)

assert os.path.isdir('data'), 'The data directory is not present, but mandatory'

# Check if training and validation is present
assert os.path.isdir('data/training'), 'The training directory in the data directory is not present, but mandatory'
assert os.path.isdir('data/validation'), 'The validation directory in the data directory is not present, but mandatory'

# Check how many directories (== Games) there are in training and validation
training_games = [game for game in os.listdir("data/training/") if os.path.isdir('data/training/{}'.format(game))]
validation_games = [game for game in os.listdir("data/validation/") if os.path.isdir('data/validation/{}'.format(game))]
assert training_games == validation_games, 'The games must be the same in training and validation'

# Check how many games there are
assert len(training_games) >= 1, 'You need to add at least one game'
assert len(training_games) >= 2, 'For a real classification scenario you should add more than one game'

# Determine training and validation sample size
training_samples = len([name for name in glob.glob('data/training/*/*') if os.path.isfile(name)])
validation_samples = len([name for name in glob.glob('data/validation/*/*') if os.path.isfile(name)])
assert training_samples > validation_samples, 'You should use more training samples than validation samples'
assert training_samples > 32, 'You should use at least 32 files'
```

## Imports and Modeldefinition


```python
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
```

    Using TensorFlow backend.
    


```python
def model(train_generator, validation_generator):
    
    training_steps = training_samples // batch_size
    validation_steps = validation_samples // batch_size
    
    input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    model.compile(loss=['categorical_crossentropy'],
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    score, acc = model.evaluate_generator(validation_generator)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
```

## Data Definition
Here we define the data image size and the dataflows, so they can be used in the model.


```python
def definitions():
    global batch_size, epochs, img_width, img_height, training_samples, validation_samples, nb_classes
    batch_size = 32
    epochs = 10
    img_width, img_height = 320, 180
    
    training_samples = len([name for name in glob.glob('data/training/*/*') if os.path.isfile(name)])
    validation_samples = len([name for name in glob.glob('data/validation/*/*') if os.path.isfile(name)])
    nb_classes = len([name for name in os.listdir('data/training') if os.path.isdir('data/training/{}'.format(name))])
    

def data():
    # Load definitions
    definitions()
    
    train_data_dir = 'data/training'
    validation_data_dir = 'data/validation'
    
    class FixedImageDataGenerator(ImageDataGenerator):
        def standardize(self, x):
            x = x / 255
            return x

    test_datagen = FixedImageDataGenerator(
        rescale=None)

    train_datagen = FixedImageDataGenerator(
        rescale=None)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator

```

## Main-Loop
This is the main loop, starting keras with the hyperparameter optimization.


```python
best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials(),
                                          functions=[definitions],
                                          notebook_name='TwitchImageClassifier')
print(best_run)
print(best_model)
```

    >>> Imports:
    #coding=utf-8
    
    try:
        import networkx
    except:
        pass
    
    try:
        import os
    except:
        pass
    
    try:
        import glob
    except:
        pass
    
    try:
        from hyperopt import Trials, STATUS_OK, tpe
    except:
        pass
    
    try:
        from hyperas import optim
    except:
        pass
    
    try:
        from hyperas.distributions import uniform
    except:
        pass
    
    try:
        from keras.models import Sequential
    except:
        pass
    
    try:
        from keras.layers.core import Dense, Dropout, Activation, Flatten
    except:
        pass
    
    try:
        from keras.layers import Conv2D, MaxPooling2D
    except:
        pass
    
    try:
        from keras.optimizers import SGD
    except:
        pass
    
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except:
        pass
    
    try:
        from keras.datasets import cifar10
    except:
        pass
    
    try:
        from keras.utils import np_utils
    except:
        pass
    
    >>> Hyperas search space:
    
    def get_space():
        return {
            'Dropout': hp.uniform('Dropout', 0, 1),
        }
    
    >>> Functions
      1: def definitions():
      2:     global batch_size, epochs, img_width, img_height, training_samples, validation_samples, nb_classes
      3:     batch_size = 32
      4:     epochs = 10
      5:     img_width, img_height = 320, 180
      6:     
      7:     training_samples = len([name for name in glob.glob('data/training/*/*') if os.path.isfile(name)])
      8:     validation_samples = len([name for name in glob.glob('data/validation/*/*') if os.path.isfile(name)])
      9:     nb_classes = len([name for name in os.listdir('data/training') if os.path.isdir('data/training/{}'.format(name))])
     10: 
     11: 
    >>> Data
      1: 
      2: # Load definitions
      3: definitions()
      4: 
      5: train_data_dir = 'data/training'
      6: validation_data_dir = 'data/validation'
      7: 
      8: class FixedImageDataGenerator(ImageDataGenerator):
      9:     def standardize(self, x):
     10:         x = x / 255
     11:         return x
     12: 
     13: test_datagen = FixedImageDataGenerator(
     14:     rescale=None)
     15: 
     16: train_datagen = FixedImageDataGenerator(
     17:     rescale=None)
     18: 
     19: train_generator = train_datagen.flow_from_directory(
     20:     train_data_dir,
     21:     target_size=(img_width, img_height),
     22:     batch_size=batch_size,
     23:     class_mode='categorical')
     24: 
     25: validation_generator = test_datagen.flow_from_directory(
     26:     validation_data_dir,
     27:     target_size=(img_width, img_height),
     28:     batch_size=batch_size,
     29:     class_mode='categorical')
     30: 
     31: 
     32: 
     33: 
    >>> Resulting replaced keras model:
    
       1: def keras_fmin_fnct(space):
       2: 
       3:     
       4:     training_steps = training_samples // batch_size
       5:     validation_steps = validation_samples // batch_size
       6:     
       7:     input_shape = (img_width, img_height, 3)
       8:     
       9:     model = Sequential()
      10:     model.add(Conv2D(16, (3, 3), input_shape=input_shape))
      11:     model.add(Activation('relu'))
      12:     model.add(MaxPooling2D(pool_size=(2, 2)))
      13:     
      14:     model.add(Conv2D(32, (3, 3)))
      15:     model.add(Activation('relu'))
      16:     model.add(MaxPooling2D(pool_size=(2, 2)))
      17: 
      18:     model.add(Conv2D(64, (3, 3)))
      19:     model.add(Activation('relu'))
      20:     model.add(MaxPooling2D(pool_size=(2, 2)))
      21: 
      22:     model.add(Flatten())
      23:     model.add(Dense(64))
      24:     model.add(Activation('relu'))
      25:     model.add(Dropout(space['Dropout']))
      26:     model.add(Dense(nb_classes))
      27:     model.add(Activation('sigmoid'))
      28:     model.compile(loss=['categorical_crossentropy'],
      29:                   optimizer='adam',
      30:                   metrics=['accuracy'])
      31: 
      32:     model.fit_generator(
      33:         train_generator,
      34:         steps_per_epoch=training_steps,
      35:         epochs=epochs,
      36:         validation_data=validation_generator,
      37:         validation_steps=validation_steps)
      38: 
      39:     score, acc = model.evaluate_generator(validation_generator)
      40: 
      41:     return {'loss': -acc, 'status': STATUS_OK, 'model': model}
      42: 
    Found 4000 images belonging to 10 classes.
    Found 1000 images belonging to 10 classes.
    Epoch 1/10
    125/125 [==============================] - 287s 2s/step - loss: 1.8093 - acc: 0.4085 - val_loss: 1.2160 - val_acc: 0.6452
    Epoch 2/10
    125/125 [==============================] - 287s 2s/step - loss: 0.8677 - acc: 0.7605 - val_loss: 0.5135 - val_acc: 0.8367
    Epoch 3/10
    125/125 [==============================] - 292s 2s/step - loss: 0.4925 - acc: 0.8662 - val_loss: 0.4390 - val_acc: 0.8649
    Epoch 4/10
    125/125 [==============================] - 292s 2s/step - loss: 0.2991 - acc: 0.9163 - val_loss: 0.4686 - val_acc: 0.8659
    Epoch 5/10
    125/125 [==============================] - 284s 2s/step - loss: 0.2306 - acc: 0.9370 - val_loss: 0.3599 - val_acc: 0.8982
    Epoch 6/10
     48/125 [==========>...................] - ETA: 2:41 - loss: 0.1982 - acc: 0.9414


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-a43b7c209f0b> in <module>()
          5                                           trials=Trials(),
          6                                           functions=[definitions],
    ----> 7                                           notebook_name='TwitchImageClassifier')
          8 print(best_run)
          9 print(best_model)
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperas\optim.py in minimize(model, data, algo, max_evals, trials, functions, rseed, notebook_name, verbose, eval_space, return_space)
         65                                      full_model_string=None,
         66                                      notebook_name=notebook_name,
    ---> 67                                      verbose=verbose)
         68 
         69     best_model = None
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperas\optim.py in base_minimizer(model, data, functions, algo, max_evals, trials, rseed, full_model_string, notebook_name, verbose, stack)
        131              trials=trials,
        132              rstate=np.random.RandomState(rseed),
    --> 133              return_argmin=True),
        134         get_space()
        135     )
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
        305             verbose=verbose,
        306             catch_eval_exceptions=catch_eval_exceptions,
    --> 307             return_argmin=return_argmin,
        308         )
        309 
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\base.py in fmin(self, fn, space, algo, max_evals, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin)
        633             pass_expr_memo_ctrl=pass_expr_memo_ctrl,
        634             catch_eval_exceptions=catch_eval_exceptions,
    --> 635             return_argmin=return_argmin)
        636 
        637 
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\fmin.py in fmin(fn, space, algo, max_evals, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin)
        318                     verbose=verbose)
        319     rval.catch_eval_exceptions = catch_eval_exceptions
    --> 320     rval.exhaust()
        321     if return_argmin:
        322         return trials.argmin
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\fmin.py in exhaust(self)
        197     def exhaust(self):
        198         n_done = len(self.trials)
    --> 199         self.run(self.max_evals - n_done, block_until_done=self.async)
        200         self.trials.refresh()
        201         return self
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\fmin.py in run(self, N, block_until_done)
        171             else:
        172                 # -- loop over trials and do the jobs directly
    --> 173                 self.serial_evaluate()
        174 
        175             if stopped:
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\fmin.py in serial_evaluate(self, N)
         90                 ctrl = base.Ctrl(self.trials, current_trial=trial)
         91                 try:
    ---> 92                     result = self.domain.evaluate(spec, ctrl)
         93                 except Exception as e:
         94                     logger.info('job exception: %s' % str(e))
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\hyperopt\base.py in evaluate(self, config, ctrl, attach_attachments)
        838                 memo=memo,
        839                 print_node_on_error=self.rec_eval_print_node_on_error)
    --> 840             rval = self.fn(pyll_rval)
        841 
        842         if isinstance(rval, (float, int, np.number)):
    

    E:\Users\Hilko\Dokumente\TwitchImageClassifier\temp_model.py in keras_fmin_fnct(space)
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\legacy\interfaces.py in wrapper(*args, **kwargs)
         85                 warnings.warn('Update your `' + object_name +
         86                               '` call to the Keras 2 API: ' + signature, stacklevel=2)
    ---> 87             return func(*args, **kwargs)
         88         wrapper._original_function = func
         89         return wrapper
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\models.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       1154                                         use_multiprocessing=use_multiprocessing,
       1155                                         shuffle=shuffle,
    -> 1156                                         initial_epoch=initial_epoch)
       1157 
       1158     @interfaces.legacy_generator_methods_support
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\legacy\interfaces.py in wrapper(*args, **kwargs)
         85                 warnings.warn('Update your `' + object_name +
         86                               '` call to the Keras 2 API: ' + signature, stacklevel=2)
    ---> 87             return func(*args, **kwargs)
         88         wrapper._original_function = func
         89         return wrapper
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\engine\training.py in fit_generator(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)
       2075                     outs = self.train_on_batch(x, y,
       2076                                                sample_weight=sample_weight,
    -> 2077                                                class_weight=class_weight)
       2078 
       2079                     if not isinstance(outs, list):
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\engine\training.py in train_on_batch(self, x, y, sample_weight, class_weight)
       1795             ins = x + y + sample_weights
       1796         self._make_train_function()
    -> 1797         outputs = self.train_function(ins)
       1798         if len(outputs) == 1:
       1799             return outputs[0]
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
       2330         updated = session.run(self.outputs + [self.updates_op],
       2331                               feed_dict=feed_dict,
    -> 2332                               **self.session_kwargs)
       2333         return updated[:len(self.outputs)]
       2334 
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
        887     try:
        888       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 889                          run_metadata_ptr)
        890       if run_metadata:
        891         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1118     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1119       results = self._do_run(handle, final_targets, final_fetches,
    -> 1120                              feed_dict_tensor, options, run_metadata)
       1121     else:
       1122       results = []
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1315     if handle is None:
       1316       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
    -> 1317                            options, run_metadata)
       1318     else:
       1319       return self._do_call(_prun_fn, self._session, handle, feeds, fetches)
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1321   def _do_call(self, fn, *args):
       1322     try:
    -> 1323       return fn(*args)
       1324     except errors.OpError as e:
       1325       message = compat.as_text(e.message)
    

    c:\users\hilko\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\client\session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1300           return tf_session.TF_Run(session, options,
       1301                                    feed_dict, fetch_list, target_list,
    -> 1302                                    status, run_metadata)
       1303 
       1304     def _prun_fn(session, handle, feed_dict, fetch_list):
    

    KeyboardInterrupt: 



```python

```


```python

```
