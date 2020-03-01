import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications import VGG16
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def data_split(n_classes, data, labels, seed=123, other_base=1):
    np.random.seed(seed)

    # python pass parameters as reference, it's better to make a copy here
    labels_copy = np.copy(labels)
    
#     other_base = 6 - n_classes
    
    if n_classes < 2:
        raise ValueError('n_classes must be between 2 and 10')

    # indices of classes that are not part of the 'other' class
    class_indices = np.argwhere(labels_copy < (n_classes - 1))[:,0]
    
    num_of_each_class = class_indices.shape[0]//(n_classes-1)

    # draw sample from 'other' class
    other_indices = np.random.choice(np.argwhere(labels_copy >= (n_classes + other_base - 1))[:,0], num_of_each_class, replace=False)

    # combine indices os the selected classs and the 'other' class
    all_indices = np.concatenate((class_indices, other_indices))

    # create new train and test datasets and labels
    selected_data = data[all_indices, :]
    selected_labels = labels_copy[all_indices]  #deep copy

    # adjust label of 'other' class: set to n_classes - 1, i.e. if we have two classes, cats and other, than cats will
    # have 0 and other will have 1
    selected_labels[selected_labels >= (n_classes - 1)] = n_classes - 1
    
    return (selected_data, selected_labels)

def generate_data(cur_target_class_ids, full_target_class_ids, data, labels, refining, seed=123):
    
    np.random.seed(seed)
    
    num_of_each_class = 5000
    
    n_final_classes_including_other = 10
    full_class_ids = np.arange(1, n_final_classes_including_other+1)
    
    if refining == True:
        other_class_ids = np.setxor1d(full_class_ids, cur_target_class_ids)
    else:
        other_class_ids = np.setxor1d(full_class_ids, full_target_class_ids)

    labels_copy = np.copy(labels)
    
    target_class_indices = np.array([], dtype='int32')
    for class_id in cur_target_class_ids:
            target_class_indices = np.append(target_class_indices, np.argwhere(labels_copy == class_id-1)[:,0])
    # TODO: temp code
    np.random.shuffle(target_class_indices)
    print(target_class_indices[:10])
    # create new train and test datasets and labels
    target_class_data = data[target_class_indices, :]
    target_class_labels = np.squeeze(labels_copy[target_class_indices, :])
    target_class_labels_copy = np.copy(target_class_labels)
    
    # Reasign the index of target-classes, starting from 1
    for i, class_id in enumerate(cur_target_class_ids):
        target_class_labels[np.argwhere(target_class_labels_copy == class_id-1)[:, 0]] = i+1
        
    # Other-class
    all_other_class_indices = np.array([], dtype='int32')
    for class_id in other_class_ids:
        if class_id == other_class_ids[0]: #TODO
            all_other_class_indices = np.append(all_other_class_indices, np.argwhere(labels_copy == class_id-1)[:,0])
    
    print('current other class: ' + str(other_class_ids))
    print('current target class: ' + str(cur_target_class_ids))
    print('all target target class: ' + str(full_target_class_ids))
    print('all class: ' + str(full_class_ids))
    print('all other class indices: ' + str(len(all_other_class_indices)))
    other_class_indices = np.random.choice(all_other_class_indices, num_of_each_class)
    
    other_class_data = data[other_class_indices, :]
    
    # set 'other' label to zero
    other_class_labels = np.array([0]*num_of_each_class)
    
    print(target_class_labels.shape, other_class_labels.shape)
    
    selected_data = np.concatenate((target_class_data, other_class_data))
    selected_labels = np.concatenate((target_class_labels, other_class_labels))
    
    return (selected_data, selected_labels)


def create_base_model(input_shape, baseMapNum = 32, weight_decay = 1e-4, net_type='kaggle'):
    '''net_type: 1. kaggle 2. VGG16'''
    model = Sequential()
    if net_type == 'kaggle':
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), 
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.2))

        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.3))

        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
#         model.add(Dropout(0.3))
        
        model.add(Dense(512, activation='relu'))
#         model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
    
    elif net_type == 'VGG16':   # not feasiable for cifar10 -- input size too small
        vgg16_net = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)
        model.add(vgg16_net)
    else:
        raise ValueError('unknown base net type')
    return model

def training(model, num_classes_including_other, train_data, train_labels, batch_size, epochs):
    
    y_train_categorical = np_utils.to_categorical(train_labels, num_classes_including_other)
    
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    training_log_ep75 = model.fit(train_data, y_train_categorical, batch_size=batch_size,epochs=3*epochs, verbose=1, shuffle=False)
    
    opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    training_log_ep100 = model.fit(train_data, y_train_categorical, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False)

    opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    training_log_ep125 = model.fit(train_data, y_train_categorical, batch_size=batch_size,epochs=epochs, verbose=1, shuffle=False)
    
#     opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
#     training_log_ep75 = model.fit_generator(
#                                     data_generator.flow(train_data, y_train_categorical, batch_size=batch_size),
#                                     steps_per_epoch=train_data.shape[0] // batch_size,epochs=3*epochs, verbose=1, shuffle=True)
    
#     opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
#     training_log_ep100 = model.fit_generator(
#                                     data_generator.flow(train_data, y_train_categorical, batch_size=batch_size),
#                                     steps_per_epoch=train_data.shape[0] // batch_size,epochs=epochs, verbose=1)

#     opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
#     training_log_ep125 = model.fit_generator(
#                                     data_generator.flow(train_data, y_train_categorical, batch_size=batch_size),
#                                     steps_per_epoch=train_data.shape[0] // batch_size,epochs=epochs, verbose=1)
    
#     trained_weights = model.get_weights()
    return 0

def tuning(model, num_classes_including_other, train_data, train_labels, data_generator, batch_size, epochs):
    
    y_train_categorical = np_utils.to_categorical(train_labels, num_classes_including_other)

    opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    training_log_ep125 = model.fit_generator(
                                    data_generator.flow(train_data, y_train_categorical, batch_size=batch_size),
                                    steps_per_epoch=train_data.shape[0] // batch_size,epochs=5*epochs, verbose=1)
     

def evaluate(model, cur_target_class_ids,test_data, test_labels, batch_size):
    acc = []
    loss = []
    confusion_mtx = []
    #num_classes = len(cur_target_class_ids)+1
    num_classes = len(cur_target_class_ids)
    for i, class_id in enumerate(cur_target_class_ids):
        target_class_indices = np.argwhere(test_labels==class_id-1)[:, 0]
        target_test_data = test_data[target_class_indices]
#         print(len(target_test_data))
        #target_test_labels = np.array([i+1]*target_test_data.shape[0])
        target_test_labels = np.array([i+1 ]*target_test_data.shape[0])
        target_test_labels_categorical = np_utils.to_categorical(target_test_labels, num_classes)
        scores = model.evaluate(target_test_data, target_test_labels_categorical, batch_size, verbose=1)
        loss += [scores[0]]
        acc += [scores[1]]
        y_pred_categorical = model.predict(target_test_data)
        y_pred = np.argmax(y_pred_categorical, axis=1)
        print('y_sum ' + str(np.sum(y_pred)))
        unique, counts = np.unique(y_pred, return_counts=True)
        xor_result = np.setxor1d(unique, np.arange(0, num_classes))
        if len(xor_result) != 0:
            y_pred = np.concatenate((y_pred, xor_result))
            unique, counts = np.unique(y_pred, return_counts=True) 
        counts[0] = class_id  # set the first column to the class_id
        if len(counts) != num_classes:
            raise ValueError('TODO: handle dim exception: (%d, %d) ' %(len(counts), num_classes)) 
        confusion_mtx.append(counts.tolist())
    return acc, loss, np.array(confusion_mtx)

def plot_confusion_mtx(confusion_mtx, class_names):
    n_classes = len(confusion_mtx)
    class_ids = confusion_mtx[:, 0]
    df_cm = pd.DataFrame(confusion_mtx[:, 1:], index = [class_names[i] for i in class_ids] ,
                  columns =[class_names[i] for i in class_ids])
    plt.figure()
    sn.heatmap(df_cm, annot=True, cmap='summer')
    plt.show()
    return 

def training_with_GC(model, full_target_class_ids, epochs, x_train, y_train, x_test, y_test, data_generator, batch_size, refining):
    acc_GC = []
    loss_GC = []
    for i in range(len(full_target_class_ids)): # exclusive 'other'
        cur_target_class_ids = full_target_class_ids[:i+1]
        if(i >= 1):
            print('-----Adding a new class (total classes including other: %s)------' % str(i+2))
            for layer in model.layers:
                layer.trainable=False    
        model.add(Dense(i+2, activation='softmax'))
        (cur_train_data, cur_train_labels) = generate_data(cur_target_class_ids, full_target_class_ids, 
                                                           x_train, y_train, refining)
        data_generator.fit(cur_train_data)
        training(model, i+2, cur_train_data, cur_train_labels, data_generator, batch_size, epochs)
        print('-----Fine Tuning------')
        for layer in model.layers[-3:-1]:
            layer.trainable = True
        tuning(model, i+2, cur_train_data, cur_train_labels, data_generator, batch_size, epochs)
        # set back to untrainable
        for layer in model.layers[-3:-1]:
            layer.trainable = False
        acc, loss, confusion_mtx = evaluate(model, cur_target_class_ids, x_test, y_test, 100)
        acc_GC += [acc]
        loss_GC += [loss]
        model.pop()
        # plot confusion mtx
    return acc_GC, loss_GC

def plot_result(acc_all_class_from_scratch, acc_GC, acc_GC_refining, full_target_class_ids, class_names):
    
    def get_leading_zeros(arr):
        n_leading_zeros = 0
        for eli in arr:
            if eli != 0:
                break
            else:
                n_leading_zeros += 1
        return n_leading_zeros
    
    
    num_classes = len(full_target_class_ids)
    
    acc_GC_for_each_class = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(len(acc_GC[i])):
            acc_GC_for_each_class[j][i] = acc_GC[i][j]
            
    acc_GC_refining_for_each_class = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(len(acc_GC_refining[i])):
            acc_GC_refining_for_each_class[j][i] = acc_GC_refining[i][j]
    
    num_logs_for_each_class = np.arange(num_classes, 0, -1)
    acc_GC_avg_for_each_class = np.sum(acc_GC_for_each_class, axis=1)/num_logs_for_each_class
    acc_GC_refining_avg_for_each_class = np.sum(acc_GC_refining_for_each_class, axis=1)/num_logs_for_each_class

    for i in range(num_classes):
        n_leading_zeros = get_leading_zeros(acc_GC_refining_for_each_class[i])
        l = len(acc_GC_refining_for_each_class[i])
        plt.plot(np.arange(n_leading_zeros, l), acc_GC_for_each_class[i][n_leading_zeros:], label='Growing Classifier', marker='^')
        plt.plot(np.arange(n_leading_zeros, l), acc_GC_refining_for_each_class[i][n_leading_zeros:], label='Growing Classifier (Refining)', marker='^')
        plt.plot(np.arange(len(acc_GC_refining_for_each_class[i])), [acc_all_class_from_scratch[i]]*l, label='Train From Scratch', marker='^')
        plt.legend()
        plt.xlabel('num classes')
        plt.ylabel('accuracy')
        plt.title(class_names[full_target_class_ids[i]])
        plt.show()

        
#     print(acc_GC_for_each_class)

    for i in range(num_classes):
        n_leading_zeros = get_leading_zeros(acc_GC_for_each_class[i])
        l = len(acc_GC_for_each_class[i])
        plt.plot(np.arange(n_leading_zeros, l), acc_GC_for_each_class[i][n_leading_zeros:], label=class_names[full_target_class_ids[i]], marker='^')
    plt.legend()
    plt.xlabel('num classes')
    plt.ylabel('accuracy')
    plt.title('GC on all classes (No Refining)')
    plt.show()

    for i in range(num_classes):
        n_leading_zeros = get_leading_zeros(acc_GC_refining_for_each_class[i])
        l = len(acc_GC_refining_for_each_class[i])
        plt.plot(np.arange(n_leading_zeros, l), acc_GC_refining_for_each_class[i][n_leading_zeros:], label=class_names[full_target_class_ids[i]], marker='^')
    plt.legend()
    plt.xlabel('num classes')
    plt.ylabel('accuracy')
    plt.title('GC on all classes (Refining)')
    plt.show()

    plt.plot(np.arange(num_classes), acc_GC_avg_for_each_class, label='Growing Classifer', marker='^')
    plt.plot(np.arange(num_classes), acc_GC_refining_avg_for_each_class, label='Growing Classifer (Refining)', marker='^')
    plt.plot(np.arange(num_classes), acc_all_class_from_scratch, label='train from scratch', marker='^')
    plt.legend()
    plt.xlabel('class ID ' + str([class_names[el] for el in full_target_class_ids]))
    plt.ylabel('accuracy')
    plt.title('GC (with and without refining) v.s. Train from scratch')
    plt.show()
    
    return


# General Procedure
'''def one_run(model, x_train, y_train, x_test, y_test, batch_size, full_target_class_ids, initial_weights, base_epochs, datagen, class_names, seed=123):'''
def one_run(model, x_train, y_train, x_test, y_test, batch_size, full_target_class_ids,initial_weights, base_epochs, class_names):
    
#     np.random.seed(seed)
    num_classes_excluding_other = len(full_target_class_ids)
    num_classes_including_other = num_classes_excluding_other + 1


    
    ### Train From Scratch ###
    print('\n======Training From Scratch=========\n')
#     weights_train_from_scratch = list(initial_weights)
    # use the same initial weights
    model.set_weights(initial_weights)
#     for layer in model.layers:
#         layer.trainable = True
#     model.add(Dense(num_classes_including_other, activation='softmax'))

    # in train-from-scratch, feed all target class at the begining, 
    # i.e., cur_target_class_ids is same with full_target_class_ids
    cur_target_class_ids = np.copy(full_target_class_ids)

#     (cur_train_data, cur_train_labels) = generate_data(cur_target_class_ids, full_target_class_ids, 
#                                                                x_train, y_train, False, seed)

#     datagen.fit(cur_train_data)

#     training(model, num_classes_including_other, cur_train_data, cur_train_labels, 
#                            datagen, batch_size, base_epochs)
    print("num of class", num_classes_excluding_other)
    training(model, num_classes_excluding_other, x_train, y_train, batch_size, base_epochs)

    acc_all_class_from_scratch, loss_all_class_from_scratch, scratch_confusion_mtx = evaluate(model, cur_target_class_ids, 
                                                                       x_test, y_test, batch_size)
    # remove the last layer
#     model.pop()
    
    print(acc_all_class_from_scratch)
    plot_confusion_mtx(scratch_confusion_mtx, class_names)
    
    ### Growing Classifier (Refining) ###
#     print('\n=====Training Using Growing Classifier (new classes appeared in \'Other\' class before) ======\n')
#     weights_growing_classifier_refining = list(initial_weights)
#     # use the same initial weights
#     model.set_weights(weights_growing_classifier_refining)
#     for layer in model.layers: layer.trainable = True

#     acc_GC_refining, loss_GC_refining = training_with_GC(model, full_target_class_ids, base_epochs, 
#                                                          x_train, y_train, x_test, y_test, 
#                                                                        datagen, batch_size, True)

#     ### Growing Classifier ###
#     print('\n=====Training Using Growing Classifier (new classes never appeared in \'Other\' class before) ======\n')
#     weights_growing_classifier = list(initial_weights)
#     # use the same initial weights
#     model.set_weights(weights_growing_classifier)

#     for layer in model.layers: layer.trainable = True

#     acc_GC, loss_GC = training_with_GC(model, full_target_class_ids, base_epochs, 
#                                                     x_train, y_train, x_test, y_test, 
#                                                      datagen, batch_size, False)
    
#     plot_result(acc_all_class_from_scratch, acc_GC, acc_GC_refining, full_target_class_ids, class_names)
    return acc_all_class_from_scratch, acc_GC_refining, acc_GC