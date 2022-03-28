import numpy as np
import pathlib

def get_mat(list_of_dics):
    max_x, max_y = 0, 0
    list_of_ends = []
    for dic in list_of_dics:
        list_of_ends.append(dic.reshape((1,))[0].pop((-1, 0), 0))
        max_x = np.max([max_x, np.max(np.array(list(dic.reshape((1,))[0].keys()))[:, 0], axis=0)])
        max_y = np.max([max_y, np.max(np.array(list(dic.reshape((1,))[0].keys()))[:, 1], axis=0)])
    mats = [np.zeros((max_x+1, max_y+1)) for _ in list_of_dics]
    charges = [0 for _ in list_of_dics]
    for i, (dic, end) in enumerate(zip(list_of_dics, list_of_ends)):
        for key, val in dic.reshape((1,))[0].items():
            mats[i][key] = val
        charges[i] = np.sum(np.real(end))
    return np.array(mats), np.array(charges)

all_arrs = {str(x.stem):get_mat(np.load(str(x), allow_pickle=True)) for x in pathlib.Path('../').glob('**/*get_measurement_record*.npy') if not str(x.stem).startswith('file')}
arrs = np.array([x[1][0] for x in sorted([(float(x.split(',')[0]), y) for x, y in all_arrs.items()])])
charges = np.array([x[1][1] for x in sorted([(float(x.split(',')[0]), y) for x, y in all_arrs.items()])])
ps = np.array([x[0] for x in sorted([(float(x.split(',')[0]), y) for x, y in all_arrs.items()])])

print(charges)
def mask_arrs(arrs):
    min_p = np.min(ps)
    masked_arrs = []
    for p, arr in zip(ps, arrs):
        del_prob = 1-min_p/p
        masking_arr = np.random.choice([0, 1], size=(len(arr), 79, 40), p=[del_prob, 1-del_prob])
        masked_arr = masking_arr*arr
        masked_arrs.append(masked_arr)
    return np.array(masked_arrs)


raise Exception
p_late = masked_arrs[:, :, 39:, :]
p_early = masked_arrs[:, :, 39:, :]

x_train = np.concatenate([p_late[0], p_late[1], p_late[4], p_late[5]], axis=0)
y_train = np.concatenate([np.zeros((400, 1)), np.ones((400, 1))], axis=0)

perm = np.arange(len(x_train))
np.random.shuffle(perm)

x_train = x_train[perm]
y_train = y_train[perm]

x_val, x_train = x_train[600:], x_train[:600]
y_val, y_train = y_train[600:], y_train[:600]

import tensorflow as tf

mlp = tf.keras.Sequential([tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer = tf.keras.regularizers.l2(0.01))])

mlp.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mlp.fit(x_train, y_train, validation_data = [x_val, y_val], epochs=10)
