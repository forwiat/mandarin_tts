import tensorflow as tf
from hyperparams import hyperparams
hp = hyperparams()
import os
import glob
def get_next_batch(dir: str, mode='train', type='duration'):
    '''
    :param dir: String. TFRecord files directory.
    :param mode: String. Mode in ['train', 'test']
    :param type: String. In ['duration', 'acoustic']
    :return: Batched tensor.
    '''
    if mode not in ['train', 'test']:
        raise Exception(f'#-------------------------No supported mode {mode}. Please check.-------------------------#')
    def _parse_function(proto):
        parse_dics = {
            'x': tf.VarLenFeature(dtype=tf.float32),
            'x_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'y': tf.VarLenFeature(dtype=tf.float32),
            'y_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
        }
        parsed_example = tf.parse_single_example(proto, parse_dics)
        parsed_example['x'] = tf.sparse_tensor_to_dense(parsed_example['x'])
        parsed_example['y'] = tf.sparse_tensor_to_dense(parsed_example['y'])
        parsed_example['x'] = tf.reshape(parsed_example['x'], parsed_example['x_shape'])
        parsed_example['y'] = tf.reshape(parsed_example['y'], parsed_example['y_shape'])
        return parsed_example
    total_tf = glob.glob(f'{dir}/*.tfrecord')
    tf_files = []
    for i in total_tf:
        if type == 'duration' and os.path.basename(i).find(f'_dur_{mode}.tfrecord') != -1:
            tf_files.append(i)
        elif type == 'acoustic' and os.path.basename(i).find(f'_syn_{mode}.tfrecord') != -1:
            tf_files.append(i)
    dataset = tf.data.TFRecordDataset(tf_files)
    parsed_dataset = dataset.map(_parse_function)
    if type == 'duration':
        bn = hp.DUR_BATCH
        x_dim = hp.DUR_LAB_DIM
        y_dim = hp.DURATION_DIM
        num_epoch = hp.DUR_EPOCH
    else:
        bn = hp.SYN_BATCH
        x_dim = hp.SYN_LAB_DIM
        y_dim = hp.ACOUSTIC_DIM
        num_epoch = hp.SYN_EPOCH
    batched_dataset = parsed_dataset.padded_batch(
        batch_size=bn,
        padded_shapes={
            'x': [None, x_dim],
            'x_shape': [None],
            'y': [None, y_dim],
            'y_shape': [None]
        }
    )
    shuffled_dataset = batched_dataset.shuffle(buffer_size=bn)
    epoched_dataset = shuffled_dataset.repeat(num_epoch)
    iterator = epoched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element['x'], next_element['y']

def prenet(inputs,
           num_units=None,
           is_training=True,
           scope='prenet',
           reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D].
    :param num_units: A list of two integers or None.
    :param is_training: Boolean.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: [N, T, num_units]
    '''
    D = inputs.get_shape().as_list()[-1]
    if num_units is None:
        num_units = [D, D]
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name='dense_1')
        outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name='dropout_1')
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name='dense_2')
        outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name='dropout_2')
    return outputs

def conv1d(inputs,
           filter_nums=None,
           kernel_size=1,
           padding='SAME',
           activation_fn=None,
           data_format='channels_last',
           scope='conv1d',
           reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D].
    :param filter_nums: An integer. Filters.
    :param kernel_size: An integer.
    :param padding: String. In ['SAME', 'VALID'].
    :param activation_fn: TF activation function.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: If padding is 'SAME' then [N, T, filter_nums] else ...
    '''
    if filter_nums is None:
        filter_nums = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(inputs, filters=filter_nums, kernel_size=kernel_size, padding=padding,
                                   activation=activation_fn, data_format=data_format, name='conv1d')
    return outputs

def highway(inputs,
            num_units=None,
            scope='highway',
            reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D].
    :param num_units: An integer or None.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: A 3-D tensor. [N, T, D].
    '''
    if num_units is None:
        num_units = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name='dense_H')
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name='dense_T')
        outputs = H * T + inputs * (1. - T)
    return outputs

def gru(inputs,
        num_units=None,
        bidirection=False,
        scope='gru',
        reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D].
    :param num_units: An integer.
    :param bidirection: Boolean.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: [N, T, num_units]
    '''
    if num_units is None:
        num_units = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        cell = tf.nn.rnn_cell.GRUCell(num_units)
        if bidirection:
            bw_cell = tf.nn.rnn_cell.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, bw_cell, inputs=inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs=inputs, dtype=tf.float32)
            return outputs

def bn(inputs,
       axis=-1,
       is_training=True,
       activation_fn=None,
       scope='bn',
       reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D]. (By default axis=-1)
    :param axis: An integer. Do normalization in axis.
    :param is_training: Boolean.
    :param activation_fn: TF activation function or None.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: A 3-D tensor. [N, T, D].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.batch_normalization(inputs, axis=axis, training=is_training)
        if activation_fn != None:
            outputs = activation_fn(outputs)
        return outputs

def conv1d_banks(inputs,
                 K=16,
                 size=None,
                 is_training=True,
                 scope='conv1d_banks',
                 reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D]
    :param K: An integer. conv1d bank nums.
    :param size: An integer. Total banks filter nums.
    :param is_training: Boolean.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return:
    '''
    if size is None:
        size = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, filter_nums=size, kernel_size=1, scope='conv1d_1')
        for i in range(2, K+1):
            _outputs = conv1d(inputs, filter_nums=size, kernel_size=i, scope=f'conv1d_{i}')
            outputs = tf.concat((outputs, _outputs), axis=-1)
        outputs = bn(outputs, is_training=is_training, scope='bn', reuse=reuse)
        return outputs

def duration_model(inputs, size: int, is_training=True, scope='duration', reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D]
    :param size: An integer.
    :param is_training: Boolean. Training or not.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: [N, T, size].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=hp.DUR_IN_DIM//2, activation=tf.nn.relu, name='fc_1')
        outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name='dropout_1')
        for i in range(2, hp.DUR_FC_NUM+1):
            outputs = tf.layers.dense(outputs, units=hp.DUR_IN_DIM//2, activation=tf.nn.relu, name=f'fc_{i}')
            outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name=f'dropout_{i}')
        outputs = tf.layers.dense(outputs, units=size, name='fc_outputs')
        return outputs

def acoustic_model(inputs, size: int, is_training=True, scope='acoustic', reuse=None):
    '''
    :param inputs: A 3-D tensor. [N, T, D].
    :param size: An integer.
    :param is_training: Boolean. Training or not.
    :param scope: String. Scope name.
    :param reuse: False or True or tf.AUTO_REUSE.
    :return: A 3-D tensor. [N, T, size]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        prenet_outputs = prenet(inputs, [hp.SYN_IN_DIM, hp.SYN_IN_DIM//2], is_training, scope='prenet')
        outputs = conv1d_banks(prenet_outputs, hp.SYN_K, hp.SYN_IN_DIM//2, is_training, scope='conv1d_banks')
        outputs = tf.layers.max_pooling1d(outputs, pool_size=2, strides=1, padding='SAME')
        outputs = conv1d(outputs, hp.SYN_IN_DIM//2, kernel_size=3, scope='fixed_conv1d_1')
        outputs = bn(outputs, axis=-1, is_training=is_training, scope='fixed_bn_1')
        outputs = conv1d(outputs, hp.SYN_IN_DIM//2, kernel_size=3, scope="fixed_conv1d_2")
        outputs = bn(outputs, axis=-1, is_training=is_training, scope='fixed_bn_2')
        outputs += prenet_outputs
        for i in range(hp.SYN_HIAHWAY_BLOCK):
            outputs = highway(outputs, num_units=hp.SYN_IN_DIM//2, scope=f'highwaynet_{i}')
        outputs = gru(outputs, hp.SYN_IN_DIM//2, False, scope='gru')
        outputs = tf.layers.dense(outputs, units=size, name='fc_outputs')
        return outputs