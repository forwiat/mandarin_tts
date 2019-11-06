import os
import numpy as np
import tensorflow as tf
from modules import get_next_batch, acoustic_model
from hyperparams import hyperparams
hp = hyperparams()
class Acoustic_Graph:
    def __init__(self, mode='train', load=True):
        self.mode = mode.lower()
        if self.mode not in ['train', 'test', 'infer']:
            raise Exception(f'No supported mode {mode}. Please check.')
        self.is_training = False
        if self.mode is 'train':
            self.is_training = True
        self.out_dim = hp.SYN_OUT_DIM
        self.scope_name = 'acoustic_net'
        self.reuse = tf.AUTO_REUSE
        self.dir = hp.SYN_TF_DIR
        self.build_model()
        self.show_info()
        self.saver = tf.train.Saver()
        self.gpu_options = tf.GPUOptions(per_process_fraction=1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, allow_soft_placement=True))
        if self.mode is 'train':
            self.writer = tf.summary.FileWriter(hp.SYN_LOG_DIR, self.sess.graph)
            tf.summary.scalar('{}/loss'.format(self.mode), self.loss)
            tf.summary.scalar('{}/loss'.format(self.mode), self.lr)
            self.merged = tf.summary.merge_all()
        if load is True:
            self.loaded = False
            try:
                print(f'Try to load trainded model in {hp.SYN_MODEL_DIR} ...')
                self.saver.restore(self.sess, hp.SYN_MODEL_DIR)
                self.loaded = True
            finally:
                if self.loaded is True:
                    print('Successfully loaded.')
                elif self.loaded is False and self.mode is 'train':
                    print(f'Loading trained model failed or No trainded model in {hp.SYN_MODEL_DIR}. Start training with initializer ...')
                elif self.loaded is False and self.mode in ['test', 'infer']:
                    raise Exception(f'Loading trained model failed or No trainded model in {hp.SYN_MODEL_DIR}. Please check.')

    def build_model(self):
        self.global_steps = tf.get_variable('global_steps', initializer=0, dtype=tf.int32, trainable=False)
        self.lr = tf.train.exponential_decay(hp.SYN_LR,
                                             decay_steps=hp.SYN_LR_DECAY_STEPS,
                                             decay_rate=hp.SYN_LR_DECAY_RATE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.mode in ['train', 'test']:
            self.x, self.y = get_next_batch(self.dir, mode=self.mode, type='acoustic')
        else:
            self.x = tf.placeholder(shape=[None, hp.SYN_IN_DIM], dtype=tf.float32, name='syn_lab')
        self.y_hat = acoustic_model(self.x, size=self.out_dim, scope=self.scope_name, reuse=self.reuse)
        self.loss = tf.reduce_mean(tf.square(self.y_hat - self.y))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_steps)

    def show_info(self):
        self.t_vars = tf.trainable_variables()
        self.num_paras = 0
        for var in self.t_vars:
            var_shape = var.get_shape().as_list()
            self.num_paras += np.prod(var_shape)
        print("Acoustic model total number of trainable parameters : %r" % (self.num_paras))

    def train(self):
        _, y_hat, loss, summary, steps = self.sess.run((self.train_op, self.y_hat, self.loss, self.merged, self.global_steps))
        self.writer.add_summary(summary, steps)
        if steps % (hp.SYN_PER_STEPS + 1) == 0:
            self.saver.save(self.sess, os.path.join(hp.SYN_MODEL_DIR, f'syn_model_{steps}steps_{loss}los'))
        return y_hat, loss, steps

    def test(self):
        acoustic_features, loss, steps = self.sess.run((self.y_hat, self.loss, self.global_steps))
        return acoustic_features, loss, steps

    def infer(self, syn_lab):
        self.sess.run(tf.global_variables_initializer())
        acoustic_features = self.sess.run(self.y_hat, feed_dict={self.x: syn_lab})
        return acoustic_features