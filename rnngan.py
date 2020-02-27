import tensorflow as tf
import tensorflow_gan as tfgan
import time
import numpy as np
import utils
from tensorflow.contrib.gan.python import namedtuples


class LSTMTFGAN:
    def __init__(self, dataset, noise_input, args):
        self.dataset = dataset
        self.noise_input = noise_input
        self.args = args

        self._build_tf_gan_model()
        self.saver = tf.train.Saver()
    
    def _build_tf_gan_model(self):
        self.iterator = tf.data.make_initializable_iterator(self.dataset)
        
        # define the generator function
        def generator_fn(noise, weight_decay=0.0, is_training=True):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.args.num_units)
            enc_rnn_outputs, enc_rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=noise
            )

            generated_seq = tf.compat.v1.layers.dense(enc_rnn_outputs, 1, activation=tf.nn.tanh)

            return generated_seq

        # define the discriminator function
        def discriminator_fn(img, unused_conditioning, weight_decay=0.0, is_training=True):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.args.num_units)
            dis_rnn_outputs, dis_rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=img
            )

            dis_logits = tf.compat.v1.layers.dense(dis_rnn_outputs, 1, activation=None)

            return dis_logits


        self.gan_model = tfgan.gan_model(
                            generator_fn,
                            discriminator_fn,
                            real_data=self.iterator.get_next(),
                            generator_inputs=self.noise_input)

        # define the loss function
        if self.args.gan_loss == 'min_max':
            self.gan_loss = tfgan.gan_loss(
                self.gan_model,
                generator_loss_fn=tfgan.losses.minimax_generator_loss,
                discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)
        else:
            # set default gan loss as wasserstein loss
            self.gan_loss = tfgan.gan_loss(
                self.gan_model,
                generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
                discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

        # define the optimizer
        generator_optimizer = tf.train.AdamOptimizer(self.args.g_lr, beta1=0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(self.args.d_lr, beta1=0.5)

        self.gan_train_ops = tfgan.gan_train_ops(
            self.gan_model,
            self.gan_loss,
            generator_optimizer,
            discriminator_optimizer)

        # define the inference network
        with tf.variable_scope('Generator', reuse=True):
            eval_sequence = self.gan_model.generator_fn(
                tf.random_normal([self.args.num_eval, self.args.sequence_length, self.args.noise_dim]),
                is_training=False)

            self.eval_sequence = eval_sequence

        with tf.variable_scope('inference', reuse=True):
            self.input_noise = tf.placeholder(shape=[None, None, self.args.noise_dim], dtype=tf.float32)
            self.inference_sequence = self.gan_model.generator_fn(self.input_noise)

    def inference(self, sess):
        eval_ret = sess.run(self.eval_sequence)
        eval_ret = np.array(eval_ret)

        return eval_ret

    # generate data through input noise
    def generate_data(self, sess, noise):
        generated_data = sess.run(self.inference_sequence, feed_dict={self.input_noise:noise})
        generated_data = np.array(generated_data)

        return generated_data

    # generate data through input noise
    def train(self, visualization=True):
        # define the training step function, In each training loop, train generator 1 time and discrimintor 5 times
        train_step_fn = tfgan.get_sequential_train_steps(
            train_steps=namedtuples.GANTrainSteps(self.args.generator_train_steps, self.args.discriminator_train_steps))

        global_step = tf.train.get_or_create_global_step()
        loss_values = []
        gen_loss_values = []
        dis_loss_values = []

        with tf.Session() as sess:
            start_time = time.time()
            sess.run(self.iterator.initializer)
            
            i = 1
            step=0
            while i < self.args.epoch+1:
                try:
                    cur_loss, _ = train_step_fn(
                        sess, self.gan_train_ops, global_step, train_step_kwargs={})
                    
                    loss_values.append(cur_loss)
                    
                except tf.errors.OutOfRangeError:
                    i+=1
                    sess.run(self.iterator.initializer)
                    
                    if i % self.args.interval == 0:
                        eval_ret = self.inference(sess)
                        print('Current loss: %f' % cur_loss)
                        if visualization == True:
                            utils.visualize_training_generator(step, start_time, eval_ret, 5, 5)
            
                if step == 0:
                    eval_ret = self.inference(sess)
                    print('Current loss: %f' % cur_loss)
                    self.save_model(sess, ('models/GAN_model_'+str(i)))
                    print("Successfully save the model!")
                    if visualization:
                        utils.visualize_training_generator(step, start_time, eval_ret, 5, 5)
                step += 1

        if visualization:
            x_data = np.arange(len(loss_values))
            utils.plot_sequence_data(x_data, loss_values, x_label='steps', y_lable='loss values')


    def save_model(self, sess, path):
        self.saver.save(sess, path)

    def load_model(self, sess, load_path):
        self.saver.restore(sess, load_path)
if __name__ == "__main__":
    tfgan.infogan_model()