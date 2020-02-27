import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow.contrib.gan.python import namedtuples

import functools
import numpy as np
import utils

ds = tf.contrib.distributions
layers = tf.contrib.layers
queues = tf.contrib.slim.queues
framework = tf.contrib.framework
slim = tf.contrib.slim

def generate_test_noise_for_infogan(sequence_length=50,
                                    unstructured_categorical_noise_dims=2,
                                    unstructured_continuous_noise_dims=3):
    num_dim_1 = 5
    num_dim_2 = 5
    cont_dim1 = np.linspace(-1.0, 1.0, num_dim_1)
    cont_dim2 = np.linspace(-1.0, 1.0, num_dim_2)

    cont_noise = []
    cat_noise = []
    for i in range(num_dim_1):
        for j in range(num_dim_2):
            cont_noise.append([0.0, cont_dim1[i], cont_dim2[j]])
            cat_noise.append(1)

    cont_noise = np.array(cont_noise, dtype=np.float32)
    cat_noise = np.array(cat_noise, dtype=np.int32)
    unconditional_noise = np.random.normal(size=[25, sequence_length,
                                                 unstructured_continuous_noise_dims])

    unconditional_noise = np.array(unconditional_noise, dtype=np.float32)

    return unconditional_noise, cat_noise, cont_noise

class LSTMTFGANCategory:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self._build_tf_gan_model()
        self.saver = tf.train.Saver()

    def _build_tf_gan_model(self):
        self.iterator = tf.data.make_initializable_iterator(self.dataset)

        # define the generator function
        def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5, is_training=True):
            # infogan generator with conditional noise.
            unstructured_noise, cat_noise, cont_noise = inputs
            cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
            conditional_noise = tf.concat([cat_noise_onehot, cont_noise], axis=-1)

            with framework.arg_scope(
                    [layers.fully_connected],
                    activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                    weights_regularizer=layers.l2_regularizer(weight_decay)), \
                 framework.arg_scope([layers.batch_norm], is_training=is_training):
                cell = tf.contrib.rnn.LSTMCell(self.args.num_units)

                state_c = layers.fully_connected(conditional_noise, self.args.num_units)
                state_h = layers.fully_connected(conditional_noise, self.args.num_units)

                # initialize the initial hidden state of lstm_info_gan
                initial_hidden_state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)

                dis_rnn_outputs, dis_rnn_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    inputs=unstructured_noise,
                    initial_state=initial_hidden_state
                )

                dis_logits = layers.fully_connected(dis_rnn_outputs, 1, activation_fn=None)

                return dis_logits

        def infogan_discriminator(sequence_data, unused_conditioning, weight_decay=2.5e-5,
                                  categorical_dim=10, continuous_dim=2, is_training=True):
            with framework.arg_scope(
                    [layers.fully_connected],
                    activation_fn=tf.nn.relu, normalizer_fn=None,
                    weights_regularizer=layers.l2_regularizer(weight_decay),
                    biases_regularizer=layers.l2_regularizer(weight_decay)):
                # define the dynamic_rnn for lstm
                cell = tf.contrib.rnn.LSTMCell(self.args.num_units)
                dis_rnn_outputs, dis_rnn_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float32,
                    inputs=sequence_data
                )
                logits_real = layers.fully_connected(dis_rnn_outputs, 1, activation_fn=None)

                # Recognition network for latent variables has an additional layer
                with framework.arg_scope([layers.batch_norm], is_training=is_training):
                    output_state = tf.concat(dis_rnn_states, axis=-1)
                    encoder = layers.fully_connected(
                        output_state, self.args.num_units, normalizer_fn=layers.batch_norm)

                # When we have stochastic features, we can set the categorical noise dim.
                # Compute logits for each category of categorical latent.
                logits_cat = layers.fully_connected(
                    encoder, categorical_dim, activation_fn=None)
                q_cat = ds.Categorical(logits_cat)

                # Compute mean for Gaussian posterior of continuous latents.
                mu_cont = layers.fully_connected(
                    encoder, continuous_dim, activation_fn=None)
                sigma_cont = tf.ones_like(mu_cont)
                q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

                return logits_real, [q_cat, q_cont]

        generator_fn = functools.partial(infogan_generator, categorical_dim=self.args.structured_categorical_dim)
        discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=self.args.structured_continuous_dim,
                                             categorical_dim=self.args.structured_categorical_dim)

        unstructured_inputs, structured_inputs = self._get_infogan_noise()

        # Create the infogan model.
        self.infogan_model = tfgan.infogan_model(
                        generator_fn=generator_fn,
                        discriminator_fn=discriminator_fn,
                        real_data=self.iterator.get_next(),
                        unstructured_generator_inputs=unstructured_inputs,
                        structured_generator_inputs=structured_inputs)

        # define the loss function
        if self.args.gan_loss == 'min_max':
            self.gan_loss = tfgan.gan_loss(
                self.infogan_model,
                generator_loss_fn=tfgan.losses.minimax_generator_loss,
                discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
                mutual_information_penalty_weight=self.args.mutual_information_penalty_weight) # define the mutual infomation penalty weight.
        else:
            # set default gan loss as wasserstein loss
            self.gan_loss = tfgan.gan_loss(
                self.infogan_model,
                generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
                discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
                mutual_information_penalty_weight=self.args.mutual_information_penalty_weight)

        # define the optimizer
        generator_optimizer = tf.train.AdamOptimizer(self.args.g_lr, beta1=0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(self.args.d_lr, beta1=0.5)

        self.gan_train_ops = tfgan.gan_train_ops(
            self.infogan_model,
            self.gan_loss,
            generator_optimizer,
            discriminator_optimizer)

        # define the inference network
        with tf.variable_scope('Generator', reuse=True):
            self.unconditional_noise = tf.placeholder(dtype=tf.float32, shape=[None, None, self.args.unstructured_continuous_noise_dims])
            self.cat_noise = tf.placeholder(dtype=tf.int32, shape=[None])
            self.cont_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.args.structured_continuous_dim])

            noise = [self.unconditional_noise, self.cat_noise, self.cont_noise]
            eval_sequence = self.infogan_model.generator_fn(
                noise,
                is_training=False)

            self.eval_sequence = eval_sequence

    def _get_infogan_noise(self):
        """Get unstructured and structured noise for InfoGAN.
        Args:
        batch_size: The number of noise vectors to generate.
        categorical_dim: The number of categories in the categorical noise.
        structured_continuous_dim: The number of dimensions of the uniform
          continuous noise.
        total_continuous_noise_dims: The number of continuous noise dimensions. This
          number includes the structured and unstructured noise.
        Returns:
        A 2-tuple of structured and unstructured noise. First element is the
        unstructured noise, and the second is a 2-tuple of
        (categorical structured noise, continuous structured noise).
        """
        # Get unstructurd noise.
        unstructured_noise = tf.random_normal(
          [self.args.batch_size, self.args.sequence_length, self.args.unstructured_continuous_noise_dims])

        # If there are categorical noise, we create the categorical dim.

        categorical_dist = ds.Categorical(logits=tf.zeros([self.args.structured_categorical_dim]))
        categorical_noise = categorical_dist.sample([self.args.batch_size])

        # Get continuous noise Tensor.
        continuous_dist = ds.Uniform(-tf.ones([self.args.structured_continuous_dim]),
                                      tf.ones([self.args.structured_continuous_dim]))
        continuous_noise = continuous_dist.sample([self.args.batch_size])

        # The return must be vectors, infogan model recieves these return and add two vector, then pass to the generator.
        return [unstructured_noise], [categorical_noise, continuous_noise]

    def inference(self, sess, unconditional_noise, cat_noise, cont_noise):
        eval_ret = sess.run(self.eval_sequence, feed_dict={self.unconditional_noise: unconditional_noise,
                                                           self.cat_noise: cat_noise,
                                                           self.cont_noise: cont_noise})
        eval_ret = np.array(eval_ret)

        return eval_ret

    def train(self, save_path, test_noise_generator, visualization=True):
        # define the training step function, In each training loop, train generator 1 time and discrimintor 5 times
        train_step_fn = tfgan.get_sequential_train_steps(
            train_steps=namedtuples.GANTrainSteps(self.args.generator_train_steps, self.args.discriminator_train_steps))

        global_step = tf.train.get_or_create_global_step()
        loss_values = []

        unconditional_noise, cat_noise, cont_noise = test_noise_generator()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)
            for i in range(self.args.epoch):
                while True:
                    try:
                        cur_loss, _ = train_step_fn(
                            sess, self.gan_train_ops, global_step, train_step_kwargs={})
                        loss_values.append((i, cur_loss))
                    except tf.errors.OutOfRangeError:
                        sess.run(self.iterator.initializer)
                        break

                if i % self.args.interval == 0:
                    display_seq_np = self.inference(sess, unconditional_noise, cat_noise, cont_noise)
                    utils.plot_sequence(display_seq_np, 5, 5, fig_title='Sample Sequence')
                    self.save_model(sess,path=(save_path+str(i)))
                    print("Successfully save infogan model!")
                    print('Current loss: %f' % cur_loss)

        self.save_model(sess, path=(save_path + str(self.args.epoch)))
        print("Successfully save infogan model!")

    def save_model(self, sess, path):
        self.saver.save(sess, path)

    def load_model(self, sess, load_path):
        self.saver.restore(sess, load_path)

if __name__ == "__main__":
    hparams = tf.contrib.training.HParams(
        is_training=True,
        batch_size=50,
        buffer_size=14000,
        sequence_length=50,
        epoch=200,
        structured_continuous_dim=3,
        structured_categorical_dim=2,
        unstructured_continuous_noise_dims=3,
        mutual_information_penalty_weight=1.0,

        num_units=128,
        gan_loss='min_max',
        d_lr=1e-4,
        g_lr=1e-3,
        generator_train_steps=1,
        discriminator_train_steps=4,
        interval=40,
        num_eval=128
    )

    train_sequence = utils.sine_wave(seq_length=hparams.sequence_length, num_samples=hparams.buffer_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_sequence).shuffle(hparams.buffer_size).batch(
        hparams.batch_size)

    lstm_gan = LSTMTFGANCategory(train_dataset, hparams)

    unconditional_noise, cat_noise, cont_noise = generate_test_noise_for_infogan()

    print(unconditional_noise)
    print()
    print(cat_noise)
    print()
    print(cont_noise)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        generated_sequence = lstm_gan.inference(sess=sess,
                       unconditional_noise=unconditional_noise,
                       cat_noise=cat_noise,
                       cont_noise=cont_noise)

        print(np.array(generated_sequence).shape)