#coding:utf-8
from __future__ import division

import inspect
import numpy as np
import random
import tensorflow as tf

from beam import BeamSearch
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

flags = tf.flags

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class LMModel(object):
  """The LM model."""

  def __init__(self, is_training, config):

    self.batch_size=batch_size = config.batch_size
    self.num_steps=num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    def lstm_cell():
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    
    self.cell=cell
    #self.cell.zero_state
    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding,self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    self._predicts=tf.nn.softmax(logits)
    self._log_prob=tf.log(self._predicts,name="log_prob")
    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
			  self._targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )

    # update the cost variables
    self._cost = cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
  
  @property
  def input_data(self):
    return self._input_data
  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost
  
  @property
  def predicts(self):
    return self._predicts
  
  @property
  def log_prob(self):
    return self._log_prob
      
  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  def sample(self, sess, words, vocab, num=200, prime=None, sampling_type=1,width=4):
    def beam_search_predict(sample, state):
      """Returns the updated probability distribution (`probs`) and
      `state` for a given `sample`. `sample` should be a sequence of
      vocabulary labels, with the last word to be tested against the RNN.
      """
      x = np.zeros((1, 1))
      x[0, 0] = sample[-1]
      feed = {self.input_data: x, self.initial_state: state}
      [probs, final_state] = sess.run([self.predicts, self._final_state],feed)
      
      return probs, final_state

    def beam_search_pick(prime, width):
      """Returns the beam search pick."""
      if not len(prime) or prime == ' ':
        prime = random.choice(list(vocab.keys()))
      prime_labels = [vocab.get(word.decode("utf-8"),0) for word in prime.split()]
      bs = BeamSearch(beam_search_predict,
                       sess.run(self.cell.zero_state(1, tf.float32)),
                       prime_labels)
      samples, scores = bs.search(None, None, k=width, maxsample=num)
      print np.shape(samples)
      print samples
      print np.shape(scores)
      print scores
      print np.argmin(scores)
      return samples[np.argmin(scores)]

    ret = ''
    print prime
    pred = beam_search_pick(prime, width)
    print pred
    for i, label in enumerate(pred):
      ret += ' ' + words[label] if i > 0 else words[label]
    
    return ret
