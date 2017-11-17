#coding:utf-8
from __future__ import absolute_import
from __future__ import division

import time
import sys
import numpy as np
import tensorflow as tf

from model import LMModel
import reader
from config import LMConfig

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", 'data',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'save/',
                    "Model output directory.")

FLAGS = flags.FLAGS

def get_config():
  return LMConfig()

def do_eval(session, m, data):
  costs = 0.0
  iters = 0
  state = session.run(m.initial_state)
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,m.num_steps)):
    
    feed={m.input_data:x,m.targets:y,m.initial_state:state}

    cost, state = session.run([m.cost, m.final_state],feed_dict=feed)
    costs += cost
    iters += m.num_steps
  return np.exp(costs / iters)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path,True)
  train_data, valid_data, test_data, vocab_size = raw_data
  print ('vocab size is %d'% vocab_size)
  config = get_config()
  config.vocab_size=vocab_size
  eval_config = get_config()
  eval_config.vocab_size=vocab_size
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  config_gpu = tf.ConfigProto(allow_soft_placement=True)
  #config_gpu.gpu_options.allow_growth=True
  with tf.Graph().as_default(),tf.Session(config=config_gpu) as sess:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale) 
    with tf.name_scope("Train"):
      with tf.variable_scope("model", reuse=None,initializer=initializer):
        m = LMModel(is_training=True, config=config)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)
      
    with tf.name_scope("Valid"):
      with tf.variable_scope("model", reuse=True,initializer=initializer):
        mvalid = LMModel(is_training=False, config=config)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("model", reuse=True,initializer=initializer):
        mtest = LMModel(is_training=False, config=eval_config)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    checkpoint=tf.train.latest_checkpoint(FLAGS.save_path)
    best_eval=float('inf')
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(sess, config.learning_rate * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
      epoch_size = ((len(train_data) // m.batch_size) - 1) // m.num_steps
      start_time = time.time()
      costs = 0.0
      iters=0
      state= sess.run(m.initial_state)
      for step, (x, y) in enumerate(reader.ptb_iterator(train_data, m.batch_size,m.num_steps)):
        feed={m.input_data:x,m.targets:y,m.initial_state:state}
        cost, state, _ = sess.run([m.cost, m.final_state, m.train_op],feed_dict=feed)
        costs += cost
        iters += m.num_steps
        
        if step % (epoch_size // 10) == 10:
          print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))
          
        #save model
        if step % 40000 == 0:
          print("Saving model to %s." % FLAGS.save_path)
          saver.save(sess, FLAGS.save_path,global_step=step)
      valid_perplexity=do_eval(sess, mvalid, valid_data)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    test_perplexity = do_eval(sess, mtest, test_data)
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
