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

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  test_data= raw_data
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  config_gpu = tf.ConfigProto(allow_soft_placement=True)
  with tf.Graph().as_default(),tf.Session(config=config_gpu) as sess:
    initializer = tf.random_uniform_initializer(-eval_config.init_scale,eval_config.init_scale) 

    with tf.name_scope("Test"):
      with tf.variable_scope("model", reuse=None,initializer=initializer):
        mtest = LMModel(is_training=False, config=eval_config)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    checkpoint=tf.train.latest_checkpoint(FLAGS.save_path)
    saver.restore(sess,checkpoint)
    
    test_perplexity = do_eval(sess, mtest, test_data)
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
