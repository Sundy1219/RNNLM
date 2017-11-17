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
flags.DEFINE_string("save_path", 'save',
                    "Model output directory.")
FLAGS = flags.FLAGS


def get_config():
  return LMConfig()

def do_eval(sess, m, data):
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = sess.run(m.initial_state)
  score=[]
  probs=[]
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,m.num_steps)):
    
    feed={m.input_data:x,m.targets:y,m.initial_state:state}

    prob,log_prob,cost, state = sess.run([m.predicts,m.log_prob,m.cost, m.final_state],feed_dict=feed)
    costs += cost
    probs.append(prob[0])
    score.append(log_prob[0])
    iters += m.num_steps
    
  return probs,score,np.exp(costs / iters)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  test_data = raw_data
  lists=[]
  k=0
  for l in range(len(test_data)):
    if test_data[l]==0:
      lists.append(test_data[k:l+1])
      k=l+1
  config = get_config()
  config.num_steps=1
  config.batch_size=1

  config_gpu = tf.ConfigProto(allow_soft_placement=True)
  
  with tf.Graph().as_default(),tf.Session(config=config_gpu) as sess:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    with tf.name_scope("Test"):
      with tf.variable_scope("model", reuse=None, initializer=initializer):
        mtest = LMModel(is_training=False, config=config)

    tf.global_variables_initializer().run()
    saver=tf.train.Saver()
    checkpoint=tf.train.latest_checkpoint(FLAGS.save_path)
    saver.restore(sess,checkpoint)  
    for n in range(len(lists)):
      print lists[n]
      print ("sentence #%d" % n)
      probs,log_probs,test_perplexity = do_eval(sess, mtest, lists[n])
      print "****************************************************"
      print "score of words :"
      
      reader.write_word_scores(log_probs,lists[n])
      print("Perplexity of the sentence is: %.3f" % test_perplexity)
      
      scores=reader.sentence_score(log_probs,lists[n])
      print "Score of the sentence is: {:.6f} ".format(scores)
      print "****************************************************\n"

if __name__ == "__main__":
  tf.app.run()
