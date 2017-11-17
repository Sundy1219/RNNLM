#coding:utf-8
from __future__ import absolute_import
from __future__ import division

import numpy as np
import collections
import os
import sys
from six.moves import cPickle
import tensorflow as tf

try:
  reload(sys)
  sys.setdefaultencoding('utf-8')
except:
  pass

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", " <eos> ").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
  
  return word_to_id,id_to_word

def dump_vocab(filename,dicts):
  with open(filename,'wb') as f:
	  cPickle.dump(dicts,f)

def load_vocab(filename):
  with open(filename,'rb') as f:
    return cPickle.load(f)

def get_vocab_size(vacab):
  return len(vacab)

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def id_to_words(filename,words_id):
  id_to_word=load_vocab(filename)
  return [id_to_word[word_id] for word_id in words_id if word_id in id_to_word]

def ptb_raw_data(data_path=None,is_training=False):
  """Load Chinese raw data from data directory "data_path".
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  
  train_path = os.path.join(data_path, "train.txt")
  valid_path = os.path.join(data_path, "valid.txt")
  test_path = os.path.join(data_path, "test.txt")
  if is_training==True:
    word_to_id,id_to_word = _build_vocab(train_path)
    dump_vocab('vocab/word_to_id.pkl',word_to_id)
    dump_vocab('vocab/id_to_word.pkl',id_to_word)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocab_size=len(word_to_id) 
    
    return train_data, valid_data, test_data,vocab_size
  
  else:
    word_to_id=load_vocab('vocab/word_to_id.pkl')

    test_data = _file_to_word_ids(test_path, word_to_id)
    return test_data

def ptb_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def word_scores(log_probs,sentence):
  word_scores=[]
  counter=0
  while True:
    if sentence[counter+1]==0:
      word_scores.append(log_probs[counter][sentence[counter+1]])
      break
    else:
      word_scores.append(log_probs[counter][sentence[counter+1]])
    counter+=1
	
  return word_scores

def add(x,y):
  return x+y

def sentence_score(log_probs,sentence):
  """Computer score of a sentence
  """
  scores=word_scores(log_probs,sentence)
  score=reduce(add,scores)
  
  return score

def write_word_scores(logprobs,sentence):#,output_file)#,log_scale):
  """Writes word-level scores to an output file
	"""
  logprobs=word_scores(logprobs,sentence)
  words=id_to_words('vocab/id_to_word.pkl',sentence)
  if len(logprobs)!=len(words)-1:
    raise ValueError("Number of logprobs should be exactly one less than" 
                      "the number of words.")

  #logprobs = [None if x is None else x/log_scale for x in logprobs]
  for index, logprob in enumerate(logprobs):
    if index-2>0:
      history_list = ['...']
      history_list.extend(words[index-2:index+1])
    else:
      history_list = words[:index+1]
    history = ' '.join(history_list)
    predicted = words[index+1]
    
    if logprob is None:
      print ("p({0} | {1} is not predicted\n".format(
           predicted,history))
    else:
      print ("log(p({0} |{1})) = {2}".format(
            predicted,history,logprob))

