class LMConfig(object):
  """Languange Model config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 3
  num_steps = 20
  hidden_size = 512# 256
  max_epoch = 4
  max_max_epoch = 15
  keep_prob = 0.8
  lr_decay = 0.5
  batch_size = 128
  vocab_size = 9620 
#  vocab_size = 9484 
