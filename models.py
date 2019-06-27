# DA SENTIMENT ANALYSIS/MISCELLANEOUS MODELS ----------------------------------
# 
# Models for zero-shot domain adaptation experiments on sentiment analysis task.


# imports
import torch
from train_utils import write
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# DA_BASELINE -----------------------------------------------------------------
# 
# Baseline model for Zero-Shot DA for sentiment analysis.
# Consists of a sentence encoder (LSTM) and a classifier layer.
# LSTM allows specifying number of layers, dropout, and bidirectionality.

class DA_B(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers, dropout, bi):
    super(DA_B, self).__init__()
    
    # Baseline modules
    out_vects = 2 if bi else 1
    self.lstm = MY_LSTM(in_d, h_d, layers, dropout, bi)
    self.classifier = MY_NL(h_d*out_vects, 1)
  
  # forward propagation
  def forward(self, inputs, targets, lengths):
    
    # set up batch in pytorch
    b_size = len(inputs)
    inputs, indices = batch_setup(inputs, lengths)
    targets = torch.from_numpy(targets).float()
    targets = targets[indices][:,None]
    
    # data --> GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # forward computation
    inputs_proc = self.lstm.forward(inputs, b_size)
    outputs = self.classifier.forward(inputs_proc)
    
    return outputs, targets
  
  # accuracy computation
  def accuracy(self, inputs, targets, lengths, batch_size):
    
    self.eval()
    classfn_acc = 0
    
    # switch off gradients, get accuracy, data if on
    with torch.no_grad():

      if len(inputs) % batch_size == 0: iterations = len(inputs)//batch_size
      else: iterations = len(inputs)//batch_size + 1
      for batch in range(iterations):
        
        # get batch, forward, backward, update
        batch_inputs = inputs[batch*batch_size : (batch+1)*batch_size]
        batch_targets = targets[batch*batch_size : (batch+1)*batch_size]
        batch_lengths = lengths[batch*batch_size : (batch+1)*batch_size]
        
        # forward pass, accuracy for batch
        batch_outputs, batch_targets = self.forward(batch_inputs, batch_targets, batch_lengths)
        classfn_acc += acc_helper(batch_outputs, batch_targets)

    classfn_acc = classfn_acc/len(inputs)
    self.train()
    
    return classfn_acc


# DA_TRANS --------------------------------------------------------------------
# 
# Embedding-transformation model for Zero-Shot DA for sentiment analysis.
# Consists of sentence encoder, transformation, attn-transformation, classifier,
# and domain encoder (the latter only if domain information is sequential).
# Number of layers, dropout, and bidirectionality can be specified
# for sentence and domain encoders.

class DA_TRANS(torch.nn.Module):

  # initialization method
  def __init__(self, in_d, attn_width, dom_seq,
               s_h_d, s_layers, s_dropout, s_bi,
               d_h_d, d_layers, d_dropout, d_bi):
    super(DA_TRANS, self).__init__()
    
    # parts of model that don't depend on the shape of domain information
    s_out_vects = 2 if s_bi else 1
    self.sent_lstm = MY_LSTM(in_d, s_h_d, s_layers, s_dropout, s_bi)
    self.dom_lstm = None
    trans_in = in_d
    
    # If domain information is a sequence of vectors (not a single one)
    if dom_seq:
      d_out_vects = 2 if d_bi else 1
      self.dom_lstm = MY_LSTM(in_d, d_h_d, d_layers, d_dropout, d_bi)
      trans_in =  d_h_d * d_out_vects
    
    # Transformation, attention-transformation, classifier
    self.in_d = in_d
    self.attn_width = attn_width
    self.trans = MY_L(trans_in, in_d*in_d)
    self.attn_trans = MY_L(trans_in, attn_width*(in_d+2) + 1)
    self.classifier = MY_NL(s_h_d*s_out_vects, 1)
  
  # forward propagation
  def forward(self, inputs, targets, lengths,
              doms, dom_lengths, get_data=False):
    
    # create any required modules
    drop = torch.nn.Dropout(p=0.5)

    # set up batch
    b_size = len(inputs)
    inputs, indices, lengths = batch_setup(inputs, lengths, pack=False)
    targets = torch.from_numpy(targets).float()

    targets = targets[indices][:,None]
    
    # prep domain information
    d_size = len(doms)
    if self.dom_lstm != None:
      doms, dom_indices = batch_setup(doms, dom_lengths)
      dom_ind_rev = torch.zeros(len(dom_indices)).long()
      dom_ind_rev[dom_indices] = torch.arange(len(dom_indices)).long()
      dom_ind_rev = dom_ind_rev.to(device)
    else: 
      doms = torch.from_numpy(doms).float()
      doms = doms[indices]
    
    # sent data --> GPU
    doms = doms.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # get domain-specific transformation, attn-transformation vectors
    if self.dom_lstm != None: doms = self.dom_lstm.forward(doms, d_size)
    t, attn_t = self.trans.forward(doms), self.attn_trans.forward(doms)
    if self.dom_lstm != None: transforms, attn_transforms = t[dom_ind_rev][indices], attn_t[dom_ind_rev][indices]
    else: transforms, attn_transforms = t, attn_t

    # apply dropout
    transforms = drop(transforms)
    attn_transforms = drop(attn_transforms)
    
    # get transformed/domain-specific embeddings, attn weights
    trans_inputs = self.transform_inputs(transforms, inputs, lengths)
    attn_weights = self.compute_attn(attn_transforms, inputs, lengths)
    
    # apply attention to take weighted combo of domain specific/generic embeddings
    weighted_inputs = (inputs * attn_weights) + (trans_inputs * (1 - attn_weights))
    
    # pack the padded, transformed input sequence
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(weighted_inputs, lengths, batch_first=True)
    
    # finish processing the sentence
    inputs_proc = self.sent_lstm.forward(packed_inputs, b_size)
    outputs = self.classifier.forward(inputs_proc)
    
    # save data if on
    if get_data: return outputs, targets, [t.clone().numpy(), attn_t.clone().numpy()]
    else: return outputs, targets
  
  # accuracy computation
  def accuracy(self, inputs, targets, lengths, domains,
               doms, dom_lengths, batch_size):
    
    self.eval()
    classfn_acc = 0
    
    # switch off gradients, get accuracy, data if on
    with torch.no_grad():

      if len(inputs) % batch_size == 0: iterations = len(inputs)//batch_size
      else: iterations = len(inputs)//batch_size + 1
      for batch in range(iterations):

        # get batch, forward, backward, update
        b_inputs = inputs[batch*batch_size : (batch+1)*batch_size]
        b_targets = targets[batch*batch_size : (batch+1)*batch_size]
        b_lengths = lengths[batch*batch_size : (batch+1)*batch_size]
        b_domains = domains[batch*batch_size : (batch+1)*batch_size]

        # get domain information for batch domains
        b_doms = doms[b_domains]
        if dom_lengths != None: b_dom_lengths = dom_lengths[b_domains]
        else: b_dom_lengths = None
        
        # forward pass, accuracy for batch
        b_outputs, b_targets = self.forward(b_inputs, b_targets, b_lengths,
                                            b_doms, b_dom_lengths)
        classfn_acc += acc_helper(b_outputs, b_targets)
    
    classfn_acc = classfn_acc/len(inputs)
    self.train()
    
    return classfn_acc
  

  # helper method for getting transformation, applying it
  def transform_inputs(self, transforms, inputs, lengths):
    
    # set up transformed sentences
    trans_inputs = torch.zeros(inputs.size()).float()
    trans_inputs = trans_inputs.to(device)
    
    # iterate over batch sentences, transform them
    for i in range(len(inputs)):
      trans = transforms[i].view(self.in_d, self.in_d)
      trans_inputs[i][:int(lengths[i])] = torch.matmul(inputs[i][:int(lengths[i])], trans)
    
    return trans_inputs
  

  # helper method for getting attn network, applying it
  def compute_attn(self, attn_transforms, inputs, lengths):

    # first & second layer projections, bias vectors
    attn_trans_m1 = attn_transforms[:,:self.attn_width*self.in_d]
    attn_trans_b1 = attn_transforms[:,self.attn_width*self.in_d : self.attn_width*(self.in_d+1)]
    attn_trans_m2 = attn_transforms[:,self.attn_width*(self.in_d+1) : self.attn_width*(self.in_d+2)]
    attn_trans_b2 = attn_transforms[:,self.attn_width*(self.in_d+2):]

    # set up attention weights for word embeddings in batch sentences
    attn_weights = torch.zeros(inputs.size()[0], inputs.size()[1], 1).float()
    attn_weights = attn_weights.to(device)

    # iterate over batch sentences, compute attention weights
    for i in range(len(inputs)):

      # get projections, bias vectors for current domain in proper form
      m1 = attn_trans_m1[i].view(self.in_d, self.attn_width)
      b1 = attn_trans_b1[i].view(1, self.attn_width)
      m2 = attn_trans_m2[i].view(self.attn_width, 1)
      b2 = attn_trans_b2[i].view(1, 1)

      # apply attn network to get weights for current sentence words
      relu, sigmoid = torch.nn.LeakyReLU(), torch.nn.Sigmoid()
      hiddens = relu(torch.matmul(inputs[i][:int(lengths[i])], m1) + b1)
      attn_weights[i][:int(lengths[i])] = sigmoid(torch.matmul(hiddens, m2) + b2)
    
    return attn_weights


# LSTM ------------------------------------------------------------------------
# 
# Creates LSTM module. Applies Xavier initialization to weights.
# Expects inputs as packed sequence. Supports multiple layers, directions,
# and sets up initial values of hidden/cell state as parameters.

class MY_LSTM(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers=1, dropout=0.0, bi=False, lang=False):
    super(MY_LSTM, self).__init__()
    
    # initial states
    self.lang = lang
    self.out_vects = 2 if bi else 1
    self.h_init = torch.nn.Parameter(torch.randn(layers*self.out_vects, 1, h_d).float()) * 0.1
    self.c_init = torch.nn.Parameter(torch.randn(layers*self.out_vects, 1, h_d).float()) * 0.1
    
    # inner lstm
    self.lstm = torch.nn.LSTM(input_size=in_d, hidden_size=h_d, \
                              num_layers=layers, dropout=dropout, bidirectional=bi)
    
    # xavier initialization
    for param in self.lstm.parameters(): 
      torch.nn.init.xavier_normal_(param) if len(param.size()) > 1 else param.data.fill_(0)
  
  # forward propagation
  def forward(self, inputs, b_size):
    
    # set initial states for current batch size
    h_0 = self.h_init.repeat(1, b_size, 1).to(device)
    c_0 = self.c_init.repeat(1, b_size, 1).to(device)
    
    # compute output and return
    h_all, (h_final, _) = self.lstm.forward(inputs, (h_0, c_0))
    if not self.lang: 
      if self.out_vects == 2: final_output = torch.cat((h_final[-2], h_final[-1]), 1) 
      else: final_output = h_final[-1]
    else: final_output = h_all
    return final_output


# LINEAR TRANSFORMATION -------------------------------------------------------
#
# Creates Linear module, applies xavier initialization to weights
# Returns Linear output

class MY_L(torch.nn.Module):
  
  def __init__(self, in_d, out_d):
    super(MY_L, self).__init__()
    self.linear = torch.nn.Linear(in_features=in_d, out_features=out_d)
    for param in self.linear.parameters():
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
      else: param.data.fill_(0)
  
  def forward(self, inputs):
    return self.linear.forward(inputs)


# NON-LINEAR TRANSFORMATION ---------------------------------------------------
#
# Creates Linear module, applies xavier initialization to weights.
# Returns sigmoid fn applied to Linear output.

class MY_NL(torch.nn.Module):
  
  def __init__(self, in_d, out_d):
    super(MY_NL, self).__init__()
    self.linear = torch.nn.Linear(in_features=in_d, out_features=out_d)
    for param in self.linear.parameters():
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
      else: param.data.fill_(0)
  
  def forward(self, inputs):
    sigmoid = torch.nn.Sigmoid()
    return sigmoid(self.linear.forward(inputs))


# BASIC NET -------------------------------------------------------------------
#
# Creates Basic 1-hidden-layer neural net. Hidden layer uses relu
# activations, output uses sigmoid activation.

class NET(torch.nn.Module):
  
  def __init__(self, in_d, out_d):
    super(NET, self).__init__()
    self.l1 = torch.nn.Linear(in_features=in_d, out_features=in_d)
    self.l2 = torch.nn.Linear(in_features=in_d, out_features=out_d)
    for param in list(self.l1.parameters()) + list(self.l2.parameters()):
      if len(param.size()) > 1: torch.nn.init.xavier_normal_(param)
      else: param.data.fill_(0)
  
  def forward(self, inputs):
    relu, sigmoid = torch.nn.LeakyReLU(), torch.nn.Sigmoid()
    h = relu(self.l1.forward(inputs))
    return sigmoid(self.l2.forward(h))


# VARIABLE FEEDFORWARD NET ----------------------------------------------------
# 
# Feedforward neural net whose architecture the user can specify, in the form of
# a list (the arch_spec argument). For example: setting arch_spec to [100, 50, 10]
# will create a network with a 100-dimensional input layer, a 50-dimensional
# hidden layer, and a 10-dimensional output layer. 

class SPEC_NET(torch.nn.Module):
  
  def __init__(self, arch_spec):
    super(SPEC_NET, self).__init__()
    self.params = torch.nn.ParameterList()
    for i in range(len(arch_spec)-1):
      self.params.append(torch.nn.Parameter(torch.Tensor(arch_spec[i+1], arch_spec[i]).float()))
      self.params.append(torch.nn.Parameter(torch.zeros(arch_spec[i+1]).float()))
    for param in [self.params[i] for i in range(len(self.params)) if (i%2 == 0)]:
      torch.nn.init.xavier_normal_(param)

  def forward(self, sentence):
    logistic, relu = torch.nn.Sigmoid(), torch.nn.ReLU()
    for i in range(int((len(self.params)/2))-1): 
      sentence = relu(torch.matmul(self.params[2*i], sentence) + self.params[2*i+1])
    return logistic(torch.matmul(self.params[-2], sentence) + self.params[-1])


# UTILITIES -------------------------------------------------------------------

# list of arrays --> list of tensors
def to_tensor(inputs, x_or_y="X"):
  
  tor_inputs = []
  for element in inputs: 
    if x_or_y == "X": tor_inputs.append(torch.from_numpy(element).float())
    else: tor_inputs.append(torch.from_numpy(element).long())
  
  return tor_inputs

# set up batch inputs as packed sequence
def batch_setup(inputs, lengths, pack=True, pad_val=0, x_or_y="X"):
  
  inputs = to_tensor(inputs, x_or_y)
  lengths = torch.from_numpy(lengths).float()

  inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_val)
  lengths, indices = torch.sort(lengths)
  lengths, indices = lengths.flip(0).tolist(), indices.flip(0)
  inputs = inputs[indices]
  
  if pack:
    inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
    return inputs, indices
  else:
    return inputs, indices, lengths

# accuracy computation helper
def acc_helper(outputs, targets):
  
   outputs = torch.round(outputs)
   indicators = torch.abs(outputs - targets)
   classfn_acc = len(indicators) - torch.sum(indicators)
   
   return classfn_acc

