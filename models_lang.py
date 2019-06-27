# DA LANGUAGE MODELS ----------------------------------------------------------
# 
# Models for zero-shot domain adaptation experiments on language modeling task. 


# imports
import numpy as np
import torch
from SST_models import get_sentence_vectors
from train_utils import write
from models import MY_L, MY_NL, MY_LSTM, to_tensor, batch_setup, acc_helper
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# DA_BASELINE -----------------------------------------------------------------
# 
# Baseline model for Zero-Shot DA for language modeling.
# Consists of a sentence encoder (LSTM) and a classifier layer.
# LSTM allows specifying number of layers, dropout, and bidirectionality.

class DA_B_lang(torch.nn.Module):
  
  # initialization method
  def __init__(self, in_d, h_d, layers, dropout, bi,
               vocab_size, vectors, output_vectors):
    super(DA_B_lang, self).__init__()
    
    # Baseline modules
    out_vects = 2 if bi else 1
    self.vectors = vectors
    self.lstm = MY_LSTM(in_d, h_d, layers, dropout, bi, lang=True)
    self.map_output = MY_L(h_d, 100)
    self.output = torch.from_numpy(output_vectors).float()
    self.output.requires_grad = False
    self.output = torch.t(self.output)
    self.output = self.output.to(device)
  
  # forward propagation
  def forward(self, input_inds, targets, lengths):
    
    # set up batch in pytorch
    b_size = len(input_inds)
    inputs = get_sentence_vectors(input_inds, self.vectors)
    inputs, indices = batch_setup(inputs, lengths)
    targets, _, _ = batch_setup(targets, lengths, pack=False, pad_val=-1, x_or_y="Y")
    
    # data --> GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # forward computation (LSTM output)
    inputs_proc = self.lstm.forward(inputs, b_size)

    # get processed inputs in correct form for output layer - (B x S) x 200
    inputs_proc, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs_proc, batch_first=True)
    inputs_proc = inputs_proc.contiguous()
    inputs_proc = inputs_proc.view(-1, inputs_proc.shape[2])

    # get targets in correct form for output layer - (B x S)
    targets = targets.contiguous()
    targets = targets.view(-1)

    # get final outputs and return
    # return self.output.forward(inputs_proc), targets
    to_outputs = self.map_output.forward(inputs_proc)
    return torch.matmul(to_outputs, self.output), targets
  
  # cross-entropy/perplexity computation
  def accuracy(self, inputs, targets, lengths, batch_size, loss):
    
    self.eval()
    crossent_loss = 0
    perplexity = 0

    # total number of words in data (for avg loss/perplexity)
    tot_words = np.sum(lengths)
    
    # switch off gradients, get accuracy, data if on
    with torch.no_grad():

      if len(inputs) % batch_size == 0: iterations = len(inputs)//batch_size
      else: iterations = len(inputs)//batch_size + 1
      for batch in range(iterations):
        
        # get batch
        b_inputs = inputs[batch*batch_size : (batch+1)*batch_size]
        b_targets = targets[batch*batch_size : (batch+1)*batch_size]
        b_lengths = lengths[batch*batch_size : (batch+1)*batch_size]
        
        # forward pass, compute loss
        b_outputs, b_targets = self.forward(b_inputs, b_targets, b_lengths)
        crossent_loss += loss(b_outputs, b_targets).item()
      
    crossent_loss = crossent_loss/tot_words
    perplexity = np.exp(crossent_loss)
    self.train()
    
    return crossent_loss, perplexity


# DA_TRANS --------------------------------------------------------------------
# 
# Embedding-transformation model for Zero-Shot DA for language modeling.
# Consists of sentence encoder, transformation, attn-transformation, classifier,
# and domain encoder (if we're using sequential domain knowledge).
# number of layers, dropout, and bidirectionality can be specified
# for sentence and domain encoders.

class DA_TRANS_lang(torch.nn.Module):

  # initialization method
  def __init__(self, in_d, attn_width, dom_seq, 
               vocab_size, out_style, vectors, output_vectors,
               s_h_d, s_layers, s_dropout, s_bi,
               d_h_d, d_layers, d_dropout, d_bi):
    super(DA_TRANS_lang, self).__init__()
    
    # If domain information is a single vector
    self.vectors = vectors
    s_out_vects = 2 if s_bi else 1
    self.sent_lstm = MY_LSTM(in_d, s_h_d, s_layers, s_dropout, s_bi, lang=True)
    self.dom_lstm = None
    trans_in = in_d
    
    # If domain information is a sequence of vectors
    if dom_seq:
      d_out_vects = 2 if d_bi else 1
      self.dom_lstm = MY_LSTM(in_d, d_h_d, d_layers, d_dropout, d_bi)
      trans_in =  d_h_d * d_out_vects
    
    # some fixed variables
    self.out_style = out_style
    self.in_d = in_d
    self.out_d = 100
    self.map_d = s_h_d*s_out_vects
    self.attn_width = attn_width

    # Transformation, attention-transformation, output layer
    self.trans = MY_L(trans_in, in_d*in_d)
    self.attn_trans = MY_L(trans_in, attn_width*(in_d+2) + 1)
    self.map_output = MY_L(self.map_d, self.out_d)
    self.output = torch.from_numpy(output_vectors).float()
    self.output.requires_grad = False
    self.output = torch.t(self.output)
    self.output = self.output.to(device)

    # output conditioning
    if out_style == "attn": 
      self.attn_out = MY_L(trans_in, vocab_size)
    elif out_style == "ta":
      self.trans_out = MY_L(trans_in, self.out_d*self.out_d)
      self.attn_trans_out = MY_L(trans_in, attn_width*(self.out_d+2) + 1)
  
  # forward propagation
  def forward(self, input_inds, targets, lengths,
              doms, dom_lengths, get_data=False):

    # create any required modules
    drop = torch.nn.Dropout(p=0.5)
    
    # set up batch
    b_size = len(input_inds)
    inputs = get_sentence_vectors(input_inds, self.vectors)
    inputs, indices, in_lengths = batch_setup(inputs, lengths, pack=False)
    targets, _, _ = batch_setup(targets, lengths, pack=False, pad_val=-1, x_or_y="Y")
    
    # prep domain knowledge
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
    
    # apply attention, pack sequence
    weighted_inputs = (inputs * attn_weights) + (trans_inputs * (1 - attn_weights))
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(weighted_inputs, in_lengths, batch_first=True)
    
    # process inputs, get the processed inputs back in (B x S) x 200 form
    inputs_proc_init = self.sent_lstm.forward(packed_inputs, b_size)
    inputs_proc_init, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs_proc_init, batch_first=True)
    
    inputs_proc = torch.zeros(inputs_proc_init.size()[0], inputs_proc_init.size()[1], self.out_d).float()
    inputs_proc = inputs_proc.to(device)
    for i in range(len(inputs_proc_init)):
      inputs_proc[i] = self.map_output.forward(inputs_proc_init[i])

    # if we're using domain information to apply attention over final word scores
    if self.out_style == "attn":

      # get attention vectors for each domain
      attn_out = self.attn_out.forward(doms)
      if self.dom_lstm != None: attn_out_vects = attn_out[dom_ind_rev][indices]
      else: attn_out_vects = attn_out
      
      # get attention in appropriate form
      full_attn = torch.zeros(inputs_proc.size()).float()
      for i in range(len(full_attn)):
        full_attn[i] = attn_out_vects[i].repeat(len(inputs_proc[i]), 1)
      
      # get processed inputs, attention, and targets in correct form
      inputs_proc = inputs_proc.contiguous()
      inputs_proc = inputs_proc.view(-1, inputs_proc.shape[2])
      full_attn = full_attn.view(-1, full_attn.shape[2])
      targets = targets.contiguous()
      targets = targets.view(-1)

      # compute outputs, apply attention to them
      outputs = torch.matmul(inputs_proc, self.output)
      outputs = outputs * full_attn

    # if we're using domain information to apply transformation + attention to LSTM output
    elif self.out_style == "ta":
      
      # get transformation, attention vectors for each domain
      t_out, attn_t_out = self.trans_out.forward(doms), self.attn_trans_out.forward(doms)
      if self.dom_lstm != None: trans_out, attn_trans_out = t_out[dom_ind_rev][indices], attn_t_out[dom_ind_rev][indices]
      else: trans_out, attn_trans_out = t_out, attn_t_out
      
      # apply dropout
      trans_out = drop(trans_out)
      attn_trans_out = drop(attn_trans_out)
      
      # get transformed/domain-specific embeddings, attn weights
      trans_inputs_proc = self.transform_inputs(trans_out, inputs_proc, in_lengths, in_lstm=False)
      attn_weights_proc = self.compute_attn(attn_trans_out, inputs_proc, in_lengths, in_lstm=False)
      
      # apply attention
      inputs_proc_attn = (inputs_proc * attn_weights_proc) + (trans_inputs_proc * (1 - attn_weights_proc))
      
      # get inputs, targets in appropriate form
      inputs_proc_attn = inputs_proc_attn.contiguous()
      inputs_proc_attn = inputs_proc_attn.view(-1, inputs_proc_attn.shape[2])
      targets = targets.contiguous()
      targets = targets.view(-1)

      # compute outputs
      outputs = torch.matmul(inputs_proc_attn, self.output)
    
    # if we're not applying anything to LSTM output/final word scores
    else:

      # get LSTM output in correct form
      inputs_proc = inputs_proc.contiguous()
      inputs_proc = inputs_proc.view(-1, inputs_proc.shape[2])

      # get targets in correct form for output layer - (B x S)
      targets = targets.contiguous()
      targets = targets.view(-1)

      # get final outputs
      outputs = torch.matmul(inputs_proc, self.output)
    
    # save data if on
    if get_data:
      if self.out_style == "attn": 
        return outputs, targets, \
          [t.clone().numpy(), attn_t.clone().numpy(), attn_out.clone().numpy()]
      else:
        return outputs, targets, \
          [t.clone().numpy(), attn_t.clone().numpy(), t_out.clone().numpy(), attn_t_out.clone().numpy()]
    else: 
      return outputs, targets

  # accuracy computation
  def accuracy(self, inputs, targets, lengths, domains,
               doms, dom_lengths, batch_size, loss):
    
    self.eval()
    crossent_loss = 0
    perplexity = 0

    # total number of words in data (for avg loss/perplexity)
    tot_words = np.sum(lengths)
    
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

        # get domain knowledge for batch domains
        b_doms = doms[b_domains]
        if dom_lengths != None: b_dom_lengths = dom_lengths[b_domains]
        else: b_dom_lengths = None
        
        # forward pass, compute loss
        b_outputs, b_targets = self.forward(b_inputs, b_targets, b_lengths,
                                            b_doms, b_dom_lengths)
        crossent_loss += loss(b_outputs, b_targets).item()
      
    crossent_loss = crossent_loss/tot_words
    perplexity = np.exp(crossent_loss)
    self.train()
    
    return crossent_loss, perplexity

  # helper method for getting transformation, applying it
  def transform_inputs(self, transforms, inputs, lengths, in_lstm=True):
    
    # set dimensionality of transformation we're generating
    if in_lstm == True: dim = self.in_d
    else: dim = self.out_d
    
    # set up transformed representations
    trans_inputs = torch.zeros(inputs.size()).float()
    trans_inputs = trans_inputs.to(device)

    # iterate over batch representations, transform them
    for i in range(len(inputs)):
      trans = transforms[i].view(dim, dim)
      trans_inputs[i][:int(lengths[i])] = torch.matmul(inputs[i][:int(lengths[i])], trans)
    
    return trans_inputs
  
  # helper method for getting attn network, applying it
  def compute_attn(self, attn_transforms, inputs, lengths, in_lstm=True):

    # set dimensionality  of attention we're generating
    if in_lstm == True: dim = self.in_d
    else: dim = self.out_d
    
    # first & second layer projections, bias vectors
    attn_trans_m1 = attn_transforms[:,:self.attn_width*dim]
    attn_trans_b1 = attn_transforms[:,self.attn_width*dim : self.attn_width*(dim+1)]
    attn_trans_m2 = attn_transforms[:,self.attn_width*(dim+1) : self.attn_width*(dim+2)]
    attn_trans_b2 = attn_transforms[:,self.attn_width*(dim+2):]

    # set up attention weights for batch representations
    attn_weights = torch.zeros(inputs.size()[0], inputs.size()[1], 1).float()
    attn_weights = attn_weights.to(device)

    # iterate over batch representations, get attention weights for each
    for i in range(len(inputs)):

      # get projections, bias vectors for current domain in proper form
      m1 = attn_trans_m1[i].view(dim, self.attn_width)
      b1 = attn_trans_b1[i].view(1, self.attn_width)
      m2 = attn_trans_m2[i].view(self.attn_width, 1)
      b2 = attn_trans_b2[i].view(1, 1)

      # apply attn network to get weights for current representations
      relu, sigmoid = torch.nn.LeakyReLU(), torch.nn.Sigmoid()
      hiddens = relu(torch.matmul(inputs[i][:int(lengths[i])], m1) + b1)
      attn_weights[i][:int(lengths[i])] = sigmoid(torch.matmul(hiddens, m2) + b2)
    
    return attn_weights

