# DA TRAINING UTILITIES/FUNCTIONS ---------------------------------------------
# 
# Methods for training models for Zero-Shot Domain Adaptation for sentiment analysis.


# imports
import sys
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function for training baseline model
def train_DA_B(model,
               tr_inputs, tr_targets, tr_lengths,
               sd_te_inputs, sd_te_targets, sd_te_lengths,
               te_inputs, te_targets, te_lengths,
               epochs, learn_rate, loss, batch_size, 
               report_acc=0, weaken_lr=0):
  
  # optimizer, model --> GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
  model.to(device)
  
  # best validation acc. variable, corresponding training/test acc. variables
  best_sd_te_acc = 0
  corr_tr_acc = 0
  corr_te_acc = 0
  model_dict = None

  # history
  history = []
  
  # iterate over epochs
  for epoch in range(epochs):
    
    # decay learning rate if on
    if weaken_lr != 0 and (epoch+1) % 10 == 0:
      for param_group in optimizer.param_groups: param_group["lr"] *= weaken_lr
    
    # permute datapoints
    perm = np.random.permutation(len(tr_inputs))
    
    # iterate over minibatches
    if len(tr_inputs) % batch_size == 0: iterations = len(tr_inputs)//batch_size
    else: iterations = len(tr_inputs)//batch_size + 1
    for batch in range(iterations):
      
      # get batch, forward pass, backward pass, update
      batch_inputs = tr_inputs[perm[batch*batch_size : (batch+1)*batch_size]]
      batch_targets = tr_targets[perm[batch*batch_size : (batch+1)*batch_size]]
      batch_lengths = tr_lengths[perm[batch*batch_size : (batch+1)*batch_size]]
      
      batch_outputs, batch_targets = model.forward(batch_inputs, batch_targets, batch_lengths)
      
      optimizer.zero_grad()
      curr_loss = loss(batch_outputs, batch_targets)
      curr_loss.backward()
      optimizer.step()
    
    # compute accuracies after epoch, add to history
    tr_acc = model.accuracy(tr_inputs, tr_targets, tr_lengths, batch_size)
    sd_te_acc = model.accuracy(sd_te_inputs, sd_te_targets, sd_te_lengths, batch_size)
    te_acc = model.accuracy(te_inputs, te_targets, te_lengths, batch_size)
    history.append([tr_acc, sd_te_acc, te_acc])
    
    # update best accuracies, model dictionaries
    if sd_te_acc > best_sd_te_acc:
      best_sd_te_acc = sd_te_acc
      corr_tr_acc = tr_acc
      corr_te_acc = te_acc
      model_dict = model.state_dict()
    
    # report accuracies if on
    if report_acc != 0 and ((epoch+1) % report_acc == 0 or epoch == 0):
      write("Training accuracy after epoch " + str(epoch) + ": " + str(tr_acc))
      write("Same-domain test accuracy after epoch " + str(epoch) + ": " + str(sd_te_acc))
      write("Test accuracy after epoch " + str(epoch) + ": " + str(te_acc) + "\n")
  
  return best_sd_te_acc, corr_tr_acc, corr_te_acc, model_dict, history


# function for training transformation-based model
def train_DA_Con(model,
                 doms, dom_lengths,
                 tr_inputs, tr_targets, tr_lengths, tr_domains,
                 sd_te_inputs, sd_te_targets, sd_te_lengths, sd_te_domains,
                 te_inputs, te_targets, te_lengths, te_domains,
                 epochs, learn_rate, loss, batch_size,
                 report_acc=0, weaken_lr=0):
  
  # set up optimizer, model --> GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
  model.to(device)

  # best validation acc. variable, corresponding training/test acc. variables
  best_sd_te_acc = 0
  corr_tr_acc = 0
  corr_te_acc = 0
  model_dict = None

  # history
  history = []
  
  # iterate over epochs
  for epoch in range(epochs):
    
    # decay learning rate if on
    if weaken_lr != 0 and (epoch+1) % 10 == 0:
      for param_group in optimizer.param_groups: param_group["lr"] *= weaken_lr
    
    # permute datapoints
    perm = np.random.permutation(len(tr_inputs))
    
    # iterate over minibatches
    if len(tr_inputs) % batch_size == 0: iterations = len(tr_inputs)//batch_size
    else: iterations = len(tr_inputs)//batch_size + 1
    for batch in range(iterations):

      # get batch
      b_inputs = tr_inputs[perm[batch*batch_size : (batch+1)*batch_size]]
      b_targets = tr_targets[perm[batch*batch_size : (batch+1)*batch_size]]
      b_lengths = tr_lengths[perm[batch*batch_size : (batch+1)*batch_size]]
      b_domains = tr_domains[perm[batch*batch_size : (batch+1)*batch_size]]

      # get domain knowledge for batch domains
      b_doms = doms[b_domains]
      if dom_lengths != None: b_dom_lengths = dom_lengths[b_domains]
      else: b_dom_lengths = None
      
      # forward pass, backward pass, update
      b_outputs, b_targets = model.forward(b_inputs, b_targets, b_lengths,
                                           b_doms, b_dom_lengths)
      
      optimizer.zero_grad()
      curr_loss = loss(b_outputs, b_targets)
      curr_loss.backward()
      optimizer.step()
    
    # compute accuracies after epoch, add to history
    tr_acc = model.accuracy(tr_inputs, tr_targets, tr_lengths, \
      tr_domains, doms, dom_lengths, batch_size)
    sd_te_acc = model.accuracy(sd_te_inputs, sd_te_targets, sd_te_lengths, \
      sd_te_domains, doms, dom_lengths, batch_size)
    te_acc = model.accuracy(te_inputs, te_targets, te_lengths, \
      te_domains, doms, dom_lengths, batch_size)
    history.append([tr_acc, sd_te_acc, te_acc])
    
    # update best accuracies, model dicts
    if sd_te_acc > best_sd_te_acc:
      best_sd_te_acc = sd_te_acc
      corr_tr_acc = tr_acc
      corr_te_acc = te_acc
      model_dict = model.state_dict()
    
    # report accuracies if on
    if report_acc != 0 and ((epoch+1) % report_acc == 0 or epoch == 0):
      write("Training accuracy after epoch " + str(epoch) + ": " + str(tr_acc))
      write("Same-domain test accuracy after epoch " + str(epoch) + ": " + str(sd_te_acc))
      write("Test accuracy after epoch " + str(epoch) + ": " + str(te_acc) + "\n")
  
  return best_sd_te_acc, corr_tr_acc, corr_te_acc, model_dict, history


# UTILITIES -------------------------------------------------------------------

# write output immediately
def write(string):
  print(string)
  sys.stdout.flush()

# write history
def write_history(history):
  for epoch in history:
    epoch_accs = str(round(epoch[0], 4)) + " " + str(round(epoch[1], 4)) + " " + str(round(epoch[2], 4))
    write(epoch_accs)
  write("")

