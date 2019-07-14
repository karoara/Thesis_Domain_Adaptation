# Conditioning Language Models for Domain Adaptation

This repository contains the code for my senior thesis on a method for zero-shot domain adaptation. In simpler terms, this means training a model to be able to perform a task well in a domain that it has not trained on, optionally using very little data from the domain in question (if available) to fine-tune the model. All models here are implemented in Pytorch. Read the thesis [here](https://www.dropbox.com/s/qu3k4m6lou60u5b/Thesis_Final_Report.pdf?dl=0).

At a very high level, I've attempted to solve the aforementioned problem in a language context, by training a model to modulate the way it performs a task based on some information about the domain that its input is coming from. From a more theoretical/"pure" AI perspective, this problem matters because humans solve it all the time - we're very good at learning something in a new domain with very little experience, or learning to condition the way we do something in a new setting based on some high-level knowledge about that setting. We'd like to make AI that can do this stuff too. From a more practical perspective, oftentimes little-to-no training data is available in a particular domain - coming up with a good way to modulate a pretrained model with some information about the domain would likely improve performance over training a model from scratch on a very small amount of data, or simply using a pre-trained model that has not been trained to take domain information into account.

## About the Method

### Some Intuition

The way you interpret language when reading a horror film review is different from the way you do so when reading a restaurant review (for the former, the word "terrifying" might be used to indicate a positive experience; in the latter, that seems unlikely). Neural models used for a variety of language tasks might implicitly learn to recognize what domain an input sentence belongs to among those it trains on, and condition the way they treat its language accordingly. However, if we want a model to generalize well to some domain that it has not trained on, it might be tricky for the model to recognize how to treat language from that domain. It may have simply "memorized" how to treat language for the domains it has trained on - it has not explicitly learned how to do so based on the kind of domain it is dealing with. For my thesis, I attempted to address this by training models to modify the way they perform a task based on some information about the domain their input is from. 

## Problem Setting

The method I developed assumes that labeled training data for a desired task is available for a set of domains, and that some high-level information on these domains - and on others that we might want a model to generalize well to, but for which there is no training data - is also available. 

### The Conditioning Mechanism

In order to get models to generalize well to domains they do not train on, I 

### Experiments

## Running the Code

### Dependencies

### Running Experiments
