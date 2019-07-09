# Thesis Code - Conditioning Language Models for Domain Adaptation

This repository contains code for my senior thesis. [The report (on Dropbox)](https://www.dropbox.com/s/qu3k4m6lou60u5b/Thesis_Final_Report.pdf?dl=0)

Abstract: We consider a setting in which a language model - given access to some information about an input’s domain - is trained to learn a task over an entire distribution of domains, with the goal of generalizing to inputs from domains that are not in its training data. Drawing inspiration from existing methods outside our problem setting, we develop a mechanism that conditions an operation in a language model to modify its representation of an input based on information about a domain. This mechanism is meant to be trained jointly with the task- performing model, and makes few assumptions about the model architecture. We perform experiments in which we compare the performance of a model that is augmented with our mechanism to a baseline that is not for language modeling and sentiment analysis tasks. While the conditioning mechanism does not currently provide a performance improvement on real data, experiments with synthetic data suggest that it is capable of doing so, and that some fine-tuning and further experimentation may enable it to work better.

Hi.
