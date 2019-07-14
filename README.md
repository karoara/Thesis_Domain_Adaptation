# Conditioning Language Models for Domain Adaptation

This repository contains the code for my senior thesis on a method for training language models to perform zero-shot domain adaptation. All models are implemented in Pytorch. Read the thesis [here](https://www.dropbox.com/s/qu3k4m6lou60u5b/Thesis_Final_Report.pdf?dl=0).

## About the Method

### What is "zero-shot domain adaptation"? Why does it matter?

In simple terms, "zero-shot domain adaptation" here means training a model to be able to perform a task well in domains that it has not trained on - to adapt well to such domains with "zero" training data. At a high level, I've attempted to solve this problem in a language context, by coming up with a mechanism that learns to modulate a language model's parameters based on some information about the domain that the model's input is coming from.

To motivate the problem from a practical perspective, consider a setting in which we want to perform a task in a certain domain, but have no training data for that domain. Perhaps we have some labeled training data for the task from other domains. We might try training a model on these domains and then transfer it to the domains we care about.

The model ends up learning to look for a set of features in its input for the domains it trains on. Assuming that the domains not present in the training data are different from the ones that are, a couple undesirable things might be happening:
- The model learns "average" features across inputs from the training domains. In this case, it doesn't even learn to use domain specific features for the training domains; it certainly isn't capable of figuring out what kind of domain-specific features need to be used for inputs from non-training domains. 
- The model might memorize domain-specific features for the training domains. In this case, it still may not have learned to figure out what *kind* of domain it's input is from and use the appropriate features, preventing it from doing so for inputs from non-training domains.

Instead of simply pre-training a model on a set of domains and hoping it transfers well to other ones, we might explicitly provide it with some information about the domain that its input is from, both at training and test time. If the information accurately represents domain similarity, the model should learn to modify the features it uses based on the kind of domain its input is from. In a nutshell, coming up with good ways to condition a model on domain information would be useful because they would likely lead to better domain transfer in the scenarios where little-to-no target domain data is available. 

The problem has significance from a "pure AI" perspective too, as humans solve it all the time. We're very good at learning something in a new domain with little-to-no experience, or conditioning the way we do something in a new setting based on some high-level knowledge about that setting. We'd like to make AI that can do this stuff too. 

### Conditioning Models

There are multiple ways, each with different structures and biases, to have a model take domain information into account. The simplest option might be to concatenate an information vector to the model's input. However, more structured methods exist, as discussed in Dumoulin et al.'s paper on [Feature-Wise Transformations](https://distill.pub/2018/feature-wise-transformations/). They discuss a general-purpose method for conditioning a model's computation on some domain or task information, called "FiLM", or Feature-Wise Linear Modulation. In FiLM, a transformation is trained to take an information vector, and output the parameters of an affine layer that acts on a representation in a model. The affine layer essentially consists of gating and bias vectors - in the domain adaptation context, they switch nodes and feature maps in the model on and off for the given domain. 



### The Conditioning Mechanism

As stated, for my thesis, I've considered the zero-shot adaptation problem in the language context. A simple way in which the same language varies in its meaning/interpretation across domains is through changes in word meaning. While reading a horror film review, the word "terrifying" might be used to indicate a positive experience; in a restaurant review, this seems unlikely. Words like "LGBT" and "gun" are used with different connotations depending on whether they appear on a left-leaning or right-leaning online community. 

To account for this kind of change from domain to domain that we see in language, we might train a "mechanism" that learns to accept some domain information as input, and produce an operation that modifies the words in an input sentence properly for the corresponding domain. The mechanism is parametrized by two transformations, each of which take domain information as input. These transformations learn to produce a transformation and a single-layered attention network, which work in the following way:

**1.** The produced transformation takes the generic embeddings in the input sentence and transforms them for the current domain:
![Mechanism Transformation](images/Mech1.png)

**2.** The attention network outputs a set of attentional weights describing the extent to which the transformed embedding - as opposed to the initial, generic embedding - should be used to represent each word:
![Attention Network](images/Mech2.png)

**3.** Using the attentional weights, a weighted linear combination between the transformed and generic embeddings is taken to create new embeddings that represent the input sentence:
![Weighted Combination](images/Mech3.png)

Once the new set of embeddings has been obtained, they are input to the language model that performs the task. A straightforward way to think about this is that the transformation and attention network the mechanism produces are part of the language model's embedding layer - we are thus using domain information to condition the embedding layer of a language model. The two transformations that parametrize the mechanism are trained in tandem with the language model on data and information from a set of training domains.

To illustrate what this is meant to do in practice, consider the following scenario: we want to train a model to perform binary sentiment classification on comments from online communities. We might train a model to perform the task on comments from a subset of communities, and train a mechanism simultaneously to produce transformations/attn networks that modify a comment's embeddings for the online community that it is from. During training, the mechanism might learn to recognize indicators of political orientation from the domain information it is given, and produce transformations/attn networks that modify the embeddings in comments accordingly. When given domain information for a new domain, it might recognize that the domain is left-leaning, and: 
- Produce a transformation that maps "LGBT" to a latent positive meaning, and "gun" to a latent negative meaning.
- Produce an attention network that chooses to represent the words "LGBT" and "gun" with the transformed embeddings, but chooses to represent words like "hello" or "goodbye" with generic ones (as their meaning doesn't really change with political orientation). 

### Experiments

## Running the Code

### Dependencies

### Running Experiments
