---
layout: post
title: Introduction to RL and coding its building blocks
date: 2021-12-31 00:15:00 +0300
description: What is Reinforcement Learning and why is it useful? You'll find these details in this post.
img: software.jpg # Add image post (optional)
tags: [RLIntro, AlgorithmClass] # add tag
---

Welcome. 

I write posts about reinforcement learning. You may have heard about the term machine learning. Is RL the same? Well, it belong to the same field 'Artificial Intelligence' as machine learning. There are 3 subfields of artificial intelligence:

1. Machine Learning
2. Deep Learning
3. Reinforcement Learning

I assuming you are already have encountered, studied deeply or pioneered first 2 concepts. Because, these two will be very important building blocks as we study reinforcement learning much much deeper!


# What is reinforcement learning?

According to [wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning), reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

Reinforcement learning differs from supervised learning in not needing labelled input/output pairs be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge). Partially supervised RL algorithms can combine the advantages of supervised and RL algorithms.

The environment is typically stated in the form of a Markov decision process (MDP), because many reinforcement learning algorithms for this context use dynamic programming techniques. The main difference between the classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the MDP and they target large MDPs where exact methods become infeasible.

We've encountered very import concept called **Markov Decision Process.**

# What is Markov Decision Process?

Markov Decision Process(MDP) is a mathematical framework that describes sequential decision making process where each action and environment is affected by previous actions and environments. I am quite sure you don't know what this actually means. Let's briefly go over this concept mathematically.

I will introduce variables which will be used in describing MDP process.

* state \(s_t\)
* action \( a_t \)
* reward \( r_t \)

