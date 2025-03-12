# Optimization Experiments

This repo is a small set of experiments relating to optimization where you have an existing model, that you cannot update for whatever reason, but you would still like to optimize the weights between features.

The way I've tried to accomplish this across the optim[X].py files is by using a neural network as an optimization framework for our upstream model that we cannot directly optimize or mutate.  

