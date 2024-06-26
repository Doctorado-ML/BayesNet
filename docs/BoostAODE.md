# BoostAODE Algorithm Operation

## Hyperparameters

The hyperparameters defined in the algorithm are:

- ***bisection*** (*boolean*): If set to true allows the algorithm to add *k* models at once (as specified in the algorithm) to the ensemble. Default value: *true*.
- ***bisection_best*** (*boolean*): If set to *true*, the algorithm will take as *priorAccuracy* the best accuracy computed. If set to *false⁺ it will take the last accuracy as *priorAccuracy*. Default value: *false*.

- ***order*** (*{"asc", "desc", "rand"}*): Sets the order (ascending/descending/random) in which dataset variables will be processed to choose the parents of the *SPODEs*. Default value: *"desc"*.

- ***block_update*** (*boolean*): Sets whether the algorithm will update the weights of the models in blocks. If set to false, the algorithm will update the weights of the models one by one. Default value: *false*.

- ***convergence*** (*boolean*): Sets whether the convergence of the result will be used as a termination condition. If this hyperparameter is set to true, the training dataset passed to the model is divided into two sets, one serving as training data and the other as a test set (so the original test partition will become a validation partition in this case). The partition is made by taking the first partition generated by a process of generating a 5 fold partition with stratification using a predetermined seed. The exit condition used in this *convergence* is that the difference between the accuracy obtained by the current model and that obtained by the previous model is greater than *1e-4*; otherwise, one will be added to the number of models that worsen the result (see next hyperparameter). Default value: *true*.

- ***maxTolerance*** (*int*): Sets the maximum number of models that can worsen the result without constituting a termination condition. if ***bisection*** is set to *true*, the value of this hyperparameter will be exponent of base 2 to compute the number of models to insert at once. Default value: *3*

- ***select_features*** (*{"IWSS", "FCBF", "CFS", ""}*): Selects the variable selection method to be used to build initial models for the ensemble that will be included without considering any of the other exit conditions. Once the models of the selected variables are built, the algorithm will update the weights using the ensemble and set the significance of all the models built with the same &alpha;<sub>t</sub>. Default value: *""*.

- ***threshold*** (*double*): Sets the necessary value for the IWSS and FCBF algorithms to function. Accepted values are:
  - IWSS: $threshold \in [0, 0.5]$
  - FCBF: $threshold \in [10^{-7}, 1]$

  Default value is *-1* so every time any of those algorithms are called, the threshold has to be set to the desired value.

- ***predict_voting*** (*boolean*): Sets whether the algorithm will use *model voting* to predict the result. If set to false, the weighted average of the probabilities of each model's prediction will be used. Default value: *false*.

## Operation

### [Base Algorithm](./algorithm.md)
