# Adaptive Search-based Repair of Deep Neural Networks

Deep Neural Networks (DNNs) are finding a place at the heart of more and more critical systems, and it is necessary to ensure they perform in as correct a way as possible. Search-based repair methods, that search for new values for target neuron weights in the network to better process fault-inducing inputs, have shown promising results. These methods rely on fault localisation to determine what weights the search should target. However, as the search progresses and the network evolves, the weights responsible for the faults in the system will change, and the search will lose in effectiveness. In this work, we propose an adaptive search method for DNN repair that adaptively updates the target weights during the search by performing fault localisation on the current state of the model. We propose and implement two methods to decide when to update the target weights, based on the progress of the search's fitness value or on the evolution of fault localisation results. We apply our technique to two image classification DNN architectures against a dataset of autonomous driving images, and compare it with a state-of-the art search-based DNN repair approach.
