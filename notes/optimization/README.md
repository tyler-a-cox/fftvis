## Notes on Optimization

One challenging aspect in fitting a sky model to our dataset is the sheer volume of data involved. Even our most reduced dataset in H6C has a data volume of ~200 GB. 

I think the approach that makes the most sense data parallel processing. With this approach, we split data across multiple processing units on lustre (say 10 files worth), compute gradients, and accumulate gradients to a central processing node.

Another approach to may be to run a gradient descent optimization assuming that we don't actually care what the true structure of the beam/sky is and fit for the combined chromaticity of their product

TODO:
- Choose a package for optimization with autograd that supports gradient accumulation
    - Both `pytorch` and `jax/optax` support gradient accumulation
- Support gradient accumulation  