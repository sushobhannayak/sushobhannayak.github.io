---
layout: post
comments: true
title:  "On the convergence of SGD for large datasets"
date:   2015-10-23 20:31:00
categories: sgd ml crf
---
>Keep digging. 

In my [last post]({% post_url 2015-10-10-crfpp-sgd %}), we saw, or rather took it on faith, 
that Stochastic Gradient Descent(SGD) is much faster in getting to a reasonable solution compared to Gradient Descent(GD). 
Instead of investigating the theoretical underpinnings, 
I took the easier way out and just pointed the reader to the excellent discussion in [this paper][botou]. 
Also, that post was more focused on the experiments I ran with CRF++, and a detailed theoretical discussion would have been a little out of place.
This is an attempt to rectify that, to go through the derivations myself, to get a better understanding of the concepts in the process.


To recap, the supervised learning problem we are trying to solve is this: We are given a set of $$n$$ i.i.d. data-points $$(x_i, y_i)$$ drawn from an unknown distribution $$P(x,y$$), 
where $$x_i$$ is a vector and $$y_i$$ is a scalar, and we are trying to learn a function $$f$$ such that we can correctly predict $$y=f(x)$$ given a new instance $$x$$. 
How do we determine whether we got a good $$f$$ or not? One way to gauge that is to assign some penalty for wrong predictions: If we have a *cost function* $$l(\hat{y}, y)$$ which
is a measure of how much penalty we pay for mis-predicting $$y$$ as $$\hat{y}=f(x)$$, then one criterion would be to find an $$f$$ that leads to the least expected value
of that penalty. Formally, we want to find an $$f$$ such that the following quantity is minimized:

$$
E(f) = \int l(f(x), y) dP(x,y)
$$

Well, that sounds topping, except that we are bogged down by two limitations:

1. We can't possibly investigate all the functions in the world to come up with the one that minimizes this quantity.
2. We don't know the distribution $$P(x,y)$$; all we have are samples drawn from the distribution.

So we do what we can - we approximate. We create a pool of candidate functions/function-families to investigate (call that set $$F$$), and we approximate the expectation $$E$$ 
with sample average $$E_n$$, called the *empirical risk* so that our objecive takes the following form:

$$
f^*_F = argmin_{f\in F} E_n(f) = argmin_{f\in F} \frac{1}{n}\sum_{i=1}^n l(f(x_i), y_i)
$$

One of the most common ways to generate the family of functions is to take up a family of the form $$\{f_w(x)\}$$, in which each function is parametrized by some
weight vector $$w$$, e.g. $$\{ w^Tx+b \}$$, $$\{(1+\exp(-w^Tx-b))^-1\}$$. Assuming our function belongs to such a family, the problem then reduces to just finding 
the optimum $$w^*$$ that minimizes $$E_n$$, i.e.

$$
w^* = argmin_{w} E_n(f_w) = argmin_{w} \frac{1}{n}\sum_{i=1}^n l(f_w(x_i), y_i)
$$

As we have already seen, gradient descent(GD) starts with an initial estimate $$w_0$$ for $$w^*$$, iterates over the complete data and updates the current estimate on the basis of
the gradient of the loss function:

$$
w_{t+1} = w_t - \gamma \frac{1}{n} \sum_{i=1}^n \bigtriangledown_w l(f_w(x_i), y_i)
$$

where $$\gamma$$ is the learning rate. Stochastic gradient descent (SGD), on the other hand, updates the current estimate based on a random data point from the dataset on each
iteration (instead of going over the whole dataset before making an update like GD) :

$$
w_{t+1} = w_t - \gamma \bigtriangledown_w l(f_w(x_t), y_t)
$$

So, intuitively, GD calculates the proper gradient (with all the data available to it) and descends in that direction whereas SGD only approximates the original gradient 
based on a single data point. So, it stands to reason that GD should lead to a faster convergence, making the best possible movement towards the optimum on each iteration
whereas SGD will be making noisy, sub-optimal movements, hiccupping around, resulting in a gnarly progress. And theory does support this conjecture: Under some regularity
assumptions, when $$w_0$$ is close enough to $$w^*$$, and $$\gamma$$ is reasonable, it takes $$O(\log 1/\rho)$$ iterations to get to an $$E_n$$ that's in a $$\rho$$
neighborhood of $$E^*_n$$ (the $$E_n$$ corresponding to $$w^*$$), i.e. $$E_n < E^*_n+\rho$$. SGD, on
the other hand, takes $$O(1/\rho)$$ iterations to achieve the same. Remember that each iteration of SGD is $$n$$ times faster than that of GD, as GD has to go through all
the datapoints in an iteration while SGD looks at only one. So, while SGD takes $$O(1/\rho)$$ time to reach the optimum,  GD does so in $$O(n\log 1/\rho)$$ time. 

|  | GD | SGD |
| ------------- |:-------------:| :-----:|
| Time per iteration | $$n$$ | $$1$$
| Iterations to accuracy $$\rho$$ | $$\log 1/\rho$$ | $$1/\rho$$
| Time to accuracy $$\rho$$ | $$n\log 1/\rho$$| $$1/\rho$$


To put these
numbers into perspective, if we have a million data points ($$n = 10^6$$), and $$\rho = 10^{-8}$$, GD would reach the optimum ~10 times faster than SGD.
But guess what... SGD almost always performs better with large datasets in 
practice[[1]][botou], reaching to an acceptable solution much faster than GD! So what exactly is happening here? Well, the catch is, in practice, we don't
always need $$w^*$$, the optimal $$w$$ defined beforehand (in fact it might even be a bad idea to use $$w^*$$ as it might not lead to good generalization); 
all we need is a $$w$$ that's in the vicinity of $$w^*$$. And SGD is extremely good at getting to a $$w$$ close to $$w^*$$; alas, once there, 
it bumps around a lot before reaching 
$$w^*$$. GD, on the other hand, takes a longer time than SGD to reach the vicinity of $$w^*$$, but races to the optimum once it gets there. The question now is, can we
 get a theoretically sound explanation for this behavior?

Yes, we can (duh -- why else would I be writing this)! We just have to begin at the beginning. 
Remember what our original goal was - to find an $$f$$ which minimizes $$E$$, the real expectation of loss $$l$$. 
But limited by the inability to explore an unbounded function space with inexhaustible datapoints for endless time, we took recourse to approximations, to minimize $$E_n$$
in lieu of $$E$$, on a parametrized set of functions. Let's go through our approximations, and try to get an estimate of how much error we introduced as we made each
approximation. We started out to find $$f* = argmin_f E(f)$$. Constraining the function space we could explore to $$F$$ meant we could only hope to find 
$$f^*_F = argmin_{f \in F} E(f)$$. But we don't have access to $$E$$ - only an approximation through $$E_n$$. So all we can do is get $$f^*_n = argmin_{f\in F} E_n(f)$$.
Now optimization takes time: So we would be happy to get within a $$\rho$$ neighborhood of $$E^*_n$$, the optimal $$E_n$$, instead of getting to the exact value of
$$E^*_n$$, so that we ultimately arrive at $$\tilde{f_n}$$ such that $$E_n(\tilde{f_n}) < E_n(f^*_n) + \rho$$. 
The error we appropriate through all these approximations, by finding $$\tilde{f_n}$$ instead of $$f^*$$ is (using linearity of expectation): 

$$
\begin{align*}
\epsilon & = \mathbb{E}[E(\tilde{f_n})-E(f^*)] \\
	 & = \mathbb{E}[E(\tilde{f_n})-E(f^*_n)] + \mathbb{E}[E(f^*_n)-E(f^*_F)] + \mathbb{E}[E(f^*_F)-E(f^*)]\\
	 & = \epsilon_{opt} + \epsilon_{est} + \epsilon_{app}
\end{align*}
$$

* $$\epsilon_{app}$$ is the approximation error, sometimes also called the bias. This arises because we can only explore a limited number of function-families belonging
to $$F$$. If we increase the coverage of $$F$$, we will increase the chance of finding a better function to approximate $$f^*$$, leading to a lower error. 
* $$\epsilon_{est}$$ is the estimation error, sometimes also called the variance. This arises because we can't explore an infinite number of samples from the distribution. 
There are two ways we can reduce this error: 
   1. Use simpler function families, which have less expressive capabilities and fewer parameters, and we would be able to get a good
fit even without a lot of data<sup>1</sup>
   2. Get more data, so that $$E_n$$ becomes a better approximation of $$E$$.
* $$\epsilon_{opt}$$ is the optimization error, which can of course be reduced if we run more iterations of the optimization routine. 

Now we know that $$\epsilon_{opt} = O(\rho)$$, since $$E_n(\tilde{f_n}) < E_n(f^*_n) + \rho$$. Assuming the [VC dimension][vcdim] of $$F$$ is bound by $$K$$, and $$n >> K$$,
which is the case when we are dealing with big datasets, we have from learning theory <sup>[2][botou][3][vcdim][4][vcdim2]</sup>:

$$\epsilon_{est} = O\Big(\sqrt{\frac{K(\log(2n/K)+1)}{n}}\Big) = O\Big(\sqrt{\frac{\log n}{n}}\Big)$$

so that:

$$\epsilon = \epsilon_{app} + O\Big(\sqrt{\frac{\log n}{n}}\Big) + O(\rho) \sim \epsilon_{app} + \Big(\frac{\log n}{n}\Big)^\alpha + \rho$$

with $$\alpha \in [0.5, 1]$$, where the final equivalence is done because it gives a more realistic view of the asymptotic behavior as compared to the pessimistic bound
of the equality<sup>[[1]][botou]</sup>. 

Now let's step back for a moment and recap what we are doing. We saw that due to certain limitations, we have to resort to a few approximations, and we have to make do with a 
sub-optimal function $$\tilde{f_n}$$ instead of $$f^*$$, which leads to an error $$\epsilon$$ in our estimation. Our goal is to minimize this, and we can do that in the 
following ways:

   1. Get a bigger $$F$$, explore more functions, reduce $$\epsilon_{app}$$. The catch: takes more time to explore more functions
   2. Process more samples, use a bigger $$n$$, reduce $$\epsilon_{est}$$. The catch: takes more time to operate on more samples
   3. Use a smaller $$\rho$$, reduce $$\epsilon_{opt}$$. The catch: takes more iterations, and consequently time, to get closer to $$E^*$$

In a large scale learning task, time is of the essence, and since all the 3 quantities- $$F, n, \rho$$ - are variables we can manipulate, and since
each one of them has a strong influence on the running time, we are faced with a trade-off involving the three quantities, or by proxy, a trade-off on which component
of the error $$\epsilon$$ we want to reduce by what degree. Now, in the limiting case, we will have $$\epsilon_{app} \sim \epsilon_{est} \sim \epsilon_{opt}$$. Why is that so?
Because the convergence rate of $$\epsilon$$ is limited by the convergence rate of its slowest term, and it won't make sense to spend resources making another term decrease 
faster. For example, if $$\epsilon_{opt}$$ is the one bumming around, and the other two are cruising forward with abandon, then we would be better off making $$\rho$$ smaller 
at the expense of a smaller $$F$$ and $$n$$, to put $$\epsilon_{opt}$$ on steroids and help it catch up to its fervent compatriots. Since, $$\epsilon_{est} \sim \epsilon_{opt}$$,
$$\epsilon_{opt} \sim \rho$$, and 
$$\epsilon_{est} \sim \Big(\frac{\log n}{n}\Big)^\alpha$$, we have $$\rho^{1/\alpha} \sim \frac{\log n}{n}$$, i.e. $$n \sim \rho^{-1/\alpha}\log n$$. Also,
 
$$\rho^{1/\alpha} \sim \frac{\log n}{n} \implies \alpha^{-1}\log\rho \sim (\log\log n - \log n) \sim -\log n \implies \log n \sim \log \rho^{-1/\alpha}$$ 

so that 
$$n \sim \rho^{-1/\alpha}\log \rho^{-1/\alpha}$$. Now, from the table, GD takes 

$$n\log 1/\rho \sim \alpha^{-1}\rho^{-1/\alpha} \log^2 1/\rho \sim \rho^{-1/\alpha} \log^2 1/\rho$$ 

time to reach accuracy $$\rho$$, whereas SGD does it in $$1/\rho$$ time. 
Since in the asymptotic case<sup>2</sup> $$\epsilon \sim \rho$$, GD takes $$ \rho^{-1/\alpha} \log^2 1/\rho$$ time to error $$\epsilon$$,
compared to $$1/\rho$$ for SGD. Substituting our earlier value of $$\rho = 10^{-8}$$ and assuming $$\alpha = 1$$, SGD would reach the optimum ~100 times faster than GD!
So even though SGD, with its languid traipsing to the optimal $$E_n$$, sucks in comparison to GD as an optimizer, it needs far less time to get to a 
pre-defined expected risk $$\epsilon$$, which is what we are usually looking for in practice. Isn't that interesting!


#REFERENCES

	  1. Stochastic Gradient Descent Tricks, Bottou L



<sup>1</sup><sub>Notice that reducing the expressivity of a function space would lead to an increase in the bias or approx. error: the proverbial bias-variance trade-off. </sub>

<sup>2</sup><sub>Because, $$\epsilon_{app} \sim \epsilon_{est} \sim \epsilon_{opt} \sim \rho \sim \epsilon/3 \sim \epsilon$$. Even if we remain conservative and 
say $$\epsilon \sim 3\rho$$, the conclusion still holds. </sub>







[botou]:	    http://research.microsoft.com/pubs/192769/tricks-2012.pdf
[vcdim]: 	    https://en.wikipedia.org/wiki/VC_dimension
[vcdim2]:	    http://www.cs.cmu.edu/~guestrin/Class/10701-S05/slides/pac-vc.pdf