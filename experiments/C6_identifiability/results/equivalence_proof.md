# Two different equivalence claims

**Representation equivalence.** For any registered source input I, first-order state recurrence is U[t+1]=r U[t]+(1-r)I[t]. Eliminating U gives the same exponential convolution and exactly the same Gaussian X distribution. Applying the same U/Y observation functions and the same do(Y=I) intervention preserves equality. No experiment can distinguish two writings of this same physical model.

**Different causal models, same passive law.** H1: Y=A step(t-t0), tau Udot=Y-U. H2: Y=A(1-exp(-(t-t0)/tau))step(t-t0), U=Y directly. Their continuous effective forces U(t) are identical, so their passive X and noisy V_U distributions are identical. With equal priors the best mechanism-label accuracy from X or (X,V_U) is 50%, irrespective of sample size. V_Y differs. Under do(Y=I), H1 filters I but H2 applies I directly, breaking passive equivalence. The intervention must act on the same physical source definition, with modularity/invariance assumed.

The four-class main library assumes a step source and known candidate equations. Its conditional model selection does not solve unknown-source causal identification. Positive fixed-nuisance KL describes distinct conditional laws; profile-grid distance is a finite-grid diagnostic, not proof of global composite identifiability. Finite-posterior abstention is not structural nonidentifiability.
