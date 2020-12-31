# exomoon-in-binaries
This is a repository containing processed data and python scripts for the article "Exomoons in Systems with a Strong Perturber: Applications to \alpha Cen AB".  The paper investigates the conditions for orbital and tidal stability of a retrograde orbiting exomoon and updates previous results from [Domingos, Winter, & Yokoyama (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.373.1227D/abstract).  Additionally, we identify the constraints form orbital and tidal stability for an exomoon where the host planet orbits within the habitable zone of one component of the binary star system $\alpha$ Cen AB.   The possible transit timing variations (TTVs) of exomoons are affected by the perturbations from the companion star and can provide a more specific regime for which exomoons could be dectected in the future.

The python scripts use packages from numpy, scipy (>= 1.4), & matplotlib.  The N-body code [Rebound](https://rebound.readthedocs.io/en/latest/) is used for most of the numerical integrations, while [Posidonius](https://www.blancocuaresma.com/s/posidonius) evaluates a tidal model from the instantaneous torques (see [Bolmont et al. 2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...583A.116B/abstract)).  See the python-scripts folder for the scripts used to produce figures in the paper and example codes to reproduce our results.  The python scripts assume a directory tree produced from "git clone \*.git" to find/use the "data" and "Figs" subfolders.

Attribution
--------
A more detailed description of these simulations, methods, and the context for the future observations will be available in the following paper.  Please use the following citation, if you find these tools useful in your research. 

```
@article{Quarles2021,
author = {{Quarles}, B. and {Eggl}, S. and {Rosario-Franco}, M. and {Li}, G.},
title = "{Exomoons in Systems with a Strong Perturber: Applications to $\alpha$ Cen AB}",
journal = {\mnras},
year = 2021,
note = "submitted"
}
```
