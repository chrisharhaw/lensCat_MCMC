# Project Motivation

Strong gravitational lensing provides a powerful tool to investigate cosmology as it is a probe that is completely independent of the cosmic distance ladder. Of particular importance are combinations of small numbers of lensed quasar systems which can be used to infer the Hubble constant from the measured time-delay between lensed images (see the H0LiCOW collaboration and TDcosmo for the most advanced investigations). However, only a small fraction of the total number of observed lensing systems are available for this kind of analysis. Instead, it is possible to make use of a much larger fraction of the catalogue of known strong gravitational lens systems for cosmological inference using the 'distance ratio' technique (namely using the relation between the ratio of angular diameter distances in a system, Einstein radiusm, and velocity dispersion, for a given lens model).

When we model strong gravitational lenses in this way we typically focus on systems which are somewhat isolated in the sky, such that there are no other nearby luminous deflectors along the line of sight that contribute to the deflection. However, in reality this isn't always entirely possible and there will be additional mass along the line of sight that contributes to the deflection, which we call external convergence. In time-delay analyses of individual strong lens systems, the effect of external convergence and shear can bias the fit for the Hubble constant by a few percent. However, I am unsure of the scale of the effect in large statistical studies. So the aim of this project will be to investigate this effect, determine how much it can bias the inference of cosmological parameters and in what way it does this. 

## Background Material

In order to test the effects of external convergence on estimates of cosmological parameters, I think you will need to know the basics of:
- the standard strong gravitional lensing formalism
- distances in the standard $\Lambda$ CDM cosmological model (particularly the angular diameter distance)
- MCMC sampling and some basic Baysian statistics 

I would reccomend the following reading to start you of:
- **Gravitational Lensing: Strong, Weak, and Micro** by Schneider, Kochanek, and Wambsganss
- **Meneghetti's Gravitational Lensing Lecture series** (https://ita.uni-heidelberg.de/~jmerten/misc/meneghetti_lensing.pdf or  http://pico.oabo.inaf.it/~massimo/teaching.html)
- **Distance measures in cosmology** https://arxiv.org/pdf/astro-ph/9905116 or equivalently any decent cosmology textbook.
- **Statistical methods in cosmology** https://arxiv.org/pdf/0911.3105

Once you feel like you have a decent grip of the main concepts of lensing (i.e., the lensing equation, the lensing potential, einstein radius, critical and caustic curves, and have had a look at some simple lens model examples e.g. the singular isothermal sphere) then I would reccomend looking at the paper by Chen et al. 2019 - https://doi.org/10.1093/mnras/stz1902 - which uses the distance ratio methodology to compare different lens model approaches to cosmological parameter estimation. It also has a large catalogue of lenses and their parameters in a table in the Appendix. This was at one point the largest catalogue compiled, it may not be anymore though after recent survey results so this is worth checking. 

