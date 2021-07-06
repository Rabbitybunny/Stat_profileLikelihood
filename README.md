# Using Profile Likelihood to Fit a Gaussian from scratch
Profile likelihood is used when estimating the confidence interval in difficult for method like maximum likelihood; the difficulties may come from having complicated pdf or a large number of nuisance paramters. The profile likelihood method estimates one parameter at a time, as if it's probing the "profile" of the paramter. However, as we will see in the following example, the profile from profile likelihood is not quite the profile of the maximum likelihood. The confidence interval from the profile likelihood can be obtain with relative ease using the so call general likelihood ratio (GLR) test.

In this example, the parameters are the &mu; and &sigma; sampling from a gaussian with:<br/>
&ensp;&ensp;&mu; = 4.8 and &sigma; = 0.6.
For this simple test, the profile likelihood isn't quite necessary as the maximum likelihood method can do it just fine.

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    python3 profileLikelihoodGaus.py
The code outputs the following image:

<img src="https://github.com/Rabbitybunny/Stat_profileLikelihood/blob/main/gausProfileNoNoise.png" width="630" height="490">

- Top-left: blue distribution is the sample drawn from the red Gaussian curve. The top-left coner gives the true values (True Val) and the confidence intervals using point estimate (Pt Est). The point estimate confident interval for &sigma; is shown to be not availabe; it's possible to do, but very difficult.
- Bottom-right: the optimization trace of the maximum likelihood method. The initial point is label "init" in blue, and the optimal point is labeled green. The green error bars are the corresponding confidence intervals (Max Like) are provided on the top-left coner. These confidence intervals are obtained using the inverse of the Fisher information.
- Top-right: profile log-likelihood for &mu; is given in purple. The corresponding interval using the GLR-test is also prodived by the purple dashed lines. In this case, $sigma; is treated as the nuisance parameter. The green dash line is the actual profile of the maximum log-likelihood where the trace traverses on in the bottom-right plot. We can see that in this case the log-likelihood functions don't quite match because for the profile likelihood, &sigma; is optimized independently at every $mu;.
- Bottom-left: similar to the top-right plot, this plot shows the profile log-likelihood for &sigma;. The axis is flip for the ease to compare with the bottom-right plot. Notice the log-likelihood functions essentially overlap with each other for the profile and maximum likelihood.



<img src="https://github.com/Rabbitybunny/Stat_profileLikelihood/blob/main/gausProfileUniNoise.png" width="630" height="490">



References:
- G. J. Feldman and R. D. Cousins, Phys. Rev. D 57, 3873 (1998) (<a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873">Phy Rev D</a>, <a href="https://arxiv.org/abs/physics/9711021">arxiv</a>)
- B. Cousins, Virvual Talk, Univ. of California, Los Angeles (2011) (<a href="http://www.physics.ucla.edu/~cousins/stats/cousins_bounded_gaussian_virtual_talk_12sep2011.pdf">PPT</a>)

