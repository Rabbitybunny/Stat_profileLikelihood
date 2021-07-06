# Using Profile Likelihood to Fit a Gaussian from scratch
Profile likelihood is used when estimating the confidence interval in difficult for method like maximum likelihood; the difficulties may come from having complicated pdf or a large number of nuisance paramters. The profile likelihood method estimates one parameter at a time, as if it's probing the "profile" of the paramter. However, as we will see in the following example, the profile from profile likelihood is not quite the profile of the maximum likelihood.

In this example, the parameters are the &mu; and &sigma; sampling from a gaussian with:<br/>
&ensp;&ensp;&mu; = 4.8 and &sigma; = 0.6.

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    python3 profileLikelihoodGaus.py
The code outputs the following image:

<img src="https://github.com/Rabbitybunny/Stat_profileLikelihood/blob/main/gausProfileNoNoise.png" width="630" height="490">

- Top-left: blue distribution are the sample drawn from the red Gaussian curve. The top-left coner gives the true values (True Val) and the confidence interval using point estimate (Pt Est). The point estimate confident interval for &sigma; is shown to be not availabe; it's possible to do, but very difficult.
- Bottom-right: the maximum likelihood

and maximum likelihood method (Max Like)
- Top-right: 
- Bottom-left:
<img src="https://github.com/Rabbitybunny/Stat_profileLikelihood/blob/main/gausProfileUniNoise.png" width="630" height="490">



References:
- G. J. Feldman and R. D. Cousins, Phys. Rev. D 57, 3873 (1998) (<a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873">Phy Rev D</a>, <a href="https://arxiv.org/abs/physics/9711021">arxiv</a>)
- B. Cousins, Virvual Talk, Univ. of California, Los Angeles (2011) (<a href="http://www.physics.ucla.edu/~cousins/stats/cousins_bounded_gaussian_virtual_talk_12sep2011.pdf">PPT</a>)

