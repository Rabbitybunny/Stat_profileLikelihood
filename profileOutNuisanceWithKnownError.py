import sys, os, math, time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import optimize
from scipy import stats
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore");

TOLERANCE = pow(10.0, -10);
SNUMBER   = pow(10, -124);

def gaussian(mu, sig, x):
    X = np.array(x);
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    vals[vals < SNUMBER] = SNUMBER;
    return vals;
def logGaus(mu, sig, x, trace=None):
    if trace != None:
        trace.append([mu, sig]);
    X = np.array(x);
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    vals = np.log(vals);
    LL = sum(vals);
    return LL;
def negLogLikelihood(x, trace=None):
    return lambda par : -1.0*logGaus(par[0], par[1], x, trace);
def negLogLikeMu(mu, x):
    return lambda sig : -1.0*logGaus(mu, sig, x);
def negLogLikeSig(sig, x):
    return lambda mu : -1.0*logGaus(mu, sig, x);
def logGausUniBK(mu, sig, noiseR, x,\
                 noisePar=[0.0, 1.0], noiseRange=[0.0, 1.0], trace=None):
    if trace != None:
        trace.append([mu, sig]);
    X = np.array(x);
    gaus = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)));
    uni = np.ones(len(x))*1.0/(noiseRange[1] - noiseRange[0]);
    vals = np.log((1.0 - noiseR)*gaus + noiseR*uni);
    #NOTE: to profile out the nuisance parameter with know error;
    #likelihood function L = L_main(mu, sig, noiseR)*L_noise(noiseR, noisePar)
    noiseGaus = math.exp(-pow(noiseR-noisePar[0],2.0)/(2.0*pow(noisePar[1],2.0)))\
              *(1.0/(noisePar[1]*math.sqrt(2.0*math.pi)));
    LL = sum(vals) + len(x)*math.log(noiseGaus);
    return LL;
def negLogLikelihoodUniBK(x,   noisePar=[0.0, 1.0], noiseRange=[0.0, 1.0], trace=None):
    return lambda par : -1.0*logGausUniBK(par[0], par[1], par[2], x,\
                                          noisePar, noiseRange, trace);
def negLogLikeMuUniBK(mu, x,   noisePar=[0.0, 1.0], noiseRange=[0.0, 1.0]):
    return lambda par : -1.0*logGausUniBK(mu, par[0], par[1], x,\
                                          noisePar, noiseRange);
def negLogLikeSigUniBK(sig, x, noisePar=[0.0, 1.0], noiseRange=[0.0, 1.0]):
    return lambda par : -1.0*logGausUniBK(par[0], sig, par[1], x,\
                                          noisePar, noiseRange);

    
def main():
    verbosity = 1;
    binN = 200;
    rangeX = [0.0, 10.0];

    np.random.seed(0);
    dataMu  = 4.8;
    dataSig = 0.6;
    dataN   = 100;
    #assume the uniform background has a range of rangeX
    #with ratio to the data being noiseR \pm noiseRerr
    noiseR    = 0.3; 
    noiseRerr = 0.05;
    
    alpha           = 0.95;        #significance
    muRange         = [4.0, 6.0];  #search range
    sigRange        = [0.2, 2.0];
    profileStepSize = 0.001;
#data
    noiseR = np.random.normal(noiseR, noiseRerr);
    noisePDF = np.random.uniform(*rangeX, int(noiseR*dataN));
    dataPDF  = np.random.normal(dataMu, dataSig, dataN);
    dataPDF = np.concatenate((dataPDF, noisePDF));
    np.random.shuffle(dataPDF);
    nbins = np.linspace(*rangeX, binN+1)[:-1]
    dataHist = np.histogram(dataPDF[:dataN], bins=binN, range=rangeX)[0]
#point estimate
    valMu  = np.average(dataPDF);
    errMu  = np.std(dataPDF)/np.sqrt(dataN);
    valSig = np.sqrt(np.var(dataPDF));
    errSig = -1;
#maximum likelihood
    if verbosity >= 1:
        print("Processing maximum likelihood...");
    #optInitVals = [valMu, valSig, noiseR];
    optInitVals = [4.5, 1.5, noiseR];
    valTrace = [];
    negMaxLL = negLogLikelihoodUniBK(dataPDF, noisePar=[noiseR, noiseRerr], 
                                     noiseRange=rangeX, trace=valTrace);
    optResult = optimize.minimize(negMaxLL, optInitVals, method="Nelder-Mead");
    maxLikeMu, maxLikeSig, maxLikeNoiseR = optResult.x;
#max like standard error using sqrt 1/(Fisher information)
    maxErrMu = maxLikeSig*np.sqrt(1.0/dataN);
    maxErrSig = maxLikeSig*np.sqrt(1.0/(2.0*dataN));
#profile likelihood
    if verbosity >= 1:
        print("Likelihood profiling for mu...");
    muProfile = [];
    muProfileLikelihood = [];
    muMaxLikelihoodProfile = [];
    muOpt = [0, 0.0, 0.0, -pow(10, 24), 0];
    muRangeN = int((muRange[1]-muRange[0])/profileStepSize);
    for i in (tqdm(range(muRangeN)) if verbosity>=1 else range(muRangeN)):
        mu = muRange[0] + i*profileStepSize;
        negLL = negLogLikeMuUniBK(mu, dataPDF,\
                                  noisePar=[noiseR, noiseRerr], noiseRange=rangeX);
        #optResult = optimize.minimize_scalar(negLL, tol=TOLERANCE,\
        #                                     method="bounded",\
        #                                     bounds=(sigRange[0], sigRange[1]));
        #sig = 1.0*optResult.x;
        optResult = optimize.minimize(negLL, [optInitVals[1], optInitVals[2]], \
                                      method="Nelder-Mead");
        sig, optR = 1.0*optResult.x;
        optLikelihood = -1.0*negLL([sig, optR]);
        muProfile.append(mu);
        if optLikelihood > muOpt[3]:
            muOpt = [i, mu, sig, optLikelihood, optR];
        muProfileLikelihood.append(optLikelihood);
    for i in range(muRangeN):
        mu = muRange[0] + i*profileStepSize;
        negLL = negLogLikeMuUniBK(mu, dataPDF,\
                                  noisePar=[noiseR, noiseRerr], noiseRange=rangeX);
        optLikelihood = -1.0*negLL([maxLikeSig, maxLikeNoiseR]);
        muMaxLikelihoodProfile.append(optLikelihood);

    if verbosity >= 1:
        print("Likelihood profiling for sigma...");
    sigProfile = [];
    sigProfileLikelihood = [];
    sigMaxLikelihoodProfile = [];
    sigOpt = [0, 0.0, 0.0, -pow(10, 24)];
    sigRangeN = int((sigRange[1]-sigRange[0])/profileStepSize);
    for i in (tqdm(range(sigRangeN)) if verbosity>=1 else range(sigRangeN)):
        sig = sigRange[0] + i*profileStepSize;
        negLL = negLogLikeSigUniBK(sig, dataPDF,\
                                   noisePar=[noiseR, noiseRerr], noiseRange=rangeX);
        #optResult = optimize.minimize_scalar(negLL, tol=TOLERANCE,\
        #                                     method="bounded",\
        #                                     bounds=(muRange[0], muRange[1]));
        #mu = 1.0*optResult.x;
        optResult = optimize.minimize(negLL, [optInitVals[0], optInitVals[2]], \
                                      method="Nelder-Mead");
        mu, optR = 1.0*optResult.x;
        optLikelihood = -1.0*negLL([mu, optR]);
        sigProfile.append(sig);
        if optLikelihood > sigOpt[3]:
            sigOpt = [i, mu, sig, optLikelihood, optR];
        sigProfileLikelihood.append(optLikelihood);
    for i in range(sigRangeN):
        sig = sigRange[0] + i*profileStepSize;
        negLL = negLogLikeSigUniBK(sig, dataPDF,\
                                   noisePar=[noiseR, noiseRerr], noiseRange=rangeX);
        optLikelihood = -1.0*negLL([maxLikeMu, maxLikeNoiseR]);
        sigMaxLikelihoodProfile.append(optLikelihood);
#alpha significance from likelihood ratio test for profile likelihood
    chi2ppf = stats.chi2.ppf(alpha, 1);
    muConfInt = [0.0, 0.0];
    stepIdx = 0;
    logLikelihood = muOpt[3];
    #the likelihood ratio is a subtraction in log likelihood
    while 2*abs(logLikelihood - muOpt[3]) < chi2ppf:
        stepIdx += 1;
        if muOpt[0] + stepIdx > muRangeN:
            print("WARNING: Please increase the search range for mu.");
            break;
        logLikelihood = muProfileLikelihood[muOpt[0] + stepIdx];
    muConfInt[0] = abs(muProfile[muOpt[0] + stepIdx] - muOpt[1]);
    stepIdx = 0;
    logLikelihood = muOpt[3];
    while 2*abs(logLikelihood - muOpt[3]) < chi2ppf:
        stepIdx += 1;
        if muOpt[0] - stepIdx < 0:
            print("WARNING: Please increase the search range for mu.");
            break;
        logLikelihood = muProfileLikelihood[muOpt[0] - stepIdx];
    muConfInt[1] = abs(muProfile[muOpt[0] - stepIdx] - muOpt[1]);

    sigConfInt = [0.0, 0.0];
    stepIdx = 0;
    logLikelihood = sigOpt[3];
    while 2*abs(logLikelihood - sigOpt[3]) < chi2ppf:
        stepIdx += 1;
        if sigOpt[0] + stepIdx > sigRangeN:
            print("WARNING: Please increase the search range for sigma.");
            break;
        logLikelihood = sigProfileLikelihood[sigOpt[0] + stepIdx];
    sigConfInt[0] = abs(sigProfile[sigOpt[0] + stepIdx] - sigOpt[2]);
    stepIdx = 0;
    logLikelihood = sigOpt[3];
    while 2*abs(logLikelihood - sigOpt[3]) < chi2ppf:
        stepIdx += 1;
        if sigOpt[0] - stepIdx < 0:
            print("WARNING: Please increase the search range for sigma.");
            break;
        logLikelihood = sigProfileLikelihood[sigOpt[0] - stepIdx];
    sigConfInt[1] = abs(sigProfile[sigOpt[0] - stepIdx] - sigOpt[2]);
#alpha confidence convertion
    alphaCov = alpha + (1.0 - alpha)/2.0;
    errBarRatio = stats.norm.ppf(alphaCov);
    errMu      = errBarRatio*errMu;
    if errSig > 0:
        errSig = errBarRatio*errSig;
    maxErrMu   = errBarRatio*maxErrMu;
    maxErrSig  = errBarRatio*maxErrSig;
#plots
    fig = plt.figure(figsize=(18, 14));
    gs = gridspec.GridSpec(2, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);
    ax2 = fig.add_subplot(gs[2]);
    ax3 = fig.add_subplot(gs[3]);
    #plot 0
    gaussPlot = gaussian(dataMu, dataSig, nbins);
    ax0.plot(nbins, dataHist, linewidth=2, color="blue", drawstyle="steps-post");
    ax0.plot(nbins, gaussPlot*np.sum(dataHist)/np.sum(gaussPlot), linewidth=2, \
             alpha=0.8, color="red")
    ax0.axhline(y=0, color="black", linestyle="-");
    ax0.axvline(x=np.average(dataPDF), ymin=0, ymax=1, color="green", \
                linestyle="--");
    ax0.set_title("Data and Point Estimate, alpha="+str(alpha),\
                  fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0]-1.0, rangeX[1]+1.0);
    
    digit0       = -math.floor(math.log10(errMu)) + 1;
    valMu0r      = ("{:." + str(digit0) + "f}").format(valMu);
    errMu0r      = ("{:." + str(digit0) + "f}").format(errMu);
    valSig0r     = ("{:." + str(digit0) + "f}").format(valSig);
    errSig0r     = "NA";
    if errSig > 0:
        errSig0r = ("{:." + str(digit0) + "f}").format(errSig);
    xmin, xmax = ax0.get_xlim();
    ymin, ymax = ax0.get_ylim();
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18};
    strTemp = "True Val: ";
    ax0.text(xmin+0.01*(xmax-xmin),ymin+0.96*(ymax-ymin),strTemp,fontdict=font); 
    strTemp = "mu = " + str(dataMu);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.92*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "sig = " + str(dataSig);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.88*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "noise r = " + str(round(noiseR, 2));
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.84*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "Pt Est: ";
    ax0.text(xmin+0.01*(xmax-xmin),ymin+0.79*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "mu = " + str(valMu0r) + "$\pm$" + str(errMu0r);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.75*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "sig = " + str(valSig0r) + "$\pm$" + str(errSig0r);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.71*(ymax-ymin),strTemp,fontdict=font);
    #plot 3
    #idxList = np.unique(np.array(valTrace), axis=0, return_index=True)[1];
    #valTrace = [valTrace[idx] for idx in sorted(idxList)];
    tracePlotTolerance = pow(10, -3);
    plotTrace = [];
    prevXY = valTrace[0];
    plotTrace.append(valTrace[0]);
    for XY in valTrace[1:]:
        transDis = np.linalg.norm(np.array(XY) - np.array(prevXY));
        if transDis > tracePlotTolerance:
            plotTrace.append(XY);
        prevXY = deepcopy(XY);
        
    xList = [trace[0] for trace in plotTrace];
    yList = [trace[1] for trace in plotTrace];
    ax3.plot(xList, yList, "-x", color="black", markersize=6, markeredgewidth=2);
    ax3.set_title("Maximum Likelihood Optimization Trace", fontsize=24, y=1.03);
    ax3.set_xlabel("mu", fontsize=18);
    ax3.set_ylabel("sig", fontsize=18);
    ax3.set_xlim(muRange[0], muRange[1]);
    ax3.set_ylim(sigRange[0], sigRange[1]);
    #https://stackoverflow.com/questions/58342419
    dataframe = pd.DataFrame.from_dict({"x": xList, "y": yList});
    xDF = dataframe["x"];
    yDF = dataframe["y"];
    x0 = xDF.iloc[range(len(xDF)-1)].values;
    x1 = xDF.iloc[range(1,len(xDF))].values;
    y0 = yDF.iloc[range(len(yDF)-1)].values;
    y1 = yDF.iloc[range(1,len(yDF))].values;
    xpos = (x0+x1)/2;
    ypos = (y0+y1)/2;
    xdir = x1-x0;
    ydir = y1-y0;
    for i, (X, Y, dX, dY) in enumerate(zip(xpos, ypos, xdir, ydir)):
        ax3.annotate("", xytext=(X,Y), xy=(X+0.001*dX,Y+0.001*dY), size=20,\
                     arrowprops=dict(arrowstyle="->",color="blue",linewidth=2,\
                                     alpha=(1.0-1.0*i/len(xpos))));
    ax3.plot(xList[0],  yList[0],  "ro", color="blue", markersize=8, zorder=5); 
    ax3.plot(xList[-1], yList[-1], "ro", color="green", markersize=8, zorder=5);
    ax3.errorbar(xList[-1], yList[-1], xerr=maxErrMu, yerr=maxErrSig,\
                 color="green", linewidth=2.5, ls="none", zorder=5,\
                 capsize=6, markeredgewidth=2.5);

    digit1       = -math.floor(math.log10(maxErrMu)) + 1;
    maxLikeMu1r  = ("{:." + str(digit1) + "f}").format(maxLikeMu);
    maxErrMu1r   = ("{:." + str(digit1) + "f}").format(maxErrMu);
    maxLikeSig1r = ("{:." + str(digit1) + "f}").format(maxLikeSig);
    maxErrSig1r  = ("{:." + str(digit1) + "f}").format(maxErrSig);
    xmin, xmax = ax3.get_xlim();
    ymin, ymax = ax3.get_ylim();
    font = {"family": "serif", "color": "blue", "weight": "bold", "size": 10};
    ax3.text(valTrace[0][0]+0.01*(xmax-xmin),\
             valTrace[0][1]+0.01*(ymax-ymin), "init", fontdict=font);
    font = {"family": "serif", "color": "green", "weight": "bold", "size": 18};
    strTemp = "Max Like: ";
    ax3.text(xmin+0.01*(xmax-xmin),ymin+0.96*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "mu = " + str(maxLikeMu1r) + "$\pm$" + str(maxErrMu1r);
    ax3.text(xmin+0.05*(xmax-xmin),ymin+0.92*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "sig = " + str(maxLikeSig1r) + "$\pm$" + str(maxErrSig1r);
    ax3.text(xmin+0.05*(xmax-xmin),ymin+0.88*(ymax-ymin),strTemp,fontdict=font);
    #plot 1
    ax1.plot(muProfile, muProfileLikelihood, alpha=1.0, color="purple",\
             linewidth=3, drawstyle="steps-post", zorder=0);
    ax1.plot(muProfile, muMaxLikelihoodProfile, alpha=1.0, color="green",\
             linewidth=2, linestyle="--", drawstyle="steps-post");
    ax1.set_title("Log-Likelihood Mu Profile", fontsize=24, y=1.03);
    ax1.set_xlabel("mu", fontsize=18);
    ax1.set_ylabel("log-likelihood", fontsize=18);
    ax1.set_xlim(muRange[0], muRange[1]);

    ax1.axvline(x=muOpt[1], ymin=0, ymax=1, color="purple", linestyle="--");
    ax1.axvline(x=(muOpt[1]-muConfInt[0]), ymin=0, ymax=1, \
                color="purple", alpha=0.5, linestyle=":");
    ax1.axvline(x=(muOpt[1]+muConfInt[1]), ymin=0, ymax=1, \
                color="purple", alpha=0.5, linestyle=":");
    xmin, xmax = ax1.get_xlim();
    ymin, ymax = ax1.get_ylim();
    font = {"family": "serif", "color": "purple", "weight": "bold", "size": 18};
    digit2  = -math.floor(math.log10(max(muConfInt))) + 1;
    valMu2r  = ("{:." + str(digit2) + "f}").format(muOpt[1]);
    errMu2rN = ("{:." + str(digit2) + "f}").format(muConfInt[0]);
    errMu2rP = ("{:." + str(digit2) + "f}").format(muConfInt[1]);
    strTemp = "mu = "+str(valMu2r)+"+"+str(errMu2rP)+"-"+str(errMu2rN);
    ax1.text(muOpt[1], ymin+0.01*(ymax-ymin), strTemp, fontdict=font);
    strTemp = "Nuisance Par:\n  sig, noise r";
    ax1.text(xmin+0.72*(xmax-xmin), ymin+0.91*(ymax-ymin), strTemp,fontdict=font);
    #plot 2
    ax2.plot(sigProfileLikelihood, sigProfile, alpha=1.0, color="purple",\
             linewidth=3, drawstyle="steps-post", zorder=0);
    ax2.plot(sigMaxLikelihoodProfile, sigProfile, alpha=1.0, color="green",\
             linewidth=2, linestyle="--", drawstyle="steps-post");
    ax2.set_title("Log-Likelihood Sigma Profile", fontsize=24, y=1.03);
    ax2.set_xlabel("log-likelihood", fontsize=18);
    ax2.set_ylabel("sigma", fontsize=18);
    ax2.set_ylim(sigRange[0], sigRange[1]);
    

    ax2.axhline(y=sigOpt[2], xmin=0, xmax=1, color="purple", linestyle="--");
    ax2.axhline(y=(sigOpt[2]-sigConfInt[0]), xmin=0, xmax=1, \
                color="purple", alpha=0.5, linestyle=":");
    ax2.axhline(y=(sigOpt[2]+sigConfInt[1]), xmin=0, xmax=1, \
                color="purple", alpha=0.5, linestyle=":");
    ax2.invert_xaxis();
    xmin, xmax = ax2.get_xlim();
    ymin, ymax = ax2.get_ylim();
    digit3  = -math.floor(math.log10(max(sigConfInt))) + 1;
    valSig3r  = ("{:." + str(digit3) + "f}").format(sigOpt[2]);
    errSig3rN = ("{:." + str(digit3) + "f}").format(sigConfInt[0]);
    errSig3rP = ("{:." + str(digit3) + "f}").format(sigConfInt[1]);
    strTemp = "sig = "+str(valSig3r)+"+"+str(errSig3rP)+"-"+str(errSig3rN);
    ax2.text(xmin+0.5*(xmax-xmin), sigOpt[2], strTemp, fontdict=font);
    strTemp = "Nuisance Par:\n  mu, noise r";
    ax2.text(xmin+0.72*(xmax-xmin), ymin+0.91*(ymax-ymin), strTemp,fontdict=font);

    if verbosity >= 1:
        print("True Val: "); 
        print("    mu  = " + str(dataMu));
        print("    sig = " + str(dataSig), end = "");
        print("Pt Est: ");
        print("    mu  = " + str(valMu) + " +/- " + str(errMu));
        print("    sig = " + str(valSig), end = "");
        if errSig > 0:
            print(" +/- " + str(errSig));
        else:
            print("");
        print("Max Like: ");
        print("    mu  = " + str(maxLikeMu) + " +/- " + str(maxErrMu));
        print("    sig = " + str(maxLikeSig) + " +/- " + str(maxErrSig));
        print("Prof Like: ");
        print("    mu  = " + str(muOpt[1])  + \
              " + " + str(muConfInt[1])  + " - " + str(muConfInt[0]));
        print("    sig = " + str(sigOpt[2]) + \
              " + " + str(sigConfInt[1]) + " - " + str(sigConfInt[0]));
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/gausProfileUniNoise.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




