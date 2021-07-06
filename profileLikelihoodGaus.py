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
    vals = -np.log(sig*np.sqrt(2.0*np.pi))\
           -np.power(X-mu,2.0)/(2.0*np.power(sig,2.0));
    LL = sum(vals);
    return LL;
def negLogLikelihood(x, trace=None):
    return lambda par : -1.0*logGaus(par[0], par[1], x, trace);
def negLogLikeMu(mu, x):
    return lambda sig : -1.0*logGaus(mu, sig, x);
def negLogLikeSig(sig, x):
    return lambda mu : -1.0*logGaus(mu, sig, x);

    
def main():
    verbosity = 1;
    binN = 200;
    rangeX = [-10.0, 10.0];

    np.random.seed(2);
    dataMu  = 0.1;
    dataSig = 0.8;
    dataN   = 30;
    
    alpha           = 0.95;         #significance
    muRange         = [-2.0, 2.0];  #search range
    sigRange        = [0.2, 2.0];
    profileStepSize = 0.001;
#data
    nbins = np.linspace(rangeX[0], rangeX[1], binN);
    dataPDF = np.random.normal(dataMu, dataSig, dataN);
    dataHist = np.zeros(binN);
    for x in dataPDF:
        if rangeX[0] < x and x < rangeX[1]:
            dataHist[int(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
#point estimate
    valMu  = np.average(dataPDF);
    errMu  = np.std(dataPDF)/np.sqrt(dataN);
    valSig = np.sqrt(np.var(dataPDF));
    errSig = -1;
#maximum likelihood
    if verbosity >= 1:
        print("Processing maximum likelihood...");
    #optInitVals = [valMu, valSig];
    optInitVals = [1.0, 1.0];
    valTrace = [];
    negMaxLL = negLogLikelihood(dataPDF, trace=valTrace);
    optResult = optimize.minimize(negMaxLL, optInitVals, method="Nelder-Mead");
    [maxLikeMu, maxLikeSig] = optResult.x;
#max like standard error using sqrt 1/(Fisher information)
    maxErrMu = maxLikeSig*np.sqrt(1.0/dataN);
    maxErrSig = maxLikeSig*np.sqrt(1.0/(2.0*dataN));
#profile likelihood
    if verbosity >= 1:
        print("Likelihood profiling for mu...");
    muProfile = [];
    muProfileLikelihood = [];
    muMaxLikelihoodProfile = [];
    muOpt = [0, 0.0, 0.0, -pow(10, 24)];
    muRangeN = int((muRange[1]-muRange[0])/profileStepSize);
    for i in (tqdm(range(muRangeN)) if verbosity>=1 else range(muRangeN)):
        mu = muRange[0] + i*profileStepSize;
        negLL = negLogLikeMu(mu, dataPDF);
        optResult = optimize.minimize_scalar(negLL, tol=TOLERANCE,\
                                             method="bounded",\
                                             bounds=(sigRange[0], sigRange[1]));
        sig = 1.0*optResult.x;
        optLikelihood = -1.0*negLL(sig);
        muProfile.append(mu);
        if optLikelihood > muOpt[3]:
            muOpt = [i, mu, sig, optLikelihood];
        muProfileLikelihood.append(optLikelihood);
    for i in range(muRangeN):
        mu = muRange[0] + i*profileStepSize;
        negLL = negLogLikeMu(mu, dataPDF);
        optLikelihood = -1.0*negLL(maxLikeSig);
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
        negLL = negLogLikeSig(sig, dataPDF);
        optResult = optimize.minimize_scalar(negLL, tol=TOLERANCE,\
                                             method="bounded",\
                                             bounds=(muRange[0], muRange[1]));
        mu = 1.0*optResult.x;
        optLikelihood = -1.0*negLL(mu);
        sigProfile.append(sig);
        if optLikelihood > sigOpt[3]:
            sigOpt = [i, mu, sig, optLikelihood];
        sigProfileLikelihood.append(optLikelihood);
    for i in range(sigRangeN):
        sig = sigRange[0] + i*profileStepSize;
        negLL = negLogLikeSig(sig, dataPDF);
        optLikelihood = -1.0*negLL(maxLikeMu);
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
    ax0.plot(nbins, dataHist, linewidth=2, color="blue", linestyle="steps-mid");
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
    strTemp = "Pt Est: ";
    ax0.text(xmin+0.01*(xmax-xmin),ymin+0.83*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "mu = " + str(valMu0r) + "$\pm$" + str(errMu0r);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.79*(ymax-ymin),strTemp,fontdict=font);
    strTemp = "sig = " + str(valSig0r) + "$\pm$" + str(errSig0r);
    ax0.text(xmin+0.05*(xmax-xmin),ymin+0.75*(ymax-ymin),strTemp,fontdict=font);
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
             linewidth=3, ls="steps-mid", zorder=0);
    ax1.plot(muProfile, muMaxLikelihoodProfile, alpha=1.0, color="green",\
             linewidth=2, linestyle="--", ls="steps-mid");
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
    strTemp = "Nuisance Par:\n  sig";
    ax1.text(xmin+0.72*(xmax-xmin), ymin+0.91*(ymax-ymin), strTemp,fontdict=font);
    #plot 2
    ax2.plot(sigProfileLikelihood, sigProfile, alpha=1.0, color="purple",\
             linewidth=3, ls="steps-mid", zorder=0);
    ax2.plot(sigMaxLikelihoodProfile, sigProfile, alpha=1.0, color="green",\
             linewidth=2, linestyle="--", ls="steps-mid");
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
    strTemp = "Nuisance Par:\n  mu";
    ax2.text(xmin+0.72*(xmax-xmin), ymin+0.91*(ymax-ymin), strTemp,fontdict=font);

    if verbosity >= 1:
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
    filenameFig = exepath + "/gausProfile.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




