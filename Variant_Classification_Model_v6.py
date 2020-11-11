import numpy
import numbers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from collections import defaultdict


def prod(theList):
    if not isinstance(theList, list):
        if isinstance(theList, numbers.Number):
            x=theList
            theList=list()
            theList.append(x)
        else:
            raise Exception('argument must be a list of numbers')
    result = 1.0
    for i in range(len(theList)):
        result *= theList[i]
    return result

def rpois(num, lam):
    return numpy.random.poisson(lam, num)

def sample(theList, numSamples, replace):
    if numSamples > 1:
        return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
    elif numSamples == 1:
        return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
    else:
        return []

def rep(elt, num):
    return [elt for i in range(num)]

def getRandomUniformPriorLR():
    myRandProbability = numpy.random.uniform(0.1, 0.9, 1)[0]
    return myRandProbability / (1 - myRandProbability)

def getEvidenceForObservations(observations, frequency):
    numVariants = sum(rpois(len(observations), frequency))
    return sample(observations, int(numVariants), replace=True)

class testCenter:
    def __init__(self, name, initialSize, testsPerYear):
        self.name = name
        self.initialSize = initialSize
        self.testsPerYear = testsPerYear
        self.benignObservations = list()
        self.pathogenicObservations = list()
        self.benignLRs = list()
        self.pathogenicLRs = list()

    def runSimulation(self, p, b, P, B, f, PSF, num):
        pathogenicObservations, benignObservations = self.generateNewObservations(num, p, b, P, B, PSF)
        benignLRs = [getRandomUniformPriorLR()]
        pathogenicLRs = [getRandomUniformPriorLR()]
        pathogenicLRs += getEvidenceForObservations(self.pathogenicObservations, f)
        benignLRs += getEvidenceForObservations(self.benignObservations, f)
        self.addBenignLRList(benignLRs)
        self.addPathogenicLRList(pathogenicLRs)

    def generateNewObservations(self, n, p, b, P, B, PSF):
        fractionBenign = 1 / (PSF + 1)
        numBenign = int(fractionBenign * n)
        numPathogenic = n - numBenign

        # observations for pathogenic variant
        Obs = \
            [rep(P['PM'], int(p[2] * numPathogenic)) + rep(B['BP'], int(p[4] * numPathogenic)) +
             rep(B['BP'], int(p[5] * numPathogenic)) + rep(P['PP'], int(p[6] * numPathogenic)) +
             rep(P['PS'], int(p[6] * numPathogenic)) + rep(B['BS'], int(p[7] * numPathogenic)) +
             rep(B['BS'], int(p[8] * numPathogenic)) +
             rep(1.0, int((1 - (p[2] + p[4] + p[5] + p[6] + p[6] + p[7] + p[8])) * numPathogenic))]
        pathObs = Obs[0]
        self.pathogenicObservations += pathObs

        # observations for benign variant
        ObsB = \
            [rep(P['PM'], int(b[2] * numBenign)) + rep(B['BP'], int(b[4] * numBenign)) +
             rep(B['BP'], int(b[5] * numBenign)) + rep(P['PP'], int(b[6] * numBenign)) +
             rep(P['PS'], int(b[6] * numBenign)) + rep(B['BS'], int(b[7] * numBenign)) +
             rep(B['BS'], int(b[8] * numBenign)) +
             rep(1.0, int((1 - (b[2] + b[4] + b[5] + b[6] + b[6] + b[7] + b[8])) * numBenign))]
        benObs = ObsB[0]
        self.benignObservations += benObs

        return pathObs, benObs


    def addBenignLRList(self, lrList):
        self.benignLRs.append(lrList)

    def addPathogenicLRList(self, lrList):
        self.pathogenicLRs.append(lrList)


    def getDistributionOfBenignObservations(self, P, B):
        obs = defaultdict()
        for p in P:
            obs[P[p]] = 0
        for b in B:
            obs[B[b]] = 0
        obs[1.0] = 0
        for o in self.benignObservations:
            obs[o] += 1
        return obs

    def getDistributionOfPathogenicObservations(self, P, B):
        obs = defaultdict()
        for p in P:
            obs[P[p]] = 0
        for b in B:
            obs[B[b]] = 0
        obs[1.0] = 0
        for o in self.pathogenicObservations:
            obs[o] += 1
        return obs

    def getDistributionOfBenignLRs(self, P, B):
        lrs = dict()
        for p in P:
            lrs[P[p]] = 0
        for b in B:
            lrs[B[b]] = 0
        lrs[1.0] = 0
        for l in self.benignLRs:
            for lr in l:
                if not lr in lrs:
                    lrs[lr] = 0
                lrs[lr] += 1
        return lrs

    def getDistributionOfPathogenicLRs(self, P, B):
        lrs = dict()
        for l in self.pathogenicLRs:
            for lr in l:
                if not lr in lrs:
                    lrs[lr] = 0
                lrs[lr] += 1
        return lrs

    def getNumberOfVariants(self):
        size = 0
        for l in self.pathogenicLRs:
            size += len(l)
        for l in self.benignLRs:
            size+= len(l)
        return size

    def getNumberOfObservations(self):
        return len(self.pathogenicObservations) + len(self.benignObservations)

def graphAllLR(centerList, f, years, thresholds, bins):
    numVars = 0
    observations = 0
    testsPerYear = 0
    benign_x = list()
    pathogenic_x = list()
    for center in centerList:
        for lrList in center.benignLRs:
            mySum = 0.0
            for lr in lrList:
                mySum += math.log(lr, 10)
            benign_x.append(mySum)
        for lrList in center.pathogenicLRs:
            mySum = 0.0
            for lr in lrList:
                mySum += math.log(lr, 10)
            pathogenic_x.append(mySum)
        numVars += center.getNumberOfVariants()
        testsPerYear += center.testsPerYear
        observations += center.getNumberOfObservations()
    ax = plt.figure(figsize=(8,6)).gca()
    if f <= 1e-6:
        plt.xlim(-4, 4)
        plt.ylim(0, 40)
    elif f <= 1e-5:
        plt.xlim(-8, 8)
        bins = numpy.arange(-8, 8 + 0.2, 0.2)
        plt.ylim(0, 50)
    elif f <= 1e-4:
        plt.xlim(-20, 20)
        plt.ylim(0, 60)
    elif f <= 1e-3:
        plt.xlim(-15, 15)
        plt.ylim(0,80)
    else:
        plt.xlim(-20, 20)
        plt.ylim(0, 100)
    plt.axvline(x=thresholds[0], color='green', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[1], color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[2], color='black', linestyle='solid', linewidth=1)
    plt.axvline(x=thresholds[3], color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[4], color='red', linestyle='dashed', linewidth=1)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], bins=bins)
    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], bins=bins)
    plt.xlabel(r'$\sum_{i} log(LR_i)$')
    plt.ylabel('binned counts')
    #centerNames = [x.name for x in centerList]
    if len(centerList) == 1:
        centerNames = centerList[0].name
    else:
        centerNames = 'all'
    plt.title('center=' + str(centerNames) + '|frequency=' + str(f) + '|variants=' + str(numVars) +
                '|observations=' + str(observations) + '|year=' + str(years))
    plt.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    #plt.savefig('/Users/jcasaletto/Desktop/RESEARCH/BRIAN/MODEL/PLOTS/' + centerNames + '_y' + str(years) + '_' + str(f))

def main():
    ### gene specific probablities for laboratory observations of pathogenic variants
    p0 = 0 # placeholder
    p1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive
    p2_PM6 = 0.007  # probability case is assumed de novo
    p3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance
    p4_BP2 = 0.001  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    p5_BP5 = 0.0001  # probability a case has an alternate molecular explanation
    p6_PP1 = 0.2  # probility of cosegregation supporting pathogenicity
    p7_PS2 = 0.003  # probabilty case is proven de novo
    p8_BS4 = 0.0001  # probibility of strong cosegregation against pathogenicity
    p = [p0, p1_PM3, p2_PM6, p3_BS2, p4_BP2, p5_BP5, p6_PP1, p7_PS2, p8_BS4]

    ### gene specific probablities for pathogenic observations of benign variats
    b0 = 0 # placeholder
    b1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive
    b2_PM6 = 0.007  # probability case is assumed de novo
    b3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance
    b4_BP2 = 0.008  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    b5_BP5 = 0.07  # probability a case has an alternate molecular explanation
    b6_PP1 = 0.01  # probility of cosegregation supporting pathogenicity (have bc and a benign brca1 variant; mom also # has bc and inherited that brca1 variant from mom; similarly path variants will get benign evidence)
    b7_PS2 = 0.003  # probabilty case is proven de novo
    b8_BS4 = 0.1  # probibility of strong cosegregation against pathogenicity
    b = [b0, b1_PM3, b2_PM6, b3_BS2, b4_BP2, b5_BP5, b6_PP1, b7_PS2, b8_BS4]

    # if you add up all the numbers, it's 30-40% which are VUS - they may be off  - we'll know after getting the data

    # straight from sean's paper
    PS = 18.7 # strong evidence for pathogenic
    PM = 4.3 # moderate evidence for pathogenic
    PP = 2.08 # supporting evidence for pathogenic
    P = {'PS': PS, 'PM': PM, 'PP': PP}
    BS = 1 / 18.7 # string evidence for benign
    BP = 1 / 2.08 # supporting evidence for benign
    B = {'BS': BS, 'BP': BP}

    years = 20 # how long in future to project (i.e. number of iterations in simulation)
    bins = 20 # number of bins for histogram plot
    PSF = 2  #pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    freq = 1e-5 # this is the frequency of the variant we are interested in

    UW = testCenter('UW', 15000, 3000)
    ambry = testCenter('ambry', 1000000, 450000)
    invitae = testCenter('invitae', 1000000, 450000)
    arup = testCenter('arup', 150000, 30000)
    centerList = [UW, ambry, invitae, arup]
    thresholds = [math.log(0.001,10), math.log(1/18.07, 10), 0, math.log(18.07, 10), math.log(100, 10)]

    # first, populate each center's db with variants based on initial sizes
    for center in centerList:
        center.runSimulation(p, b, P, B, freq, PSF, center.initialSize)
        graphAllLR([center], freq, 0, thresholds, bins)
    graphAllLR(centerList, freq, 0, thresholds, bins)
    # second, simulate forward in time, add variants to each center's db based on tests per year
    yearsOfInterest = [5, 20]
    for year in range(1, years+1):
        for center in centerList:
            center.runSimulation(p, b, P, B, freq, PSF, center.testsPerYear)
            if year in yearsOfInterest:
                graphAllLR([center], freq, year, thresholds, bins)
        if year in yearsOfInterest:
            graphAllLR(centerList, freq, year, thresholds, bins)

    numVariants = 0
    numObservations = 0
    for center in centerList:
        numVariants += center.getNumberOfVariants()
        numObservations += (len(center.benignObservations) + len(center.pathogenicObservations))
        benObsDist = center.getDistributionOfBenignObservations(P, B)
        pathObsDist = center.getDistributionOfPathogenicObservations(P, B)
        benVarDist = center.getDistributionOfBenignLRs(P, B)
        pathVarDist = center.getDistributionOfPathogenicLRs(P, B)
        print('center = ' + center.name + ' obs = ' + str(numObservations))
        print('dist of ben obs = ' + str(benObsDist))
        print('dist of path obs = ' + str(pathObsDist))
        print('dist of ben lrs = ' + str(benVarDist))
        print('dist of path lrs = ' + str(pathVarDist))
        print('ben lrs: ' + str(center.benignLRs))
    print('num variants = ' + str(numVariants))
    print('num observations = ' + str(numObservations))
    print('numVars/numObs = ' + str(numVariants/numObservations))

if __name__ == "__main__":
    main()