import numpy
import numbers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from time import time
import logging
import sys

logger = logging.getLogger()
defaultLogLevel = "WARN"
logger.setLevel('WARN')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel('WARN')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.debug("Established logger")

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
    if len(theList) == 0:
        return []
    if numSamples > 1:
        return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
    elif numSamples == 1:
        return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
    else:
        return []

def rep(elt, num):
    return [elt for i in range(num)]

def getRandomUniformPriorLR():
    # the purpose of this method is to return an initial in-silico prediction for the variant to bootstrap the variant LR
    myRandProbability = numpy.random.uniform(0.1, 0.9, 1)[0]
    return myRandProbability / (1 - myRandProbability)

def sampleEvidenceFromObservations(expectedNum, observations):
    # assign random uniform prior LR as first piece of evidence
    initialLR = [getRandomUniformPriorLR()]
    sampleLRs = sample(observations, int(expectedNum), replace=True)
    if len(sampleLRs) == 0:
        return []
    else:
        return initialLR + sampleLRs
        #return sampleLRs

def getExpectedNumsFromPSF(n, PSF):
    # for now, we've fixed PSF at a number but could in the future make it a distribution from which we sample
    fractionBenign = 1 / (PSF + 1)
    numBenign = int(fractionBenign * n)
    numPathogenic = n - numBenign
    return numBenign, numPathogenic

def sampleNumberOfPeopleWithVariant(n, freq):
    # here we're using the poisson dist b/c the sampling process satisfies:
    # 1. variant occurs randomly
    # 2. variant occurs independently
    # 3. variant counts are discrete (e.g. whole numbers only)
    # P(X=x) = lam^x * e^(-lam) / x! (where lam = mean, lam = variance)
    return sum(rpois(n, freq))

def calculateSumOfLogs(lrList):
    mySum = 0
    for sublist in lrList:
        for lr in sublist:
            mySum += math.log(lr, 10)
    return mySum

class testCenter:
    def __init__(self, name, initialSize, testsPerYear):
        self.name = name
        self.initialSize = initialSize
        self.currentSize = 0
        self.testsPerYear = testsPerYear
        self.benignObservations = list()
        self.pathogenicObservations = list()
        self.benignLRs = list()
        self.pathogenicLRs = list()
        self.benignLRPs = list()
        self.pathogenicLRPs = list()

    def runSimulation(self, pathogenicProbabilities, benignProbabilities, pathogenicLikelihoodRatios,
                      benignLikelihoodRatios, variantFrequency, PSF, numTests, year):

        # add tests to center count
        self.currentSize += numTests

        # generate observations of variant (assumed to be pathogenic) from people with variant
        self.pathogenicObservations += self.generateObservationsFromTests(pathogenicProbabilities, pathogenicLikelihoodRatios,
                                                                    benignLikelihoodRatios, numTests)

        # generate observations of variant (assumed to be benign) from people with variant
        self.benignObservations += self.generateObservationsFromTests(benignProbabilities, pathogenicLikelihoodRatios,
                                                                benignLikelihoodRatios, numTests)

        # use Poisson distribution to get number of people from this batch with that variant
        numPeopleWithVariant = sampleNumberOfPeopleWithVariant(numTests, variantFrequency)

        # use PSF to calculate expected number of benign/pathogenic observations for people with variant
        numExpectedBenign, numExpectedPathogenic = getExpectedNumsFromPSF(numPeopleWithVariant, PSF)

        # generate evidence for observations assumed pathogenic
        self.pathogenicLRs.append(sampleEvidenceFromObservations(numExpectedPathogenic, self.pathogenicObservations))

        # generate evidence for observations assumed benign
        self.benignLRs.append(sampleEvidenceFromObservations(numExpectedBenign, self.benignObservations))

        # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
        self.benignLRPs.append(calculateSumOfLogs(self.benignLRs))

        # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
        self.pathogenicLRPs.append(calculateSumOfLogs(self.pathogenicLRs))

    def generateObservationsFromTests(self, c, P, B, n):
        Obs = \
            [rep(P['PM'], int(c[2] * n)) + rep(B['BP'], int(c[4] * n)) +
             rep(B['BP'], int(c[5] * n)) + rep(P['PP'], int(c[6] * n)) +
             rep(P['PS'], int(c[6] * n)) + rep(B['BS'], int(c[7] * n)) +
             rep(B['BS'], int(c[8] * n)) +
             rep(1.0, int((1 - (c[2] + c[4] + c[5] + c[6] + c[6] + c[7] + c[8])) * n))]
        return Obs[0]

    def addBenignLRList(self, lrList):
        self.benignLRs.append(lrList)

    def addPathogenicLRList(self, lrList):
        self.pathogenicLRs.append(lrList)

    def getDistributionOfBenignObservations(self):
        obs = dict()
        for o in self.benignObservations:
            if not o in obs:
                obs[o] = 0
            obs[o] += 1
        return obs

    def getDistributionOfPathogenicObservations(self):
        obs = dict()
        for o in self.pathogenicObservations:
            if not o in obs:
                obs[o] = 0
            obs[o] += 1
        return obs

    def getDistributionOfBenignLRs(self):
        lrs = dict()
        for l in self.benignLRs:
            for lr in l:
                if not lr in lrs:
                    lrs[lr] = 0
                lrs[lr] += 1
        return lrs

    def getDistributionOfPathogenicLRs(self):
        lrs = dict()
        for l in self.pathogenicLRs:
            for lr in l:
                if not lr in lrs:
                    lrs[lr] = 0
                lrs[lr] += 1
        return lrs

    def getNumberOfLRs(self):
        size = 0
        for l in self.pathogenicLRs:
            size += len(l)
        for l in self.benignLRs:
            size+= len(l)
        return size

    def getNumberOfObservations(self):
        return len(self.pathogenicObservations) + len(self.benignObservations)

def plotLRPScatter(centerSimulations, f, year, thresholds):
    numberOfSims = len(centerSimulations)
    centerName = centerSimulations[0].name
    pathogenic_y = list()
    benign_y = list()

    for i in range(len(centerSimulations)):
        pathogenic_y.append(list())
        pathogenic_y[i].append(0)
        pathogenic_y[i] += centerSimulations[i].pathogenicLRPs[0:year]
        benign_y.append(list())
        benign_y[i].append(0)
        benign_y[i] += centerSimulations[i].benignLRPs[0:year]

    yearList = [i for i in range(0, year+1)]
    x = yearList
    '''y_plrps = [0] + center.pathogenicLRPs[0:year]
    y_blprs = [0] + center.benignLRPs[0:year]'''

    if f <= 1e-6:
        plt.xlim(0, 20)
        plt.ylim(-5, 5)
    elif f <= 1e-5:
        plt.xlim(0, 20)
        plt.ylim(-5, 5)
    else:
        plt.xlim(0, 20)
        plt.ylim(-5, 5)

    plt.axhline(y=thresholds[0], color='green', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=thresholds[1], color='blue', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=thresholds[2], color='black', linestyle='dashed', linewidth=1.0)
    plt.axhline(y=thresholds[3], color='orange', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=thresholds[4], color='red', linestyle='dashed', linewidth=0.75)

    for i in range(len(centerSimulations)):
        plt.plot(x, pathogenic_y[i], marker='x', color='red', label='pathogenic', alpha=1.0/(3+i))
        plt.plot(x, benign_y[i], marker='o', color='green', label='benign', alpha=1.0/(2+i))

    plt.ylabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.xlabel('year', fontsize=18)
    #plt.title(str(centerName) + '|freq=' + str(f) + '|year=' + str(year))

    #plt.show()
    plt.savefig('/Users/jcasaletto/Desktop/RESEARCH/BRIAN/MODEL/PLOTS/' + centerName + '_y' + str(year) + '_' + str(f) + '_lrp_scatter')
    plt.close()


def plotLRPHist(centerSimulations, f, year, thresholds):
    centerName = centerSimulations[0].name

    '''pathogenic_x = list()
    benign_x = list()
    for i in range(len(centerSimulations)):
        pathogenic_x.append(list())
        for lrList in centerSimulations[i].pathogenicLRs:
            if len(lrList) != 0:
                pathogenic_x[i].append(centerSimulations[i].pathogenicLRPs[year])
        benign_x.append(list())
        for lrList in centerSimulations[i].benignLRs:
            if len(lrList) != 0:
                benign_x[i].append(centerSimulations[i].benignLRPs[year])'''
    pathogenic_x = [0]
    benign_x = [0]
    for i in range(len(centerSimulations)):
        for lrList in centerSimulations[i].pathogenicLRs:
            if len(lrList) != 0:
                pathogenic_x.append(centerSimulations[i].pathogenicLRPs[year])
        for lrList in centerSimulations[i].benignLRs:
            if len(lrList) != 0:
                benign_x.append(centerSimulations[i].benignLRPs[year])

    # TODO calculate the plot limits dynamically, not hard-coded
    ax = plt.figure(figsize=(8, 6)).gca()
    if f <= 1e-6:
        plt.xlim(-5, 15)
        bins = numpy.arange(-5, 15, 1)
        plt.ylim(0, 1)
    elif f <= 1e-5:
        plt.xlim(-15, 40)
        bins = numpy.arange(-15, 40, 1)
        plt.ylim(0, 1)
    elif f <= 1e-4:
        plt.xlim(-20, 20)
        plt.ylim(0, 60)
    elif f <= 1e-3:
        plt.xlim(-15, 15)
        plt.ylim(0, 80)
    else:
        plt.xlim(-20, 20)
        plt.ylim(0, 100)

    plt.axvline(x=thresholds[0], color='green', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=thresholds[1], color='blue', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=thresholds[2], color='black', linestyle='dashed', linewidth=1.0)
    plt.axvline(x=thresholds[3], color='orange', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=thresholds[4], color='red', linestyle='dashed', linewidth=0.75)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], density=True, range=(-15, 50), bins=bins)

    plt.xlabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.ylabel('probability mass', fontsize=18)

    #plt.title(centerName + '|freq=' + str(f) + '|year=' + str(year))
    plt.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    #plt.show()
    plt.savefig('/Users/jcasaletto/Desktop/RESEARCH/BRIAN/MODEL/PLOTS/' + centerName + '_y' + str(year) + '_' + str(f) + '_lrp_hist')
    plt.close()

def findMinMax(myList):
    myMin = float('inf')
    myMax = -float('inf')
    if len(myList) == 0:
        return myMin, myMax
    elif isinstance(myList[0], list):
        for subList in myList:
            if len(subList) !=0 and min(subList) < myMin:
                myMin = min(subList)
            if len(subList) !=0 and max(subList) > myMax:
                myMax = max(subList)
    else:
        myMin = min(myList)
        myMax = max(myList)

    return myMin, myMax

def getLStats(LRStats, LRPStats, center):
    myMin, myMax = findMinMax(center.benignLRs)
    LRStats['minBenign'].append(center.name + ':' + str(myMin))
    LRStats['maxBenign'].append(center.name + ':' + str(myMax))

    myMin, myMax = findMinMax(center.pathogenicLRs)
    LRStats['minPathogenic'].append(center.name + ':' + str(myMin))
    LRStats['maxPathogenic'].append(center.name + ':' + str(myMax))

    myMin, myMax = findMinMax(center.benignLRPs)
    LRPStats['minBenign'].append(center.name + ':' + str(myMin))
    LRPStats['maxBenign'].append(center.name + ':' + str(myMax))

    myMin, myMax = findMinMax(center.pathogenicLRPs)
    LRPStats['minPathogenic'].append(center.name + ':' + str(myMin))
    LRPStats['maxPathogenic'].append(center.name + ':' + str(myMax))

def getStatisticsForSimulation(centerList):
    numObservations = 0
    numLRs = 0
    numLRPs = 0
    LRStats = {'minBenign': list(), 'maxBenign': list(), 'minPathogenic': list(), 'maxPathogenic': list()}
    LRPStats = {'minBenign': list(), 'maxBenign': list(), 'minPathogenic': list(), 'maxPathogenic': list()}

    for center in centerList:
        numLRs += center.getNumberOfLRs()
        numLRPs += len(center.pathogenicLRPs) + len(center.benignLRPs)
        numObservations += center.getNumberOfObservations()
        getLStats(LRStats, LRPStats, center)
        print('center = ' + center.name)
        print('obs = ' + str(center.getNumberOfObservations()))
        #print('dist of ben obs = ' + str(center.getDistributionOfBenignObservations()))
        #print('dist of path obs = ' + str(center.getDistributionOfPathogenicObservations()))
        #print('dist of ben lrs = ' + str(center.getDistributionOfBenignLRs()))
        #print('dist of path lrs = ' + str(center.getDistributionOfPathogenicLRs()))
        #print('total number of lrs = ' + str(center.getNumberOfLRs()))
        #print('observed freq = ' + str(center.getNumberOfLRs()/center.getNumberOfObservations()))
        print('ben lrs = ' + str(center.benignLRs))
        print('path lrs = ' + str(center.pathogenicLRs))
        print('ben lrps = ' + str(center.benignLRPs))
        print('path lrps = ' + str(center.pathogenicLRPs))

    print('num observations = ' + str(numObservations))
    print('num LRs = ' + str(numLRs))
    print('num LRPs = ' + str(numLRPs))

    print('numLRs/numObs = ' + str(numLRs/numObservations))
    #print('LRStats: ' + str(LRStats))
    #print('LRPStats: ' + str(LRPStats))

def combineCenter(center, allCenters, year):
    allCenters.pathogenicLRs.append([])
    allCenters.pathogenicLRs[year] += center.pathogenicLRs[year]
    allCenters.pathogenicLRPs.append(calculateSumOfLogs(allCenters.pathogenicLRs))

    allCenters.benignLRs.append([])
    allCenters.benignLRs[year] += center.benignLRs[year]
    allCenters.benignLRPs.append(calculateSumOfLogs(allCenters.benignLRs))

    allCenters.pathogenicObservations += center.pathogenicObservations
    allCenters.benignObservations += center.benignObservations
    allCenters.currentSize += center.currentSize


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
    PSF = 2  #pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    freq = 1e-5 # this is the frequency of the variant we are interested in
    numSimulations = 20
    thresholds = [math.log(0.001,10), math.log(1/18.07, 10), 0, math.log(18.07, 10), math.log(100, 10)]

    UWList = list()
    ambryList = list()
    invitaeList = list()
    arupList = list()
    allCentersList = list()
    centerListList  = [UWList, ambryList, invitaeList, arupList]

    for i in range(numSimulations):
        logger.info('============================================')
        logger.info('simulation = ' + str(i))
        logger.info('============================================')
        UWList.append(testCenter('UW', 15000, 3000))
        ambryList.append(testCenter('ambry', 1000000, 450000))
        invitaeList.append(testCenter('invitae', 1000000, 450000))
        arupList.append(testCenter('arup', 150000, 30000))
        allCentersList.append(testCenter('all', 0, 0))

        # first, populate each center's db with variants based on initial sizes
        for centers in centerListList:
            centers[i].runSimulation(p, b, P, B, freq, PSF, centers[i].initialSize, 0)
            # and combine all centers data into allCenters for year 0
            combineCenter(centers[i], allCentersList[i], 0)

        # second, simulate forward in time, add variants to each center's db based on tests per year
        yearsOfInterest = [1, 5, 10, 15, 20]
        for year in range(1, years+1):
            # run simulations at each center for subsequent years
            for centers in centerListList:
                startTime=time()
                centers[i].runSimulation(p, b, P, B, freq, PSF, centers[i].testsPerYear, year)
                endTime = time()
                #logger.info('center=' + centers[i].name + ' runSimulation took ' + str(endTime - startTime))
                # plot graphs for each individual center in years of interest
                if year in yearsOfInterest:
                    # plot the histograms
                    startTime = time()
                    plotLRPHist(centers, freq, year, thresholds)
                    endTime = time()
                    #logger.info('center=' + centers[i].name + ' lrphist took ' + str(endTime - startTime))

                    # plot the scatter plots
                    startTime = time()
                    plotLRPScatter(centers, freq, year, thresholds)
                    endTime = time()
                    #logger.info('center=' + centers[i].name + ' lrpscatter took ' + str(endTime - startTime))

                # combine the centers
                startTime = time()
                combineCenter(centers[i], allCentersList[i], year)
                endTime = time()
                logger.info('center=' + centers[i].name + ' combineCenter took ' + str(endTime - startTime))
            # plot graphs for combined center in years of interest
            if year in yearsOfInterest:
                # plot histograms for all centers combined
                startTime = time()
                plotLRPHist(allCentersList, freq, year, thresholds)
                endTime = time()
                #logger.info('center=all lrphist took ' + str(endTime - startTime))

                # plot scatter plots for all centers combined
                startTime = time()
                plotLRPScatter(allCentersList, freq, year, thresholds)
                endTime = time()
                #logger.info('center=all lrpscatter took ' + str(endTime - startTime))

    '''for centers in centerListList:
        startTime = time()
        getStatisticsForSimulation(centers)
        endTime = time()
        logger.info('center=' + centers[0].name + ' getStats took ' + str(endTime - startTime))
    
    getStatisticsForSimulation(allCentersList)'''



if __name__ == "__main__":
    main()
