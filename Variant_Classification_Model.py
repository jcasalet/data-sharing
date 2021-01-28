import numpy
import numbers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import json
import logging
import sys
import os
import argparse
import pickle

logger = logging.getLogger()
defaultLogLevel = "INFO"
logger.setLevel('INFO')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel('INFO')
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

def rpois(num, lam, seed):
    if seed == 0:
        numpy.random.seed()
    else:
        numpy.random.seed(seed)
    return numpy.random.poisson(lam, num)

def sample(theList, numSamples, replace, seed):
    if seed == 0:
        numpy.random.seed()
    else:
        numpy.random.seed(seed)
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

def sampleEvidenceFromObservations(expectedNum, observations, seed):
    # assign random uniform prior LR as first piece of evidence
    initialLR = [getRandomUniformPriorLR()]
    sampleLRs = sample(observations, int(expectedNum), replace=True, seed=seed)
    if len(sampleLRs) == 0:
        return []
    else:
        return initialLR + sampleLRs
        #return sampleLRs

def getExpectedNumsFromPSF(n, PSF):
    # for now, we've fixed PSF at a number but could in the future make it a distribution from which we sample
    fractionBenign = 1.0 / (PSF +1)
    numBenign = int(fractionBenign * n)
    numPathogenic = n - numBenign
    return numBenign, numPathogenic

def sampleNumberOfPeopleWithVariant(n, freq, seed):
    # here we're using the poisson dist b/c the sampling process satisfies:
    # 1. variant occurs randomly
    # 2. variant occurs independently
    # 3. variant counts are discrete (e.g. whole numbers only)
    # P(X=x) = lam^x * e^(-lam) / x! (where lam = mean, lam = variance)
    return sum(rpois(n, freq, seed))

def calculateSumOfLogs(lrList):
    mySum = 0
    for sublist in lrList:
        for lr in sublist:
            mySum += math.log(lr, 10)
    return mySum

class Configuration:

    def __init__(self, configFileName):
        self.configFileName = configFileName

        if self.configFileName != '' and not os.path.exists(self.configFileName):
            logger.error('config file ' + self.configFileName + ' does not exist!', file=sys.stderr)
            sys.exit(1)

        with open(self.configFileName, 'r') as myFile:
            jsonData = myFile.read()
        self.data = json.loads(jsonData)


class Simulation:
    def __init__(self, config, saType, saParam):
        simulation = config['simulation']
        self.saType = saType
        self.saParam = saParam

        self.name = simulation['name']
        self.nSmall = simulation['nSmall']
        self.nMedium = simulation['nMedium']
        self.nLarge = simulation['nLarge']
        self.numVariants = simulation['numVariants']
        self.frequency = simulation['frequency']
        self.years = simulation['years']
        self.seed = simulation['seed']

        constants = config['constants']


        self.p = {'p0': constants['p0']['med'], 'p1_M3': constants['p1_PM3']['med'], 'p2_PM6': constants['p2_PM6']['med'],
                  'p3_BS2': constants['p3_BS2']['med'], 'p4_BP2': float(eval(constants['p4_BP2']['med'])),
                  'p5_BP5': constants['p5_BP5']['med'], 'p6_PP1': constants['p6_PP1']['med'],
                  'p7_PS2': constants['p7_PS2']['med'], 'p8_BS4': constants['p8_BS4']['med']}



        self.b = {'b0': constants['b0']['med'], 'b1_PM3': constants['b1_PM3']['med'], 'b2_PM6': constants['b2_PM6']['med'],
                  'b3_BS2': constants['b3_BS2']['med'], 'b4_BP2': float(eval(constants['b4_BP2']['med'])),
                  'b5_BP5': constants['b5_BP5']['med'], 'b6_PP1': constants['b6_PP1']['med'],
                  'b7_PS2': constants['b7_PS2']['med'], 'b8_BS4': constants['b8_BS4']['med']}

        # if doing SA, override single parameter value specified in saParam to saType
        if self.saParam is None:
            pass
        elif self.saParam.startswith('p'):
            if type(constants[self.saParam][self.saType]) is str:
                self.p[self.saParam] = eval(str(constants[self.saParam][self.saType]))
            else:
                self.p[self.saParam] = constants[self.saParam][self.saType]
        elif self.saParam.startswith('b'):
            if type(constants[self.saParam][self.saType]) is str:
                self.b[self.saParam] = eval(str(constants[self.saParam][self.saType]))
            else:
                self.b[self.saParam] = constants[self.saParam][self.saType]
        else:
            logger.error('unknown saParam: ' + str(self.saParam))
            sys.exit(1)

        self.P = {'PS': constants['PS'], 'PM': constants['PM'], 'PP': constants['PP']}
        self.B = {'BS': constants['BS'], 'BP': constants['BP']}

        self.PSF = constants['PSF']

        self.smallInitialSize = constants['smallInitialSize']
        self.smallTestsPerYear = constants['smallTestsPerYear']
        self.mediumInitialSize = constants['mediumInitialSize']
        self.mediumTestsPerYear = constants['mediumTestsPerYear']
        self.largeInitialSize = constants['largeInitialSize']
        self.largeTestsPerYear = constants['largeTestsPerYear']

        self.benignThreshold = constants['benignThreshold']
        self.likelyBenignThreshold = constants['likelyBenignThreshold']
        self.likelyPathogenicThreshold = constants['likelyPathogenicThreshold']
        self.pathogenicThreshold = constants['pathogenicThreshold']

        self.thresholds = [self.benignThreshold, self.likelyBenignThreshold, 0, self.likelyPathogenicThreshold, self.pathogenicThreshold]

        self.smallCenters = list()
        self.mediumCenters = list()
        self.largeCenters = list()
        self.centerListList = [self.smallCenters, self.mediumCenters, self.largeCenters]
        # initialize all centers
        for i in range(self.nSmall):
            self.smallCenters.append(TestCenter(name='small_' + str(i),
                                                initialSize=self.smallInitialSize,
                                                testsPerYear=self.smallTestsPerYear,
                                                numVariants=self.numVariants,
                                                seed=self.seed))
        for i in range(self.nMedium):
            self.mediumCenters.append(TestCenter(name='medium_' + str(i),
                                                 initialSize=self.mediumInitialSize,
                                                 testsPerYear=self.mediumTestsPerYear,
                                                 numVariants=self.numVariants,
                                                 seed=self.seed))
        for i in range(self.nLarge):
            self.largeCenters.append(TestCenter(name='large_' + str(i),
                                                initialSize=self.largeInitialSize,
                                                testsPerYear=self.largeTestsPerYear,
                                                numVariants=self.numVariants,
                                                seed=self.seed))

        self.allCenters = TestCenter(name='all',
                                initialSize=0,
                                testsPerYear=0,
                                numVariants=self.numVariants,
                                seed=self.seed)

        for centers in self.centerListList:
            for center in centers:
                center.runSimulation(self, center.initialSize)
                self.combineAllLRsFromCenter(center, 0)
        self.calculateAllLRPs()

    def getNumberOfCenters(self):
        return self.nSmall + self.nMedium + self.nLarge

    def calculateAllLRPs(self):
        for variant in range(self.numVariants):
            # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
            self.allCenters.benignLRPs[variant].append(calculateSumOfLogs(self.allCenters.benignLRs[variant]))
            # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
            self.allCenters.pathogenicLRPs[variant].append(calculateSumOfLogs(self.allCenters.pathogenicLRs[variant]))

    def combineAllLRsFromCenter(self, center, year):
        for variant in range(self.numVariants):
            self.allCenters.pathogenicLRs[variant].append([])
            self.allCenters.pathogenicLRs[variant][year] += center.pathogenicLRs[variant][year]
            self.allCenters.benignLRs[variant].append([])
            self.allCenters.benignLRs[variant][year] += center.benignLRs[variant][year]

    def run(self):
        # run simulation over years
        for year in range(1, self.years+1):
            # run simulations at each center for subsequent years
            for centers in self.centerListList:
                for center in centers:
                    center.runSimulation(self, center.testsPerYear)
                    self.combineAllLRsFromCenter(center, year)
            self.calculateAllLRPs()
        # after all the data is generated, calculate the probability of classification for each center
        for centers in self.centerListList:
            for center in centers:
                center.probabilityOfClassification(self.thresholds, self.years)
        self.allCenters.probabilityOfClassification(self.thresholds, self.years)

    def scatter(self, outputDir):
        for year in [self.years]:
            for centers in self.centerListList:
                for center in centers:
                    plotLRPScatter(self, center, year,  outputDir)
            plotLRPScatter(self, self.allCenters, year, outputDir)

    def hist(self, outputDir):
        for year in [self.years]:
            for centers in self.centerListList:
                for center in centers:
                    plotLRPHist(self, center, year, outputDir)
            plotLRPHist(self, self.allCenters, year, outputDir)

    def prob(self, outputDir):
        for centers in self.centerListList:
            for center in centers:
                plotProbability(self, center, outputDir)
        plotProbability(self, self.allCenters,  outputDir)

    def save(self, outputDir):
        for centers in self.centerListList:
            for center in centers:
                saveProbability(self, center, outputDir)
        saveProbability(self, self.allCenters,  outputDir)

class TestCenter:
    def __init__(self, name, initialSize, testsPerYear, numVariants, seed):
        self.name = name
        self.initialSize = initialSize
        self.testsPerYear = testsPerYear
        self.numVariants = numVariants
        self.seed = seed
        self.benignObservations = list()
        self.pathogenicObservations = list()
        self.benignLRs = dict()
        self.pathogenicLRs = dict()
        self.benignLRPs = dict()
        self.pathogenicLRPs = dict()
        self.benignProbabilities = [0]
        self.pathogenicProbabilities = [0]
        self.likelyBenignProbabilities = [0]
        self.likelyPathogenicProbabilities = [0]

        # create key for variant in each dict
        for variant in range(numVariants):
            self.benignLRs[variant] = list()
            self.pathogenicLRs[variant] = list()
            self.benignLRPs[variant] = list()
            self.pathogenicLRPs[variant] = list()


    def runSimulation(self, simulation, numTests):
        for variant in range(self.numVariants):
            # generate observations of variant (assumed to be pathogenic) from people with variant
            self.pathogenicObservations = self.generatePathogenicObservationsFromTests(simulation.p,
                            simulation.P, simulation.B, numTests)

            # generate observations of variant (assumed to be benign) from people with variant
            self.benignObservations = self.generateBenignObservationsFromTests(simulation.b, simulation.P,
                                                                    simulation.B, numTests)

            # use Poisson distribution to get number of people from this batch with that variant
            numPeopleWithVariant = sampleNumberOfPeopleWithVariant(numTests, simulation.frequency, self.seed)

            # use PSF to calculate expected number of benign/pathogenic observations for people with variant
            numExpectedBenign, numExpectedPathogenic = getExpectedNumsFromPSF(numPeopleWithVariant, simulation.PSF)


            # generate evidence for observations assumed pathogenic
            self.pathogenicLRs[variant].append(sampleEvidenceFromObservations(numExpectedPathogenic, self.pathogenicObservations, self.seed))

            # generate evidence for observations assumed benign
            self.benignLRs[variant].append(sampleEvidenceFromObservations(numExpectedBenign, self.benignObservations, self.seed))

            # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
            self.benignLRPs[variant].append(calculateSumOfLogs(self.benignLRs[variant]))

            # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
            self.pathogenicLRPs[variant].append(calculateSumOfLogs(self.pathogenicLRs[variant]))

    def generatePathogenicObservationsFromTests(self, c, P, B, n):
        Obs = \
            [rep(P['PM'], int(c['p2_PM6'] * n)) + rep(B['BP'], int(c['p4_BP2'] * n)) +
             rep(B['BP'], int(c['p5_BP5'] * n)) + rep(P['PP'], int(c['p6_PP1'] * n)) +
             rep(P['PS'], int(c['p6_PP1'] * n)) + rep(B['BS'], int(c['p7_PS2'] * n)) +
             rep(B['BS'], int(c['p8_BS4'] * n)) +
             rep(1.0, int((1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] + c['p6_PP1'] +
                                c['p7_PS2'] + c['p8_BS4'])) * n))]
        return Obs[0]

    def generateBenignObservationsFromTests(self, c, P, B, n):
        Obs = \
            [rep(P['PM'], int(c['b2_PM6'] * n)) + rep(B['BP'], int(c['b4_BP2'] * n)) +
             rep(B['BP'], int(c['b5_BP5'] * n)) + rep(P['PP'], int(c['b6_PP1'] * n)) +
             rep(P['PS'], int(c['b6_PP1'] * n)) + rep(B['BS'], int(c['b7_PS2'] * n)) +
             rep(B['BS'], int(c['b8_BS4'] * n)) +
             rep(1.0, int((1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5'] + c['b6_PP1'] + c['b6_PP1'] +
                                c['b7_PS2'] + c['b8_BS4'])) * n))]
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

    def probabilityOfClassification(self, thresholds, years):
        LB = thresholds[0]
        B = thresholds[1]
        neutral = thresholds[2]
        LP = thresholds[3]
        P = thresholds[4]

        for year in range(years):
            pathogenic_y = list()
            benign_y = list()
            for variant in range(self.numVariants):
                pathogenic_y.append(list())
                pathogenic_y[variant].append(0)
                pathogenic_y[variant] += self.pathogenicLRPs[variant][year:year+1]
                benign_y.append(list())
                benign_y[variant].append(0)
                benign_y[variant] += self.benignLRPs[variant][year:year+1]

            numPathogenicClassified = 0
            numBenignClassified = 0

            for variant in range(self.numVariants):
                for lrp in pathogenic_y[variant]:
                    if lrp > P:
                        numPathogenicClassified += 1
                        break
                for lrp in benign_y[variant]:
                    if lrp < B:
                        numBenignClassified += 1
                        break

            self.benignProbabilities.append(numBenignClassified / self.numVariants)
            self.pathogenicProbabilities.append(numPathogenicClassified / self.numVariants)

            numLPClassified = 0
            numLBClassified = 0

            for variant in range(self.numVariants):
                for lrp in pathogenic_y[variant]:
                    if lrp > LP:
                        numLPClassified += 1
                        break
                for lrp in benign_y[variant]:
                    if lrp < LB:
                        numLBClassified += 1
                        break

            self.likelyBenignProbabilities.append(numLBClassified / self.numVariants)
            self.likelyPathogenicProbabilities.append(numLPClassified / self.numVariants)


def plotLRPScatter(simulation, center, year, outputDir):
    centerName = center.name
    pathogenic_y = list()
    benign_y = list()
    for variant in range(center.numVariants):
        pathogenic_y.append(list())
        pathogenic_y[variant].append(0)
        pathogenic_y[variant] += center.pathogenicLRPs[variant][0:year]
        benign_y.append(list())
        benign_y[variant].append(0)
        benign_y[variant] += center.benignLRPs[variant][0:year]

    yearList = [i for i in range(0, year+1)]
    x = yearList

    plt.xlim(0, simulation.years)
    plt.ylim(-5, 5)

    plt.axhline(y=simulation.thresholds[0], color='green', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=simulation.thresholds[1], color='blue', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=simulation.thresholds[2], color='black', linestyle='dashed', linewidth=1.0)
    plt.axhline(y=simulation.thresholds[3], color='orange', linestyle='dashed', linewidth=0.75)
    plt.axhline(y=simulation.thresholds[4], color='red', linestyle='dashed', linewidth=0.75)

    for variant in range(center.numVariants):
        plt.plot(x, pathogenic_y[variant], marker='x', color='red', label='pathogenic', alpha=1.0/(3+variant))
        plt.plot(x, benign_y[variant], marker='o', color='green', label='benign', alpha=1.0/(2+variant))

    plt.ylabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.xlabel('year', fontsize=18)
    plt.title(centerName)
    #plt.legend(loc='upper right')

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    #plt.show()

    plt.savefig(outputDir + '/' + simulation.saType + '_' + simulation.name + '_' + centerName + '_' + str(year) + 'yrs_' +
                str(simulation.frequency) + '_' + dist + '_lrp_scatter')
    plt.close()


def plotLRPHist(simulation, center, year, outputDir):
    centerName = center.name

    pathogenic_x = [0]
    benign_x = [0]
    for variant in center.pathogenicLRs:
        for lrList in center.pathogenicLRs[variant]:
            if len(lrList) != 0:
                pathogenic_x.append(center.pathogenicLRPs[variant][year])
    for variant in center.benignLRs:
        for lrList in center.benignLRs[variant]:
            if len(lrList) != 0:
                benign_x.append(center.benignLRPs[variant][year])

    if centerName == 'all':
        print('max path x  = ' + str(max(pathogenic_x)))
        print('min ben x  = ' + str(max(benign_x)))

    # TODO calculate the plot limits dynamically, not hard-coded
    ax = plt.figure(figsize=(8, 6)).gca()
    '''if f <= 1e-6:
        plt.xlim(-5, 15)
        bins = numpy.arange(-5, 15, 0.5)
        plt.ylim(0, 1)
    elif f <= 1e-5:
        plt.xlim(-15, 60)
        bins = numpy.arange(-15, 60, 0.5)
        plt.ylim(0, 1)
    elif f <= 1e-4:
        plt.xlim(-20, 20)
        plt.ylim(0, 60)
    elif f <= 1e-3:
        plt.xlim(-15, 15)
        plt.ylim(0, 80)
    else:
        plt.xlim(-20, 20)
        plt.ylim(0, 100)'''

    #lowerLimit = min(min(benign_x), min(pathogenic_x))
    #upperLimit = max(max(benign_x), max(pathogenic_x))
    lowerLimit = -10
    upperLimit = 50
    plt.xlim(lowerLimit, upperLimit)
    bins = numpy.arange(lowerLimit, upperLimit, 0.5)
    plt.ylim(0, 1)


    plt.axvline(x=simulation.thresholds[0], color='green', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[1], color='blue', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[2], color='black', linestyle='dashed', linewidth=1.0)
    plt.axvline(x=simulation.thresholds[3], color='orange', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[4], color='red', linestyle='dashed', linewidth=0.75)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], density=True, range=(-15, 50), bins=bins)

    plt.xlabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.ylabel('probability mass', fontsize=18)

    plt.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)
    plt.savefig(outputDir + '/' + simulation.saType + '_' + simulation.saParam + '_' + simulation.name + '_' + \
                centerName + '_' + str(year) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_lrphist')
    plt.close()

def plotProbability(simulation, center, outputDir):

    yearList = [i for i in range(0, simulation.years + 1)]
    plt.xlim(0, simulation.years)
    plt.ylim(0, 1)
    plt.plot(yearList, center.pathogenicProbabilities, marker='x', color='red', label='pathogenic')
    plt.plot(yearList, center.benignProbabilities, marker='o', color='green', label='benign')
    plt.plot(yearList, center.likelyPathogenicProbabilities, marker='x', color='orange', label=' likely pathogenic', linestyle='dashed')
    plt.plot(yearList, center.likelyBenignProbabilities, marker='o', color='blue', label=' likely benign', linestyle='dashed')

    plt.ylabel('probability of classification', fontsize=18)
    plt.xlabel('year', fontsize=18)
    plt.title(center.name)
    #plt.legend(loc='upper right')
    #plt.show()

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    plt.savefig(outputDir + '/' + simulation.saType + '_' + simulation.saParam + '_' + simulation.name + '_' + \
                center.name + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs')
    plt.close()

def saveProbability(simulation, center, outputDir):

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)


    outFile = outputDir + '/' + simulation.saType + '_' + simulation.saParam + '_' + simulation.name + '_' + \
              center.name + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs.dat'
    with open(outFile, 'wb') as output:
        pickle.dump(center, output, pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputDir', help='output directory for PNG plots')
    parser.add_argument('-c', '--confFile', help='path to JSON configuration file')
    parser.add_argument('-j', '--jobType', help='job type to execute: "simulate" or "analyze"')
    options = parser.parse_args()
    return options

def main():
    confFile = parse_args().confFile
    outputDir = parse_args().outputDir
    jobType = parse_args().jobType
    # diff mixes: 4s + 2m + 1l; 8s + 4m + 5l
    config = Configuration(confFile)

    types = ['low', 'med', 'hi']
    parameters = ["p2_PM6", "p4_BP2", "p5_BP5", "p6_PP1", "p7_PS2", "p8_BS4",
                  "b2_PM6", "b4_BP2", "b5_BP5", "b6_PP1", "b7_PS2", "b8_BS4"]
    if jobType == 'simulate':
        print('simulate this!')
        mySimulation = Simulation(config=config.data, saType='med', saParam=None)
        mySimulation.run()
        mySimulation.scatter(outputDir=outputDir)
        mySimulation.hist(outputDir=outputDir)
        mySimulation.prob(outputDir=outputDir)
    elif jobType == 'analyze':
        print('analyze this!')
        for t in types:
            for p in parameters:
                mySimulation = Simulation(config=config.data, saType=t, saParam=p)
                mySimulation.run()
                #mySimulation.scatter(outputDir=outputDir)
                #mySimulation.hist(outputDir=outputDir)
                mySimulation.prob(outputDir=outputDir)
                #mySimulation.save(outputDir=outputDir)

    else:
        print('whats this?: ' + jobType)

if __name__ == "__main__":
    main()
