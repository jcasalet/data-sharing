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
from multiprocessing import Process, Queue, cpu_count
import matplotlib.patches as mpatches



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

def rpois(num, lam, rng):
    #numpy.random.seed()
    #return numpy.random.poisson(lam, num)
    return rng.poisson(lam, num)

def sample(theList, numSamples, replace, rng):
    #numpy.random.seed()
    if len(theList) == 0:
        return []
    if numSamples > 1:
        #return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
        return list(rng.choice(a=theList, size=numSamples, replace=replace))
    elif numSamples == 1:
        #return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))
        return list(rng.choice(a=theList, size=numSamples, replace=replace))
    else:
        return []

def rep(elt, num):
    return [elt for i in range(num)]

def getRandomUniformPriorLR(rng):
    # the purpose of this method is to return an initial in-silico prediction for the variant to bootstrap the variant LR
    #numpy.random.seed()
    #myRandProbability = numpy.random.uniform(0.1, 0.9, 1)[0]
    myRandProbability = rng.uniform(0.1, 0.9, 1)[0]
    return myRandProbability / (1 - myRandProbability)

def sampleEvidenceFromObservations(expectedNum, observations, rng):
    # assign random uniform prior LR as first piece of evidence
    initialLR = [getRandomUniformPriorLR(rng)]
    sampleLRs = sample(observations, int(expectedNum), replace=True, rng=rng)
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

def sampleNumberOfPeopleWithVariant(n, freq, rng):
    # here we're using the poisson dist b/c the sampling process satisfies:
    # 1. variant occurs randomly
    # 2. variant occurs independently
    # 3. variant counts are discrete (e.g. whole numbers only)
    # P(X=x) = lam^x * e^(-lam) / x! (where lam = mean, lam = variance)
    return sum(rpois(n, freq, rng))

def calculateSumOfLogs(lrList):
    mySum = 0
    for sublist in lrList:
        for lr in sublist:
            mySum += math.log(lr, 10)
    return mySum


def divide(n, d):
   res = list()
   qu = int(n/d)
   rm = n%d
   for i in range(d):
       if i < rm:
           res.append(qu + 1)
       else:
           res.append(qu)
   return res

def getStartAndEnd(partitionSizes, threadID):
    start = 0
    for i in range(threadID):
        start += partitionSizes[i]
    end = start + partitionSizes[threadID]

    return start, end


class Configuration:

    def __init__(self, configFileName):
        self.configFileName = configFileName

        if self.configFileName != '' and not os.path.exists(self.configFileName):
            logger.error('config file ' + self.configFileName + ' does not exist!', file=sys.stderr)
            sys.exit(1)

        with open(self.configFileName, 'r') as myFile:
            jsonData = myFile.read()
        self.data = json.loads(jsonData)

def calculateProbabilityOfUnion(year, subsetByLenDict):
    # subsetByLenDict = {'1': [(a,), (b,), (c,)], '2': [(a, b), (a, c), (b, c)], '3': [(a, b, c)]}
    probabilities = dict()
    for length in subsetByLenDict:
        probabilities[length] = {'LB': 0, 'B':0, 'LP': 0, 'P':0}
        for subset in subsetByLenDict[length]:
            tempLB = 1.0
            tempB = 1.0
            tempLP = 1.0
            tempP = 1.0
            for center in subset:
                tempLB *= center.likelyBenignProbabilities[year]
                tempB *= center.benignProbabilities[year]
                tempLP *= center.likelyPathogenicProbabilities[year]
                tempP *= center.pathogenicProbabilities[year]
            probabilities[length]['LB'] += tempLB
            probabilities[length]['B'] += tempB
            probabilities[length]['LP'] += tempLP
            probabilities[length]['P'] += tempP

    probabilityUnion = {'LB': 0, 'B': 0, 'LP':0, 'P':0}
    for length in probabilities:
        if int(length) % 2 == 1:
            probabilityUnion['LB'] += probabilities[length]['LB']
            probabilityUnion['B'] += probabilities[length]['B']
            probabilityUnion['LP'] += probabilities[length]['LP']
            probabilityUnion['P'] += probabilities[length]['P']
        else:
            probabilityUnion['LB'] -= probabilities[length]['LB']
            probabilityUnion['B'] -= probabilities[length]['B']
            probabilityUnion['LP'] -= probabilities[length]['LP']
            probabilityUnion['P'] -= probabilities[length]['P']

    return probabilityUnion


    # {'1':    {'LB': P(LB(a)) + P(LB(b) + P(LB(c)),
    #           'B': P(B(a)) + P(B(b)) + P(B(c)),
    #           'LP': P(LP(a)) + P(LP(b)) + P(LP(c))
    #           'P': P(P(a)) + P(P(b)) + P(P(c))}
    # '2':      {'LB': P(LB(a))*P(LB(b)) + P(LB(a))*P(LB(c)) + P(LB(b))*P(LB(c)),
    #           'B': P(B(a))*P(B(b)) + P(B(a))*P(B(c)) + P(B(b))*P(B(c)),
    #           'LP': P(LP(a))*P(LP(b)) + P(LP(a))*P(LP(c)) + P(LP(b))*P(LP(c)),
    #           'P': P(P(a))*P(P(b)) + P(P(a))*P(P(c)) + P(P(b))*P(P(c))},
    #  ...
    # '20': },

    # {'size': {LB: sum-prod-probs, B: sum-prod-probs, LP: sum-prod-probs, P: sum-prod-probs}



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
        self.seed = int(simulation['seed'])
        self.numThreads = simulation['numThreads']

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
                                                numVariants=self.numVariants))
        for i in range(self.nMedium):
            self.mediumCenters.append(TestCenter(name='medium_' + str(i),
                                                 initialSize=self.mediumInitialSize,
                                                 testsPerYear=self.mediumTestsPerYear,
                                                 numVariants=self.numVariants))
        for i in range(self.nLarge):
            self.largeCenters.append(TestCenter(name='large_' + str(i),
                                                initialSize=self.largeInitialSize,
                                                testsPerYear=self.largeTestsPerYear,
                                                numVariants=self.numVariants))

        self.allCenters = TestCenter(name='all',
                                initialSize=0,
                                testsPerYear=0,
                                numVariants=self.numVariants)

        for centers in self.centerListList:
            for center in centers:
                q = Queue()
                processList = list()
                for i in range(self.numThreads):
                    rng = numpy.random.default_rng(self.seed + (i+1) * (centers.index(center)+1))
                    p = Process(target=center.runSimulation, args=(self, center.initialSize, self.numThreads, i, q,
                                                                   rng))
                    p.start()
                    processList.append(p)
                for i in range(self.numThreads):
                    self.myUpdate(center, q.get())
                for i in range(self.numThreads):
                    processList[i].join()
                #center.runSimulation(self, center.initialSize)
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

    def divide(n, d):
        res = list()
        qu = int(n / d)
        rm = n % d
        for i in range(d):
            if i < rm:
                res.append(qu + 1)
            else:
                res.append(qu)
        return res

    def getStartAndEnd(partitionSizes, threadID):
        start = 0
        for i in range(threadID):
            start += partitionSizes[i]
        end = start + partitionSizes[threadID]

        return start, end

    def run(self):
        # run simulation over years
        for year in range(1, self.years+1):
            # run simulations at each center for subsequent years
            for centers in self.centerListList:
                for center in centers:
                    q = Queue()
                    processList = list()
                    for i in range(self.numThreads):
                        rng = numpy.random.default_rng(self.seed + (i+1)*(year+1)*(centers.index(center) + 1))
                        p = Process(target=center.runSimulation, args=(self, center.testsPerYear, self.numThreads, i,q,
                                                                       rng))
                        p.start()
                        processList.append(p)
                    for i in range(self.numThreads):
                        self.myUpdate(center, q.get())
                    for i in range(self.numThreads):
                        processList[i].join()
                    #center.runSimulation(self, center.testsPerYear)
                    self.combineAllLRsFromCenter(center, year)
            self.calculateAllLRPs()
        # after all the data is generated, calculate the probability of classification for each center
        for centers in self.centerListList:
            for center in centers:
                center.probabilityOfClassification(self.thresholds, self.years)
        self.allCenters.probabilityOfClassification(self.thresholds, self.years)
        # TODO now that each center has a probability of classification for that year, calculate the
        # probability of classification of any of the centers (i.e. P(A or B or C or ...))
        # use https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle#In_probability
        self.inclusionExclusionProbability()
        self.unionProbability()

    def unionProbability(self):
        # P(A U B U C ... U Z) = (sum size 1 sets) - (sum size 2 sets) + (sum size 3 sets) - ...
        import itertools
        subsets = list()
        centers = set()
        for cl in self.centerListList:
            for c in cl:
                centers.add(c)
        for i in range(1, len(centers) + 1):
            subsets.append(itertools.combinations(centers, i))

        # {'1': [(a,), (b,), (c,)], '2': [(a, b), (a, c), (b, c)], '3': [(a, b, c)]}
        subsetByLenDict = dict()
        for subset in subsets:
            for s in subset:
                l = str(len(s))
                if not str(l) in subsetByLenDict:
                    subsetByLenDict[l] = []
                subsetByLenDict[l].append(s)

        # subsetByLenDict = {'1': [(a,), (b,), (c,)], '2': [(a, b), (a, c), (b, c)], '3': [(a, b, c)]}
        probabilityUnions = []
        for year in self.years:
            probabilityUnions.append(calculateProbabilityOfUnion(year, subsetByLenDict))

        print(probabilityUnions)

    def inclusionExclusionProbability(self):
        # https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle#In_probability
        # calculate for each year, store in list or dictionary
        LB = list()
        B = list()
        LP = list()
        P = list()
        for center in self.largeCenters:
            LB.append(center.likelyBenignProbabilities)
            B.append(center.benignProbabilities)
            LP.append(center.likelyPathogenicProbabilities)
            P.append(center.pathogenicProbabilities)
        for center in self.mediumCenters:
            LB.append(center.likelyBenignProbabilities)
            B.append(center.benignProbabilities)
            LP.append(center.likelyPathogenicProbabilities)
            P.append(center.pathogenicProbabilities)
        for center in self.smallCenters:
            LB.append(center.likelyBenignProbabilities)
            B.append(center.benignProbabilities)
            LP.append(center.likelyPathogenicProbabilities)
            P.append(center.pathogenicProbabilities)

        self.LBanyCenter = []
        self.BanyCenter = []
        self.LPanyCenter = []
        self.PanyCenter = []
        for year in range(self.years+1):
            LBprob = 1.0
            Bprob = 1.0
            LPprob = 1.0
            Pprob = 1.0
            for center in range(len(LB)):
                LBprob *= (1.0 - LB[center][year])
                Bprob *= (1.0 - B[center][year])
                LPprob *= (1.0 - LP[center][year])
                Pprob *= (1.0 - P[center][year])
            # here is the problem.
            # recall that P(LB) + P(B) <= 1 for any center.
            # so it's false that LB's add to 1 for a center
            # so subtracting from 1 is wrong

            self.LBanyCenter.append(1 - LBprob)
            self.BanyCenter.append(1 - Bprob)
            self.LPanyCenter.append(1 - LPprob)
            self.PanyCenter.append(1 - Pprob)

    def myUpdate(self, center, q):
        plrs = q[0]
        blrs = q[1]
        for p in plrs:
            center.pathogenicLRs[p].append(plrs[p][0])
        for b in blrs:
            center.benignLRs[b].append(blrs[b][0])
        for p in plrs:
            center.pathogenicLRPs[p].append(calculateSumOfLogs(center.pathogenicLRs[p]))
        for b in blrs:
            center.benignLRPs[b].append(calculateSumOfLogs(center.benignLRs[b]))


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
    def __init__(self, name, initialSize, testsPerYear, numVariants):
        self.name = name
        self.initialSize = initialSize
        self.testsPerYear = testsPerYear
        self.numVariants = numVariants
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


    def runSimulation(self, simulation, numTests, numThreads, threadID, q, rng):
    #def runSimulation(self, simulation, numTests):
        # TODO: this is where we can add parallelism
        # given the number of threads, divide the number of variants (self.numVariants) by the number of threads
        # that's how many variants each thread will run simulation for
        # put the steps to append to LRs and LRPs outside this loop in an "update()" call?

        numVariants = divide(self.numVariants, numThreads)
        start, end = getStartAndEnd(numVariants, threadID)
        pathogenicLRs = dict()
        benignLRs = dict()
        #pathogenicLRPs = dict()
        #benignLRPs = dict()

        #for variant in range(self.numVariants):
        for variant in range(start, end):
            pathogenicLRs[variant] = list()
            benignLRs[variant] = list()
            #pathogenicLRPs[variant] = list()
            #benignLRPs[variant] = list()

            # generate observations of variant (assumed to be pathogenic) from people with variant
            pathogenicObservations = self.generatePathogenicObservationsFromTests(simulation.p,
                            simulation.P, simulation.B, numTests)

            # generate observations of variant (assumed to be benign) from people with variant
            benignObservations = self.generateBenignObservationsFromTests(simulation.b, simulation.P,
                                                                    simulation.B, numTests)

            # use Poisson distribution to get number of people from this batch with that variant
            numPeopleWithVariant = sampleNumberOfPeopleWithVariant(numTests, simulation.frequency, rng)

            # use PSF to calculate expected number of benign/pathogenic observations for people with variant
            numExpectedBenign, numExpectedPathogenic = getExpectedNumsFromPSF(numPeopleWithVariant, simulation.PSF)

            # generate evidence for observations assumed pathogenic
            pathogenicLRs[variant].append(sampleEvidenceFromObservations(numExpectedPathogenic, pathogenicObservations, rng))

            # generate evidence for observations assumed benign
            benignLRs[variant].append(sampleEvidenceFromObservations(numExpectedBenign, benignObservations, rng))

            # JC I put the steps to update the benignLRPs and pathogenicLRPs in the myUpdate() call b/c those calls
            # need ALL of the LRs (current and previous years), not just the current year which is what is available
            # here
            # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
            #self.benignLRPs[variant].append(calculateSumOfLogs(self.benignLRs[variant]))

            # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
            #self.pathogenicLRPs[variant].append(calculateSumOfLogs(self.pathogenicLRs[variant]))

        q.put([pathogenicLRs, benignLRs])

    def generatePathogenicObservationsFromTests(self, c, P, B, n):
        Obs = \
            [rep(P['PM'], int(c['p2_PM6'] * n)) + rep(B['BP'], int(c['p4_BP2'] * n)) +
             rep(B['BP'], int(c['p5_BP5'] * n)) + rep(P['PP'], int(c['p6_PP1'] * n)) +
             rep(P['PS'], int(c['p7_PS2'] * n)) + rep(B['BS'], int(c['p8_BS4'] * n)) +
             rep(1.0, int((1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] +
                                c['p7_PS2'] + c['p8_BS4'])) * n))]
        return Obs[0]

    def generateBenignObservationsFromTests(self, c, P, B, n):
        Obs = \
            [rep(P['PM'], int(c['b2_PM6'] * n)) + rep(B['BP'], int(c['b4_BP2'] * n)) +
             rep(B['BP'], int(c['b5_BP5'] * n)) + rep(P['PP'], int(c['b6_PP1'] * n)) +
             rep(P['PS'], int(c['b7_PS2'] * n)) + rep(B['BS'], int(c['b8_BS4'] * n)) +
             rep(1.0, int((1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5']  + c['b6_PP1'] +
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

            self.benignProbabilities.append(float(numBenignClassified) / float(self.numVariants))
            self.pathogenicProbabilities.append(float(numPathogenicClassified) / float(self.numVariants))

            numLPClassified = 0
            numLBClassified = 0

            for variant in range(self.numVariants):
                for lrp in pathogenic_y[variant]:
                    if lrp > LP and lrp <= P:
                        numLPClassified += 1
                        break
                for lrp in benign_y[variant]:
                    if lrp < LB and lrp >= B:
                        numLBClassified += 1
                        break

            self.likelyBenignProbabilities.append(float(numLBClassified) / float(self.numVariants))
            self.likelyPathogenicProbabilities.append(float(numLPClassified) / float(self.numVariants))

    def getYearNProbabilities(self, n):
        lbYearN = self.likelyBenignProbabilities[n]
        bYearN = self.benignProbabilities[n]
        lpYearN = self.likelyPathogenicProbabilities[n]
        pYearN = self.likelyBenignProbabilities[n]
        return lbYearN, bYearN, lpYearN, pYearN

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
        if center.numVariants <= 10:
            plt.plot(x, pathogenic_y[variant], color='orange', label='pathogenic')#, alpha=(1.0+variant)/(3+variant))
            plt.plot(x, benign_y[variant], color='blue', label='benign')#, alpha=(1.0+variant)/(3+variant))
        elif center.numVariants <= 100 and variant % 10 == 0:
            plt.plot(x, pathogenic_y[variant], color='orange', label='pathogenic')#, alpha=(1.0+variant)/(3+variant))
            plt.plot(x, benign_y[variant], color='blue', label='benign')#, alpha=(1.0+variant)/(3+variant))
        elif center.numVariants <= 1000 and variant % 100 == 0:
            plt.plot(x, pathogenic_y[variant], color='orange', label='pathogenic')  # , alpha=(1.0+variant)/(3+variant))
            plt.plot(x, benign_y[variant], color='blue', label='benign')  # , alpha=(1.0+variant)/(3+variant))
        else:
            if variant % 1000 == 0:
                plt.plot(x, pathogenic_y[variant], color='orange', label='pathogenic')#, alpha=(1.0+variant)/(3+variant))
                plt.plot(x, benign_y[variant], color='blue', label='benign')#, alpha=(1.0+variant)/(3+variant))

    plt.ylabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.xlabel('year', fontsize=18)
    #plt.title(centerName)

    benignLabel = mpatches.Patch(color='blue', label='benign')
    pathogenicLabel = mpatches.Patch(color='orange', label='pathogenic')
    #plt.legend(handles=[benignLabel, pathogenicLabel], loc='lower left')

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    #plt.show()

    plt.savefig(outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' +
                centerName + '_' + str(year) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_lrp_scatter', dpi=300)
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
    upperLimit = 20
    plt.xlim(lowerLimit, upperLimit)
    bins = numpy.arange(lowerLimit, upperLimit, 0.5)
    plt.ylim(0, 1)


    plt.axvline(x=simulation.thresholds[0], color='green', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[1], color='blue', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[2], color='black', linestyle='dashed', linewidth=1.0)
    plt.axvline(x=simulation.thresholds[3], color='orange', linestyle='dashed', linewidth=0.75)
    plt.axvline(x=simulation.thresholds[4], color='red', linestyle='dashed', linewidth=0.75)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], color=['blue', 'orange'], density=True,
             range=(-15, 50), bins=bins)

    plt.xlabel('evidence = ' + r'$\sum_{i} log(LR_i)$', fontsize=18)
    plt.ylabel('probability mass', fontsize=18)

    #plt.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)
    plt.savefig(outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
                centerName + '_' + str(year) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_lrphist', dpi=300)
    plt.close()

def plotAnyCenterProbability(simulation, outputDir):

    yearList = [i for i in range(0, simulation.years + 1)]
    plt.xlim(0, simulation.years)
    plt.ylim(0, 1)
    plt.plot(yearList, simulation.PanyCenter, marker='.', color='red', label='pathogenic')
    plt.plot(yearList, simulation.BanyCenter, marker='.', color='green', label='benign')
    plt.plot(yearList, simulation.LPanyCenter, marker='.', color='orange', label=' likely pathogenic', linestyle='dashed')
    plt.plot(yearList, simulation.LBanyCenter, marker='.', color='blue', label=' likely benign', linestyle='dashed')

    plt.ylabel('probability of classification', fontsize=18)
    plt.xlabel('year', fontsize=18)
    #plt.title(center.name)
    #plt.legend(loc='upper left', prop= {'size': 8} )
    #plt.show()

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    plt.savefig(outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
                "inclusion-exclusion" + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs',
                dpi=300)
    plt.close()

def plotProbability(simulation, center, outputDir):

    yearList = [i for i in range(0, simulation.years + 1)]
    plt.xlim(0, simulation.years)
    plt.ylim(0, 1)
    plt.plot(yearList, center.pathogenicProbabilities, marker='.', color='red', label='pathogenic')
    plt.plot(yearList, center.benignProbabilities, marker='.', color='green', label='benign')
    plt.plot(yearList, center.likelyPathogenicProbabilities, marker='.', color='orange', label=' likely pathogenic', linestyle='dashed')
    plt.plot(yearList, center.likelyBenignProbabilities, marker='.', color='blue', label=' likely benign', linestyle='dashed')

    plt.ylabel('probability of classification', fontsize=18)
    plt.xlabel('year', fontsize=18)
    #plt.title(center.name)
    #plt.legend(loc='upper left', prop= {'size': 8} )
    #plt.show()

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    plt.savefig(outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
                center.name + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs',
                dpi=300)
    plt.close()

def saveProbability(simulation, center, outputDir):

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)


    outFile = outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
              center.name + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs.dat'
    with open(outFile, 'wb') as output:
        pickle.dump(center, output, pickle.HIGHEST_PROTOCOL)

def runAnalysis(types, parameters, config, outputDir):
    allLRPs = dict()
    for t in types:
        allLRPs[t] = dict()
        for p in parameters:
            mySimulation = Simulation(config=config.data, saType=t, saParam=p)
            mySimulation.run()
            # mySimulation.scatter(outputDir=outputDir)
            # mySimulation.hist(outputDir=outputDir)
            mySimulation.prob(outputDir=outputDir)
            # mySimulation.save(outputDir=outputDir)
            allLRPs[t][p] = mySimulation.allCenters.getYearNProbabilities(mySimulation.years)
    return allLRPs

def saveAllLRPs(types, parameters, allLRPs, outputDir):
    fileName = outputDir + '/allLRPs.csv'
    indices = {0:'LB', 1:'B', 2: 'LP', 3:'P'}
    with open(fileName, 'w') as f:
        print('parameters: ', end=' ', file=f)
        for p in parameters:
            print(p, end=',', file=f)
        for t in types:
            print(t, file=f)
            for i in indices:
                print(t + '_' + indices[i] + ': ', end=',', file=f)
                for p in parameters:
                    print(allLRPs[t][p][i], end=' ', flush=True, file=f)
                print(file=f)
            print(file=f)
    f.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputDir', help='output directory for PNG plots', required=True)
    parser.add_argument('-c', '--confFile', help='path to JSON configuration file', required=True)
    parser.add_argument('-j', '--jobType', help='job type to execute: "simulate" or "analyze"', required=True)
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
    #parameters = ["b7_PS2", "p2_PM6"]

    if jobType == 'simulate':
        print('simulate this!')
        mySimulation = Simulation(config=config.data, saType='med', saParam=None)
        mySimulation.run()
        mySimulation.scatter(outputDir=outputDir)
        mySimulation.hist(outputDir=outputDir)
        mySimulation.prob(outputDir=outputDir)
        plotAnyCenterProbability(mySimulation, outputDir)


    elif jobType == 'analyze':
        print('analyze this!')
        allLRPs = runAnalysis(types, parameters, config, outputDir)
        saveAllLRPs(types, parameters, allLRPs, outputDir)
    else:
        print('whats this?: ' + str(jobType))

if __name__ == "__main__":
    main()
