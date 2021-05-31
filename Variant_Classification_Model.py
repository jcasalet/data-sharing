import numpy
import json
import logging
import sys
import os
from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import utils
import plot

logger = logging.getLogger()
defaultLogLevel = "INFO"
logger.setLevel('INFO')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.debug("Established logger")



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
                    self.updateLRsAndLRPs(center, q.get())
                for i in range(self.numThreads):
                    processList[i].join()
                self.combineAllLRsFromCenter(center, 0)
        self.calculateAllLRPs()

    def calculateAllLRPs(self):
        for variant in range(self.numVariants):
            # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
            self.allCenters.benignLRPs[variant].append(utils.calculateSumOfLogs(self.allCenters.benignLRs[variant]))
            # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
            self.allCenters.pathogenicLRPs[variant].append(utils.calculateSumOfLogs(self.allCenters.pathogenicLRs[variant]))

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
                    q = Queue()
                    processList = list()
                    for i in range(self.numThreads):
                        rng = numpy.random.default_rng(self.seed + (i+1)*(year+1)*(centers.index(center) + 1))
                        p = Process(target=center.runSimulation, args=(self, center.testsPerYear, self.numThreads, i,q,
                                                                       rng))
                        p.start()
                        processList.append(p)
                    for i in range(self.numThreads):
                        self.updateLRsAndLRPs(center, q.get())
                    for i in range(self.numThreads):
                        processList[i].join()
                    self.combineAllLRsFromCenter(center, year)
            self.calculateAllLRPs()
        # after all the data is generated, calculate the probability of classification for each center
        self.pathogenicVariantClassifications = dict()
        self.benignVariantClassifications = dict()
        for centers in self.centerListList:
            for center in centers:
                center.probabilityOfClassification(self)
        self.allCenters.probabilityOfClassification(self)


        # TODO now that each center has a probability of classification for that year, calculate the
        # probability of classification of any of the centers (i.e. P(A or B or C or ...))
        # use https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle#In_probability
        self.inclusionExclusionProbability()
        self.unionProbability()
        self.probabilityOfClassification()


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
        self.probabilityUnions = []
        for year in range(self.years + 1):
            self.probabilityUnions.append(self.calculateProbabilityOfUnion(year, subsetByLenDict))


    def calculateProbabilityOfUnion(self, year, subsetByLenDict):
        # subsetByLenDict = {'1': [(a,), (b,), (c,)], '2': [(a, b), (a, c), (b, c)], '3': [(a, b, c)]}
        # output:
        #         # {'1':    {'LB': P(LB(a)) + P(LB(b) + P(LB(c)),
        #         #           'B': P(B(a)) + P(B(b)) + P(B(c)),
        #         #           'LP': P(LP(a)) + P(LP(b)) + P(LP(c))
        #         #           'P': P(P(a)) + P(P(b)) + P(P(c))}
        #         # '2':      {'LB': P(LB(a))*P(LB(b)) + P(LB(a))*P(LB(c)) + P(LB(b))*P(LB(c)),
        #         #           'B': P(B(a))*P(B(b)) + P(B(a))*P(B(c)) + P(B(b))*P(B(c)),
        #         #           'LP': P(LP(a))*P(LP(b)) + P(LP(a))*P(LP(c)) + P(LP(b))*P(LP(c)),
        #         #           'P': P(P(a))*P(P(b)) + P(P(a))*P(P(c)) + P(P(b))*P(P(c))},
        #         #  ...
        #         # '20': },
        #
        #         # {'size': {LB: sum-prod-probs, B: sum-prod-probs, LP: sum-prod-probs, P: sum-prod-probs}
        probabilities = dict()
        for length in subsetByLenDict:
            probabilities[length] = {'LB': 0, 'B': 0, 'LP': 0, 'P': 0}
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

        probabilityUnion = {'LB': 0, 'B': 0, 'LP': 0, 'P': 0}
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

            self.LBanyCenter.append(1 - LBprob)
            self.BanyCenter.append(1 - Bprob)
            self.LPanyCenter.append(1 - LPprob)
            self.PanyCenter.append(1 - Pprob)

    def updateLRsAndLRPs(self, center, q):
        plrs = q[0]
        blrs = q[1]
        for p in plrs:
            center.pathogenicLRs[p].append(plrs[p][0])
        for b in blrs:
            center.benignLRs[b].append(blrs[b][0])

        # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
        for p in plrs:
            center.pathogenicLRPs[p].append(utils.calculateSumOfLogs(center.pathogenicLRs[p]))
        # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
        for b in blrs:
            center.benignLRPs[b].append(utils.calculateSumOfLogs(center.benignLRs[b]))


    def scatter(self, outputDir):
        for year in [self.years]:
            for centers in self.centerListList:
                for center in centers:
                    plot.plotLRPScatter(self, center, year,  outputDir)
            plot.plotLRPScatter(self, self.allCenters, year, outputDir)

    def hist(self, outputDir):
        for year in [self.years]:
            for centers in self.centerListList:
                for center in centers:
                    plot.plotLRPHist(self, center, year, outputDir)
            plot.plotLRPHist(self, self.allCenters, year, outputDir)

    def prob(self, outputDir):
        for centers in self.centerListList:
            for center in centers:
                plot.plotProbability(self, center, outputDir)
        plot.plotProbability(self, self.allCenters,  outputDir)

    def save(self, outputDir):
        for centers in self.centerListList:
            for center in centers:
                utils.saveProbability(self, center, outputDir)
        utils.saveProbability(self, self.allCenters,  outputDir)

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
    # we run the simulation for a single variant assuming it's benign and assuming it's pathogenic.
    # so for every variant, we are running 2 experiments in parallel.

        # divide up the total number of variants in experiment evenly across the number of threads
        numVariantsPerThread = utils.divide(self.numVariants, numThreads)
        start, end = utils.getStartAndEnd(numVariantsPerThread, threadID)

        # keep track of likelihood ratios for each variant in local variable
        pLRs = dict()
        bLRs = dict()

        # loop thru all the variants assigned to this thread
        for variant in range(start, end):
            # each variant will get a list of LRs assigned to it as evidence
            pLRs[variant] = list()
            bLRs[variant] = list()

            # generate pool of observations of variant (assumed pathogenic)
            pathogenicObservations = self.generatePathogenicObservationsFromTests(simulation.p,
                            simulation.P, simulation.B, numTests)

            # generate pool of observations of variant (assumed benign)
            benignObservations = self.generateBenignObservationsFromTests(simulation.b, simulation.P,
                                                                    simulation.B, numTests)

            # use Poisson distribution to get number of people with that variant in this batch of tests
            numPeopleWithVariant = utils.sampleNumberOfPeopleWithVariant(numTests, simulation.frequency, rng)

            # use PSF to calculate expected number of benign/pathogenic observations for people with variant
            numExpectedBenign, numExpectedPathogenic = utils.getExpectedNumsFromPSF(numPeopleWithVariant, simulation.PSF)

            # generate evidence for observations assumed pathogenic
            pLRs[variant].append(utils.sampleEvidenceFromObservations(numExpectedPathogenic, pathogenicObservations, rng))

            # generate evidence for observations assumed benign
            bLRs[variant].append(utils.sampleEvidenceFromObservations(numExpectedBenign, benignObservations, rng))

            # JC I put the steps to update the benignLRPs and pathogenicLRPs in the updateLRsAndLRPs() call b/c those calls
            # need ALL of the LRs (current and previous years), not just the current year which is what is available
            # here


        q.put([pLRs, bLRs])

    def generatePathogenicObservationsFromTests(self, c, P, B, n):
        Obs = \
            [utils.rep(P['PM'], int(c['p2_PM6'] * n)) + utils.rep(B['BP'], int(c['p4_BP2'] * n)) +
             utils.rep(B['BP'], int(c['p5_BP5'] * n)) + utils.rep(P['PP'], int(c['p6_PP1'] * n)) +
             utils.rep(P['PS'], int(c['p7_PS2'] * n)) + utils.rep(B['BS'], int(c['p8_BS4'] * n)) +
             utils.rep(1.0, int((1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] +
                                c['p7_PS2'] + c['p8_BS4'])) * n))]
        return Obs[0]

    def generateBenignObservationsFromTests(self, c, P, B, n):
        Obs = \
            [utils.rep(P['PM'], int(c['b2_PM6'] * n)) + utils.rep(B['BP'], int(c['b4_BP2'] * n)) +
             utils.rep(B['BP'], int(c['b5_BP5'] * n)) + utils.rep(P['PP'], int(c['b6_PP1'] * n)) +
             utils.rep(P['PS'], int(c['b7_PS2'] * n)) + utils.rep(B['BS'], int(c['b8_BS4'] * n)) +
             utils.rep(1.0, int((1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5']  + c['b6_PP1'] +
                                c['b7_PS2'] + c['b8_BS4'])) * n))]
        return Obs[0]

    def probabilityOfClassification(self, simulation):
        LB = simulation.thresholds[0]
        B = simulation.thresholds[1]
        neutral = simulation.thresholds[2]
        LP = simulation.thresholds[3]
        P = simulation.thresholds[4]


        for year in range(simulation.years):
            simulation.pathogenicVariantClassifications[year] = defaultdict()
            simulation.benignVariantClassifications[year] = defaultdict()
            pathogenic_y = list()
            benign_y = list()
            for variant in range(self.numVariants):
                pathogenic_y.append(list())
                pathogenic_y[variant].append(0)
                pathogenic_y[variant] += self.pathogenicLRPs[variant][year:year+1]
                benign_y.append(list())
                benign_y[variant].append(0)
                benign_y[variant] += self.benignLRPs[variant][year:year+1]

            numPClassified = 0
            numBClassified = 0
            numLPClassified = 0
            numLBClassified = 0

            for variant in range(self.numVariants):
                for lrp in pathogenic_y[variant]:
                    var = str(variant)
                    if lrp > P:
                        numPClassified += 1
                        simulation.pathogenicVariantClassifications[year][variant] = 'P'
                        break
                    elif lrp > LP and lrp <= P:
                        numLPClassified += 1
                        simulation.pathogenicVariantClassifications[year][variant] = 'LP'
                        break
                for lrp in benign_y[variant]:
                    if lrp < B:
                        numBClassified += 1
                        simulation.benignVariantClassifications[year][variant] = 'B'
                        break
                    elif lrp < LB and lrp >= B:
                        numLBClassified +=1
                        simulation.benignVariantClassifications[year][variant] = 'LB'
                        break
            self.benignProbabilities.append(float(numBClassified) / float(self.numVariants))
            self.pathogenicProbabilities.append(float(numPClassified) / float(self.numVariants))
            self.likelyBenignProbabilities.append(float(numLBClassified) / float(self.numVariants))
            self.likelyPathogenicProbabilities.append(float(numLPClassified) / float(self.numVariants))

    def getYearNProbabilities(self, n):
        lbYearN = self.likelyBenignProbabilities[n]
        bYearN = self.benignProbabilities[n]
        lpYearN = self.likelyPathogenicProbabilities[n]
        pYearN = self.likelyBenignProbabilities[n]
        return lbYearN, bYearN, lpYearN, pYearN


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



def main():
    confFile = utils.parse_args().confFile
    outputDir = utils.parse_args().outputDir
    jobType = utils.parse_args().jobType
    config = Configuration(confFile)
    utils.save_config(confFile, outputDir)
    types = ['low', 'med', 'hi']
    parameters = ["p2_PM6", "p4_BP2", "p5_BP5", "p6_PP1", "p7_PS2", "p8_BS4",
                  "b2_PM6", "b4_BP2", "b5_BP5", "b6_PP1", "b7_PS2", "b8_BS4"]

    if jobType == 'simulate':
        print('simulating!')
        mySimulation = Simulation(config=config.data, saType='med', saParam=None)
        mySimulation.run()
        mySimulation.scatter(outputDir=outputDir)
        mySimulation.hist(outputDir=outputDir)
        mySimulation.prob(outputDir=outputDir)
        plot.plotAnyCenterProbability(mySimulation, outputDir, 'new')
        plot.plotAnyCenterProbability(mySimulation, outputDir, 'correct')


    elif jobType == 'analyze':
        print('analyzing!')
        allLRPs = runAnalysis(types, parameters, config, outputDir)
        utils.saveAllLRPs(types, parameters, allLRPs, outputDir)
    else:
        print('whats this?: ' + str(jobType))

if __name__ == "__main__":
    main()
