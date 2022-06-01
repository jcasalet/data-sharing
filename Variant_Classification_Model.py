import numpy
import json
import logging
import sys
import os
from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import utils
import plot
from evidence import Evidence
import math

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
    # this class reads in the JSON config file and stores the JSON data
    def __init__(self, configFileName):
        self.configFileName = configFileName
        if self.configFileName != '' and not os.path.exists(self.configFileName):
            logger.error('config file ' + self.configFileName + ' does not exist!', file=sys.stderr)
            sys.exit(1)
        with open(self.configFileName, 'r') as myFile:
            jsonData = myFile.read()
        self.data = json.loads(jsonData)


class Simulation:
    # this class encapsulates the simulation by reading in the configuration and determining the
    # type of run (either analyze or simulate).  There are small, medium, and large centers as well as the constants
    # from the configuration (e.g number of threads, number of variants, PSF, thresholds, ... etc)
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

        self.constants = config['constants']

        # p represents the ACMG evidence criteria priors for pathogenic variants
        self.p_priors = {'p0': self.constants['p0']['med'], 'p1_M3': self.constants['p1_PM3']['med'],
                         'p2_PM6': self.constants['p2_PM6']['med'], 'p3_BS2': self.constants['p3_BS2']['med'],
                         'p4_BP2': float(eval(self.constants['p4_BP2']['med'])), 'p5_BP5': self.constants['p5_BP5']['med'],
                         'p6_PP1': self.constants['p6_PP1']['med'], 'p7_PS2': self.constants['p7_PS2']['med'],
                         'p8_BS4': self.constants['p8_BS4']['med']}


        # b represents the ACMG evidence criteria priors for benign variants
        self.b_priors = {'b0': self.constants['b0']['med'], 'b1_PM3': self.constants['b1_PM3']['med'],
                         'b2_PM6': self.constants['b2_PM6']['med'], 'b3_BS2': self.constants['b3_BS2']['med'],
                         'b4_BP2': float(eval(self.constants['b4_BP2']['med'])), 'b5_BP5': self.constants['b5_BP5']['med'],
                         'b6_PP1': self.constants['b6_PP1']['med'], 'b7_PS2': self.constants['b7_PS2']['med'],
                         'b8_BS4': self.constants['b8_BS4']['med']}

        # if doing senstivity analysis, override single parameter value specified in saParam to saType
        if self.saParam is None:
            pass
        elif self.saParam.startswith('p'):
            if type(self.constants[self.saParam][self.saType]) is str:
                self.p_priors[self.saParam] = eval(str(self.constants[self.saParam][self.saType]))
            else:
                self.p_priors[self.saParam] = self.constants[self.saParam][self.saType]
        elif self.saParam.startswith('b'):
            if type(self.constants[self.saParam][self.saType]) is str:
                self.b_priors[self.saParam] = eval(str(self.constants[self.saParam][self.saType]))
            else:
                self.b_priors[self.saParam] = self.constants[self.saParam][self.saType]
        else:
            logger.error('unknown saParam: ' + str(self.saParam))
            sys.exit(1)

        # P_bayesian_LRs represents the LRs for pathogenic evidence (strong, moderate, supporting)
        self.P_bayesian_LRs = {'PS': self.constants['PS'], 'PM': self.constants['PM'], 'PP': self.constants['PP']}
        # B_bayesian_LRs represents the LRs for benign evidence (strong, and supporting)
        self.B_bayesian_LRs = {'BS': self.constants['BS'], 'BP': self.constants['BP']}

        # PSF is the pathogenic selection factor --> how much more likely is someone to have a pathogenic variant
        self.PSF = self.constants['PSF']

        # assumptions about the initial sizes of each type of center
        self.smallInitialSize = self.constants['smallInitialSize']
        self.smallTestsPerYear = self.constants['smallTestsPerYear']
        self.mediumInitialSize = self.constants['mediumInitialSize']
        self.mediumTestsPerYear = self.constants['mediumTestsPerYear']
        self.largeInitialSize = self.constants['largeInitialSize']
        self.largeTestsPerYear = self.constants['largeTestsPerYear']

        # thresholds from Tavtigian et al
        self.benignThreshold = self.constants['benignThreshold']
        self.likelyBenignThreshold = self.constants['likelyBenignThreshold']
        self.neutralThreshold = self.constants['neutralThreshold']
        self.likelyPathogenicThreshold = self.constants['likelyPathogenicThreshold']
        self.pathogenicThreshold = self.constants['pathogenicThreshold']

        # create dictionary for P and LP classifications pathogenicVariantClassifications[year][variant] = 'P' or 'LP'
        # similar for B and LB
        self.pathogenicVariantClassifications = dict()
        self.benignVariantClassifications = dict()
        for year in range(self.years + 1):
            self.benignVariantClassifications[year] = dict()
            self.pathogenicVariantClassifications[year] = dict()
            for variant in range(self.numVariants):
                self.benignVariantClassifications[year][variant] = ''
                self.pathogenicVariantClassifications[year][variant] = ''

        # construct lists of each type of center
        self.smallCenters = list()
        self.mediumCenters = list()
        self.largeCenters = list()
        # master list of center lists
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
        # intialize all centers object
        self.allCenters = TestCenter(name='all',
                                initialSize=0,
                                testsPerYear=0,
                                numVariants=self.numVariants)
        # run simulation on all centers for set of variants based on initial size
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
                    self.mergeDataFromThread(center, q.get())
                for i in range(self.numThreads):
                    processList[i].join()
                # combine LRs from each center into all centers object
                self.combineAllLRsFromCenter(center, 0)
        # calculate LRPs for all centers object
        self.calculateAllLRPs()

    def calculateAllLRPs(self):
        # combining LRs for variant amounts to taking sum of logs (log(prods) = sum(logs))
        for variant in range(self.numVariants):
            # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
            self.allCenters.benignLRPs[variant].append(utils.calculateSumOfLogs(self.allCenters.benignLRs[variant]))
            self.allCenters.benignEvidenceFreqPs[variant].append(utils.calculateSumOfLogs(self.allCenters.benignEvidenceFreqs[variant]))
            # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
            self.allCenters.pathogenicLRPs[variant].append(utils.calculateSumOfLogs(self.allCenters.pathogenicLRs[variant]))
            self.allCenters.pathogenicEvidenceFreqPs[variant].append(utils.calculateSumOfLogs((self.allCenters.pathogenicEvidenceFreqs[variant])))

    def combineAllLRsFromCenter(self, center, year):
        # just add the sublists for the variant for that year from each center to all centers object
        for variant in range(self.numVariants):
            self.allCenters.pathogenicLRs[variant].append([])
            self.allCenters.pathogenicEvidenceFreqs[variant].append([])

            self.allCenters.pathogenicLRs[variant][year] += center.pathogenicLRs[variant][year]
            self.allCenters.pathogenicEvidenceFreqs[variant][year] += center.pathogenicEvidenceFreqs[variant][year]

            self.allCenters.benignLRs[variant].append([])
            self.allCenters.benignEvidenceFreqs[variant].append([])

            self.allCenters.benignLRs[variant][year] += center.benignLRs[variant][year]
            self.allCenters.benignEvidenceFreqs[variant][year] += center.benignEvidenceFreqs[variant][year]

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
                        self.mergeDataFromThread(center, q.get())
                    for i in range(self.numThreads):
                        processList[i].join()
                    self.combineAllLRsFromCenter(center, year)
            self.calculateAllLRPs()
        # after all the data is generated, calculate the probability of classification for each center
        for centers in self.centerListList:
            for center in centers:
                center.probabilityOfClassification(self)
        self.allCenters.probabilityOfClassification(self)

    def showTheClassifications(self, year, variant):
        print("variant: " + str(variant) + " year: " + str(year))
        for centerList in self.centerListList:
            for center in centerList:
                print("center: " + center.name, end= ",")
                print("P prob: " + str(center.pathogenicProbabilities[year]), end=",")
                print("LP prob: " + str(center.likelyPathogenicProbabilities[year]))
        print("classification: " + self.pathogenicVariantClassifications[year][variant])

    def mergeDataFromThread(self, center, q):
        plrs = q[0]
        pfreqs = q[1]
        blrs = q[2]
        bfreqs = q[3]

        for p in plrs:
            center.pathogenicLRs[p].append(plrs[p][0])
        for p in pfreqs:
            center.pathogenicEvidenceFreqs[p].append(pfreqs[p][0])
        for b in blrs:
            center.benignLRs[b].append(blrs[b][0])
        for b in bfreqs:
            center.benignEvidenceFreqs[b].append(bfreqs[b][0])

        # calculate log(product(LRs)) = sum (log(LRs)) for pathogenic LRs
        for p in plrs:
            center.pathogenicLRPs[p].append(utils.calculateSumOfLogs(center.pathogenicLRs[p]))
        for p in pfreqs:
            center.pathogenicEvidenceFreqPs[p].append(utils.calculateSumOfLogs(center.pathogenicEvidenceFreqs[p]))
        # calculate log(product(LRs)) = sum (log(LRs)) for benign LRs
        for b in blrs:
            center.benignLRPs[b].append(utils.calculateSumOfLogs(center.benignLRs[b]))
        for b in bfreqs:
            center.benignEvidenceFreqPs[b].append(utils.calculateSumOfLogs(center.benignEvidenceFreqs[b]))

    def scatter(self, outputDir):
        for year in [self.years]:
            for centers in self.centerListList:
                for center in centers:
                    plot.plotLRPScatter(self, center, year,  outputDir)
            plot.plotLRPScatter(self, self.allCenters, year, outputDir)

    def hist(self, outputDir, year):
        for centers in self.centerListList:
            for center in centers:
                plot.plotLRPHist(self, center, year, outputDir)
        plot.plotLRPHist(self, self.allCenters, year, outputDir)

    def prob(self, outputDir, centerListList):
        for centers in centerListList:
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
        self.benignEvidenceFreqs = dict()
        self.benignEvidenceFreqPs = dict()
        self.pathogenicLRPs = dict()
        self.pathogenicEvidenceFreqs = dict()
        self.pathogenicEvidenceFreqPs = dict()
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
            self.benignEvidenceFreqs[variant] = list()
            self.pathogenicEvidenceFreqs[variant] = list()
            self.benignEvidenceFreqPs[variant] = list()
            self.pathogenicEvidenceFreqPs[variant] = list()


    def runSimulation(self, simulation, numTests, numThreads, threadID, q, rng):
    # we run the simulation for a single variant assuming it's benign and assuming it's pathogenic.
    # so for every variant, we are running 2 experiments in parallel.

        # divide up the total number of variants in experiment evenly across the number of threads
        numVariantsPerThread = utils.divide(self.numVariants, numThreads)
        start, end = utils.getStartAndEnd(numVariantsPerThread, threadID)

        # keep track of likelihood ratios for each variant in local variable
        pLRs = dict()
        pFreqs = dict()
        bLRs = dict()
        bFreqs = dict()

        # loop thru all the variants assigned to this thread
        for variant in range(start, end):
            # each variant will get a list of LRs assigned to it as evidence
            pLRs[variant] = list()
            pFreqs[variant] = list()
            bLRs[variant] = list()
            bFreqs[variant] = list()

            # generate pool of observations of variant (assumed pathogenic)
            pathogenicObservations = self.generatePathogenicObservationsFromTests(simulation.p_priors,
                            simulation.P_bayesian_LRs, simulation.B_bayesian_LRs, numTests)

            # generate pool of observations of variant (assumed benign)
            benignObservations = self.generateBenignObservationsFromTests(simulation.b_priors,
                            simulation.P_bayesian_LRs, simulation.B_bayesian_LRs, numTests)

            # use Poisson distribution to get number of people with that variant in this batch of tests
            numPeopleWithVariant = utils.sampleNumberOfPeopleWithVariant(numTests, simulation.frequency, rng)

            # use PSF to calculate expected number of benign/pathogenic observations for people with variant
            numExpectedBenign, numExpectedPathogenic = utils.getExpectedNumsFromPSF(numPeopleWithVariant, simulation.PSF)

            # generate evidence for observations assumed pathogenic
            evidenceForPathogenicVariant = utils.sampleEvidenceFromObservations(numExpectedPathogenic, pathogenicObservations, rng)
            sample_p_LRs = list()
            sample_p_Freqs = list()
            for e in evidenceForPathogenicVariant:
                sample_p_LRs.append(e.lr)
                sample_p_Freqs.append(e.freq)
            pLRs[variant].append(sample_p_LRs)
            pFreqs[variant].append(sample_p_Freqs)

            # generate evidence for observations assumed benign
            evidenceForBenignVariant = utils.sampleEvidenceFromObservations(numExpectedBenign, benignObservations, rng)
            sample_b_LRs = list()
            sample_b_Freqs = list()
            for e in evidenceForBenignVariant:
                sample_b_LRs.append(e.lr)
                sample_b_Freqs.append(e.freq)
            bLRs[variant].append(sample_b_LRs)
            bFreqs[variant].append(sample_b_Freqs)

            # JC I put the steps to update the benignLRPs and pathogenicLRPs in the mergeDataFromThread() call b/c those calls
            # need ALL of the LRs (current and previous years), not just the current year which is what is available
            # here


        q.put([pLRs, pFreqs, bLRs, bFreqs])

    def generatePathogenicObservationsFromTests(self, c, P, B, n):
        '''return utils.rep(P['PM'], int(c['p2_PM6'] * n)) + utils.rep(B['BP'], int(c['p4_BP2'] * n)) + \
             utils.rep(B['BP'], int(c['p5_BP5'] * n)) + utils.rep(P['PP'], int(c['p6_PP1'] * n)) + \
             utils.rep(P['PS'], int(c['p7_PS2'] * n)) + utils.rep(B['BS'], int(c['p8_BS4'] * n)) + \
             utils.rep(1.0, int((1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] + \
                                c['p7_PS2'] + c['p8_BS4'])) * n))'''

        Obs =utils.rep(Evidence(P['PM'], c['p2_PM6']), int(c['p2_PM6'] * n)) + \
               utils.rep(Evidence(B['BP'], c['p4_BP2']), int(c['p4_BP2'] * n)) + \
               utils.rep(Evidence(B['BP'], c['p5_BP5']), int(c['p5_BP5'] * n)) + \
               utils.rep(Evidence(P['PP'], c['p6_PP1']), int(c['p6_PP1'] * n)) + \
               utils.rep(Evidence(P['PS'], c['p7_PS2']), int(c['p7_PS2'] * n)) + \
               utils.rep(Evidence(B['BS'], c['p8_BS4']), int(c['p8_BS4'] * n)) + \
               utils.rep(Evidence(1.0, 1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] + \
                        c['p7_PS2'] + c['p8_BS4'])),  int((1 - (c['p2_PM6'] + c['p4_BP2'] + c['p5_BP5'] + c['p6_PP1'] + \
                        c['p7_PS2'] + c['p8_BS4']))* n))

        return Obs

    def generateBenignObservationsFromTests(self, c, P, B, n):
        '''return utils.rep(P['PM'], int(c['b2_PM6'] * n)) + utils.rep(B['BP'], int(c['b4_BP2'] * n)) + \
             utils.rep(B['BP'], int(c['b5_BP5'] * n)) + utils.rep(P['PP'], int(c['b6_PP1'] * n)) + \
             utils.rep(P['PS'], int(c['b7_PS2'] * n)) + utils.rep(B['BS'], int(c['b8_BS4'] * n)) + \
             utils.rep(1.0, int((1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5']  + c['b6_PP1'] + \
                                c['b7_PS2'] + c['b8_BS4'])) * n))'''
        Obs = utils.rep(Evidence(P['PM'], c['b2_PM6']), int(c['b2_PM6'] * n)) + \
               utils.rep(Evidence(B['BP'], c['b4_BP2']), int(c['b4_BP2'] * n)) + \
               utils.rep(Evidence(B['BP'], c['b5_BP5']), int(c['b5_BP5'] * n)) + \
               utils.rep(Evidence(P['PP'], c['b6_PP1']), int(c['b6_PP1'] * n)) + \
               utils.rep(Evidence(P['PS'], c['b7_PS2']), int(c['b7_PS2'] * n)) + \
               utils.rep(Evidence(B['BS'], c['b8_BS4']), int(c['b8_BS4'] * n)) + \
               utils.rep(Evidence(1.0, 1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5'] + c['b6_PP1'] + \
                        c['b7_PS2'] + c['b8_BS4'])), int((1 - (c['b2_PM6'] + c['b4_BP2'] + c['b5_BP5'] + c['b6_PP1'] + \
                        c['b7_PS2'] + c['b8_BS4'])) * n))

        return Obs

    def probabilityOfClassification(self, simulation):
        B = simulation.benignThreshold
        LB = simulation.likelyBenignThreshold
        LP = simulation.likelyPathogenicThreshold
        P = simulation.pathogenicThreshold

        for year in range(1, simulation.years + 1):
            pLRPs = list()
            pFreqPs = list()
            bLRPs = list()
            bFreqPs = list()
            for variant in range(self.numVariants):
                pLRPs.append(list())
                pLRPs[variant].append(0)
                pLRPs[variant] += self.pathogenicLRPs[variant][year:year+1]
                bLRPs.append(list())
                bLRPs[variant].append(0)
                bLRPs[variant] += self.benignLRPs[variant][year:year+1]
                pFreqPs.append(list())
                pFreqPs[variant].append(0)
                pFreqPs[variant] += self.pathogenicEvidenceFreqPs[variant][year:year + 1]
                bFreqPs.append(list())
                bFreqPs[variant].append(0)
                bFreqPs[variant] += self.benignEvidenceFreqPs[variant][year:year + 1]

            numPClassified = 0
            numBClassified = 0
            numLPClassified = 0
            numLBClassified = 0

            for variant in range(self.numVariants):
                for lrp, freqp in zip(pLRPs[variant], pFreqPs[variant]):
                    if lrp > P:
                        numPClassified += 1
                        if self.name != 'all':
                            simulation.pathogenicVariantClassifications[year][variant] = 'P'
                        break
                    elif lrp > LP and lrp <= P:
                        numLPClassified += 1
                        if self.name != 'all' and simulation.pathogenicVariantClassifications[year][variant] != 'P':
                            simulation.pathogenicVariantClassifications[year][variant] = 'LP'
                        break
                for lrp, freqp in zip(bLRPs[variant], bFreqPs[variant]):
                    if lrp < B:
                        numBClassified += 1
                        if self.name != 'all':
                            simulation.benignVariantClassifications[year][variant] = 'B'
                        break
                    elif lrp < LB and lrp >= B:
                        numLBClassified +=1
                        if self.name != 'all' and simulation.benignVariantClassifications[year][variant] != 'B':
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
        pYearN = self.pathogenicProbabilities[n]
        return {'benign': lbYearN + bYearN, 'pathogenic': lpYearN + pYearN}


def runAnalysis(types, parameters, config, outputDir):
    allLRPs = dict()
    for t in types:
        allLRPs[t] = dict()
        for p in parameters:
            mySimulation = Simulation(config=config.data, saType=t, saParam=p)
            mySimulation.run()
            mySimulation.scatter(outputDir=outputDir)
            mySimulation.hist(outputDir=outputDir, year=mySimulation.years)
            mySimulation.prob(outputDir=outputDir, centerListList = [[]])
            # mySimulation.save(outputDir=outputDir)
            allLRPs[t][p] = str(mySimulation.constants[p][t]) + '_' + \
                str(mySimulation.allCenters.getYearNProbabilities(mySimulation.years))
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
        mySimulation.hist(outputDir=outputDir, year=mySimulation.years)
        mySimulation.prob(outputDir=outputDir, centerListList=mySimulation.centerListList)
        plot.plotAnyCenterProbability(mySimulation, outputDir)


    elif jobType == 'analyze':
        print('analyzing!')
        allLRPs = runAnalysis(types, parameters, config, outputDir)
        utils.saveAllLRPs(types, parameters, allLRPs, outputDir)
    else:
        print('whats this?: ' + str(jobType))

if __name__ == "__main__":
    main()
