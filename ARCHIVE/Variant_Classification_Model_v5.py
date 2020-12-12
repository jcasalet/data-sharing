import numpy
import numbers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math


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

def getPriorProbability():
    myRand = numpy.random.uniform(0.1, 0.9, 1)
    return myRand / (1 - myRand)

def labsamPoBl(pathogenicObservations, benignObservations, numTests, frequency, PSF):
    # This function gives the expected LR (likelihood ratios) of each different individual observation obtained for a
    # single hypothetial variant seen at a lab given the number of tests in the lab database = numObservations, the frequency of
    # the variant = frequency, and the pathogenic variant sampling bias = PSF

    # run separately for path and benign variants (right now it's combined, selecting at random a variant which may be
    # path or benign ) there are more path variants in the lab, but more benign variants in the wild (maybe adjust PSF
    # to 0.5 or 0.2? but psf needs to be whole number)

    benign_LRs = [getPriorProbability()]
    pathogenic_LRs = [getPriorProbability()]
    nVar = sum(rpois(numTests, frequency))
    fractionBenign = 1/(PSF + 1)
    fractionPathogenic = 1 - fractionBenign
    benign_LRs += sample(benignObservations, int(fractionBenign * nVar), replace=True)
    pathogenic_LRs += sample(pathogenicObservations, int(fractionPathogenic * nVar), replace=True)
    return benign_LRs, pathogenic_LRs


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
        pathogenicObservations, benignObservations = self.generateNewObservations(num, p, b, P, B)
        benignLRs, pathogenicLRs = labsamPoBl(pathogenicObservations, benignObservations, num, f, PSF)
        print('new benign LRs: ' + str(benignLRs))
        print('new pathogenic LRs: ' + str(pathogenicLRs))
        self.addBenignLRList(benignLRs)
        self.addPathogenicLRList(pathogenicLRs)
        print('all benign LRs to date: ' + str(self.benignLRs))
        print('all path LRs to date: ' + str(self.pathogenicLRs))

    def generateNewObservations(self, n, p, b, P, B):
        # observations for individual pathogenic variant
        Obs = \
            [rep(P['PM'], int(p[2] * n)) + rep(B['BP'], int(p[4] * n)) + rep(B['BP'], int(p[5] * n)) + rep(P['PP'], int(
                p[6] * n)) + rep(P['PS'], int(p[6] * n)) + rep(B['BS'], int(p[7] * n)) + rep(B['BS'], int(p[8] * n)) +
                rep(1.0, int((1 - (p[2] + p[4] + p[5] + p[6] + p[7] + p[8])) * n))][0]
        self.pathogenicObservations.append(Obs)

        # observations for individual benign variant
        ObsB = \
            [rep(P['PM'], int(b[2] * n)) + rep(B['BP'], int(b[4] * n)) + rep(B['BP'], int(b[5] * n)) + rep(P['PP'], int(
                b[6] * n)) + rep(P['PS'], int(b[6] * n)) + rep(B['BS'], int(b[7] * n)) + rep(B['BS'], int(b[8] * n)) +
                rep(1.0, int((1 - (b[2] + b[4] + b[5] + b[6] + b[7] + p[8])) * n))][0]
        self.benignObservations.append(ObsB)
        return Obs, ObsB

    def addBenignLRList(self, lrList):
        self.benignLRs.append(lrList)

    def addPathogenicLRList(self, lrList):
        self.pathogenicLRs.append(lrList)

    def getNumberOfObservations(self):
        size = 0
        for l in self.pathogenicObservations:
            size += len(l)
        for l in self.benignObservations:
            size += len(l)
        return size

    def getNumberOfVariants(self):
        size = 0
        for l in self.pathogenicLRs:
            size += len(l)
        for l in self.benignLRs:
            size+= len(l)
        return size


def graphAllLR(centerList, f, years, thresholds, bins):
    currentSize = 0
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
        currentSize += center.getNumberOfVariants()
        testsPerYear += center.testsPerYear
    ax = plt.figure(figsize=(8,6)).gca()
    if f <= 1e-6:
        plt.xlim(-4, 4)
        plt.ylim(0, 10)
    elif f <= 1e-5:
        plt.xlim(-4, 4)
        plt.ylim(0, 16)
    elif f <= 1e-4:
        plt.xlim(-6, 6)
        plt.ylim(0, 30)
    elif f <= 1e-3:
        plt.xlim(-15, 40)
        plt.ylim(0,20)
    else:
        plt.xlim(-20, 20)
        plt.ylim(0, 40)
    plt.axvline(x=thresholds[0], color='green', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[1], color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[2], color='black', linestyle='solid', linewidth=1)
    plt.axvline(x=thresholds[3], color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[4], color='red', linestyle='dashed', linewidth=1)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], bins=bins)
    plt.xlabel('product of LLR')
    plt.ylabel('frequency of variant')
    centerNames = [x.name for x in centerList]
    plt.title('center = ' + str(centerNames) + '| freq = ' + str(f) + ' | size = ' + str(currentSize) +
                ' | tests = ' + str(testsPerYear) + ' | year = ' + str(years))
    plt.legend(loc='upper right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    centers = ''
    for center in centerNames:
        centers = centers + center + '_'

    #plt.savefig('/Users/jcasaletto/Desktop/RESEARCH/BRIAN/MODEL/' + centers + '_' + str(years))

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
    bins = 10 # number of bins for histogram plot
    PSF = 2  #pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    freq = 1e-5 # this is the frequency of the variant we are interested in

    '''UW = testCenter('UW', 15000, 3000)
    ambry = testCenter('ambry', 300000, 60000)
    invitae = testCenter('invitae', 250000, 50000)
    arup = testCenter('arup', 75000, 15000)'''
    UW = testCenter('UW', 15000, 3000)
    ambry = testCenter('ambry', 1000000, 450000)
    invitae = testCenter('invitae', 10000000, 450000)
    arup = testCenter('arup', 150000, 30000)
    centerList = [UW, ambry, invitae, arup]
    thresholds = [math.log(0.001,10), math.log(1/18.07, 10), 0, math.log(18.07, 10), math.log(100, 10)]

    numVariants = 0
    numObservations = 0
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
            numVariants += center.getNumberOfVariants()
            numObservations += center.getNumberOfObservations()
            if year in yearsOfInterest:
                graphAllLR([center], freq, year, thresholds, bins)
        if year in yearsOfInterest:
            graphAllLR(centerList, freq, year, thresholds, bins)

    print('num variants = ' + str(numVariants))
    print('num observations = ' + str(numObservations))
    print('numVars/numObs = ' + str(numVariants/numObservations))

if __name__ == "__main__":
    main()