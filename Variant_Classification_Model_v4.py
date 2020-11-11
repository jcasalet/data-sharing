import numpy
import numbers
import matplotlib.pyplot as plt
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

def findInterval(num, intervalList):
    if len(intervalList) != 2:
        return False
    elif not isinstance(intervalList[0], numbers.Number) or not isinstance(intervalList[1], numbers.Number) or \
            intervalList[0] > intervalList[1]:
        return False
    else:
        return num >= intervalList[0] and num <= intervalList[1]

def samPoBl(Obs):
    # figure of how long it will take to classify variant
    # function to sample until pathogenic or benign
    x=1.0
    ans=[]
    # recall     BS = 1 / 18.7 ~ 0.05 and PS = 18.7 ~ 19
    # below lhs is LB and above rhs is LP --> [0.05, 19]
    # myInterval is for B=0.001 and P=100
    myInterval = [0.001, 100]
    # another way to recode this is stop after having certain number in the interval
    while findInterval(x, myInterval) == True:
        mySample = sample(Obs, 1, replace=True)[0]
        x = x * mySample
        ans.append(x)
    return ans

def getPriorProbability():
    myRand = numpy.random.uniform(0.1, 0.9, 1)
    return myRand / (1 - myRand)

def labsamPoBl(pathogenicObservations, benignObservations, numObservations, frequency, PSF):
    # This function gives the expected LR (likelihood ratios) of each different individual observation obtained for a
    # single hypothetial variant seen at a lab given the number of tests in the lab database = LabD, the frequency of
    # the variant = Labf, and the pathogenic variant sampling bias = PSF

    # run separately for path and benign variants (right now it's combined, selecting at random a variant which may be
    # path or benign ) there are more path variants in the lab, but more benign variants in the wild (maybe adjust PSF
    # to 0.5 or 0.2? but psf needs to be whole number)

    StartLRs_B = [getPriorProbability()]
    StartLRs_P = [getPriorProbability()]
    nVar = sum(rpois(numObservations, frequency))
    fractionBenign = 1/(PSF + 1)
    fractionPathogenic = 1 - fractionBenign
    if nVar > 0:
        StartLRs_B += sample(benignObservations, int(fractionBenign * nVar), replace=True)
        StartLRs_P += sample(pathogenicObservations, int(fractionPathogenic * nVar), replace=True)
    return StartLRs_B, StartLRs_P

def labsamProd(Obs, ObsB, LabD, Labf, labPSF):
    # This function gives the expected *combined LR from all* of the different individual observations obtained from a
    # single hypothetial variant seen at a lab given the number of tests in the lab database = LabD, the frequency of
    # the variant = Labf, and the pathogenic variant sampling bias = PSF

    ProdLR = 1.0
    nVar = sum(rpois(LabD, Labf))
    if nVar > 0:
        VarBen = sample([1.0] + rep(0, labPSF), 1, replace=True)
        if VarBen == True:
            ProdLR = prod(sample(ObsB, nVar, replace=True))
        else:
            ProdLR = prod(sample(Obs, nVar, replace=True))
    return ProdLR


class testCenter:
    def __init__(self, name, initialSize, testsPerYear):
        self.name = name
        self.initialSize = initialSize
        self.testsPerYear = testsPerYear
        self.benignObservations = list()
        self.pathogenicObservations = list()
        self.benignLRs = list()
        self.pathogenicLRs = list()
        self.benignLRPList = list()
        self.pathogenicLRPList = list()

    def runSimulation(self, p, b, P, B, f, PSF, num):
        pathogenicObservations, benignObservations = self.generateNewObservations(num, p, b, P, B)
        benignLRs, pathogenicLRs = labsamPoBl(pathogenicObservations, benignObservations, num, f, PSF)
        self.addBenignLRList(benignLRs)
        self.addPathogenicLRList(pathogenicLRs)

    def generateNewObservations(self, D, p, b, P, B):
        # observations for individual pathogenic variant
        Obs = \
            [rep(P['PM'], int(p[2] * D)) + rep(B['BP'], int(p[4] * D)) + rep(B['BP'], int(p[5] * D)) + rep(P['PP'], int(
                p[6] * D)) + rep(P['PS'], int(p[6] * D)) + rep(B['BS'], int(p[7] * D)) + rep(B['BS'], int(p[8] * D)) +
                rep(1.0, int((1 - (p[2] + p[4] + p[5] + p[6] + p[7] + p[8])) * D))][0]
        self.pathogenicObservations.append(Obs)

        # observations for individual benign variant
        ObsB = \
            [rep(P['PM'], int(b[2] * D)) + rep(B['BP'], int(b[4] * D)) + rep(B['BP'], int(b[5] * D)) + rep(P['PP'], int(
                b[6] * D)) + rep(P['PS'], int(b[6] * D)) + rep(B['BS'], int(b[7] * D)) + rep(B['BS'], int(b[8] * D)) +
                rep(1.0, int((1 - (b[2] + b[4] + b[5] + b[6] + b[7] + p[8])) * D))][0]
        self.benignObservations.append(ObsB)
        return Obs, ObsB

    def addBenignLRList(self, lrList):
        self.benignLRs.append(lrList)

    def addPathogenicLRList(self, lrList):
        self.pathogenicLRs.append(lrList)

    def getBenignObservations(self):
        obs = list()
        for bList in self.benignObservations:
            for bo in bList:
                obs.append(bo)
        return obs

    def getPathogenicObservations(self):
        obs = list()
        for pList in self.pathogenicObservations:
            for po in pList:
                obs.append(po)
        return obs

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

    def getNumberOfBenignVariants(self):
        size = 0
        for l in self.benignLRs:
            size += len(l)
        return size

    def getNumberOfPathogenicVariants(self):
        size = 0
        for l in self.pathogenicLRs:
            size += len(l)
        return size

    def getProductOfBenignLRs(self):
        # sum the LRs because they are in log10

        #product = 0.0
        product = 1.0
        for lrList in self.benignLRs:
            for lr in lrList:
                #product += math.log(lr, 10)
                product *= lr
        return product

    def getProductOfPathogenicLRs(self):
        # sum the LRs because they are in log10
        #product = 0.0
        product = 1.0
        for lrList in self.pathogenicLRs:
            for lr in lrList:
                #product += math.log(lr, 10)
                product *= lr
        return product

    def graphLRP(self, f, years, bins, x_label):
        benign_x = list()
        pathogenic_x = list()
        for lrList in self.benignLRs:
            benign_x += [math.log(lr) for lr in lrList]
        for lrList in self.pathogenicLRs:
            pathogenic_x += [math.log(lr) for lr in lrList]
        plt.figure(figsize=(8,6))
        plt.hist(benign_x, label='benign')
        plt.hist(pathogenic_x, label='pathogenic')
        plt.xlabel(x_label)
        plt.ylabel('frequency of variant')
        plt.title('center = ' + self.name + '| f = ' + str(f) + ' | D = ' + str(self.getNumberOfVariants()) +
                  ' | Dgrow = ' + str(self.testsPerYear) + ' | years = ' + str(years))
        plt.legend(loc = 'upper right')
        plt.show()

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

    plt.figure(figsize=(8, 6))
    if f <= 1e-6:
        plt.xlim(-4, 4)
        plt.ylim(0, 7)
    elif f <= 1e-5:
        plt.xlim(-4, 4)
        plt.ylim(0, 10)
    elif f <= 1e-4:
        plt.xlim(-6, 6)
        plt.ylim(0, 15)
    elif f <= 1e-3:
        plt.xlim(-15, 15)
        plt.ylim(0,20)
    else:
        plt.xlim(-20, 20)
        plt.ylim(0, 40)
    plt.axvline(x=thresholds[0], color='green', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[1], color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[2], color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(x=thresholds[3], color='red', linestyle='dashed', linewidth=1)

    plt.hist([benign_x, pathogenic_x], label=['benign', 'pathogenic'], bins=bins)
    plt.xlabel('product of LLR')
    plt.ylabel('frequency of variant')
    centerNames = [x.name for x in centerList]
    plt.title('center = ' + str(centerNames) + '| f = ' + str(f) + ' | D = ' + str(currentSize) +
                ' | Dgrow = ' + str(testsPerYear) + ' | years = ' + str(years))
    plt.legend(loc='upper right')
    plt.show()

def estimateNumberOfYearsUntilClassification(centerList):
    allBenignObservations = list()
    allPathogenicObservations = list()
    for center in centerList:
        allBenignObservations += center.getBenignObservations()
        allPathogenicObservations += center.getPathogenicObservations()
    benignObservationsBeforeClassification = samPoBl(allBenignObservations)
    pathogenicObservationsBeforeClassification = samPoBl(allPathogenicObservations)

    print('benign before class: ' + str(len(benignObservationsBeforeClassification)))
    print('path before class: ' + str(len(pathogenicObservationsBeforeClassification)))

def main():
    ### gene specific probablities for laboratory observations of pathogenic variants
    p0 = 0 # placeholder
    p1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive (PM3)
    p2_PM6 = 0.007  # probability case is assumed de novo (PM6)
    p3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance
    p4_BP2 = 0.001  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    p5_BP5 = 0.0001  # probability a case has an alternate molecular explanation
    p6_PP1 = 0.2  # probility of cosegregation supporting pathogenicity
    p7_PS2 = 0.003  # probabilty case is proven de novo (PS2)
    p8_BS4 = 0.0001  # probibility of strong cosegregation against pathogenicity (see b6)
    p = [p0, p1_PM3, p2_PM6, p3_BS2, p4_BP2, p5_BP5, p6_PP1, p7_PS2, p8_BS4]

    ### gene specific probablities for pathogenic observations of benign variats
    b0 = 0 # placeholder
    b1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive
    b2_PM6 = 0.007  # probability case is assumed de novo
    b3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance (BS2)
    b4_BP2 = 0.008  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    b5_BP5 = 0.07  # probability a case has an alternate molecular explanation (BP5)
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

    years = 15 # how long in future to project (i.e. number of iterations in simulation)
    bins = 15 # number of bins for histogram plot
    # pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    PSF = 2  # how much more likely to see a pathogenic variant than a benign in  lab, roughly equal to likelihood ratio
    freq = 1e-5 # this is the frequency of the variant we are interested in

    # each variant only has 1 piece of evidence
    ## determine number of times a variant seen in initial sample

    UW = testCenter('UW', 15000, 3000)
    ambry = testCenter('ambry', 300000, 60000)
    invitae = testCenter('invitae', 250000, 50000)
    arup = testCenter('arup', 75000, 15000)
    centerList = [UW, ambry, invitae, arup]
    thresholds = [math.log(0.001,10), math.log(1/18.07, 10), math.log(18.07, 10), math.log(100, 10)]

    numVariants = 0
    numObservations = 0
    # first, populate each center's db with variants based on initial sizes
    for center in centerList:
        center.runSimulation(p, b, P, B, freq, PSF, center.initialSize)

    # second, simulate forward in time, add variants to each center's db based on tests per year
    for year in range(1, years+1):
        graphAllLR(centerList, freq, year, thresholds, bins)
        for center in centerList:
            center.runSimulation(p, b, P, B, freq, PSF, center.testsPerYear)
            numVariants += center.getNumberOfVariants()
            numObservations += center.getNumberOfObservations()


    #estimateNumberOfYearsUntilClassification(centerList)
    #print('num variants = ' + str(numVariants))
    #print('num observations = ' + str(numObservations))
    #print('numVars/numObs = ' + str(numVariants/numObservations))

if __name__ == "__main__":
    main()