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
        x = x * sample(Obs, 1, replace=True)
        ans.append(x)
    return ans

def labsamPoBl(Obs, ObsB, LabD, Labf, labPSF):
    # This function gives the expected LR (likelihood ratios) of each different individual observation obtained for a
    # single hypothetial variant seen at a lab given the number of tests in the lab database = LabD, the frequency of
    # the variant = Labf, and the pathogenic variant sampling bias = PSF

    # run separately for path and benign variants (right now it's combined, selecting at random a variant which may be
    # path or benign ) there are more path variants in the lab, but more benign variants in the wild (maybe adjust PSF
    # to 0.5 or 0.2? but psf needs to be whole number)

    StartLRs = [1.0]
    nVar = sum(rpois(LabD, Labf))
    fractionBenign = 1/(labPSF + 1)
    fractionPathogenic = labPSF/(labPSF + 1)
    if nVar > 0:
        '''VarBen = sample([1] + rep(0, labPSF), 1, replace=True)
        if VarBen == 1:
            StartLRs = sample(ObsB, nVar, replace=True)
        else:
            StartLRs = sample(Obs, nVar, replace=True)'''
        StartLRs_B = sample(ObsB, int(fractionBenign * nVar), replace=True)
        StartLRs_P = sample(Obs, int(fractionPathogenic * nVar), replace=True)
        StartLRs = StartLRs_B + StartLRs_P

    return StartLRs

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

def generateNewObservations(D, p, b, P, B):
    Obs = \
        [rep(P['PM'], int(p[2] * D)) + rep(B['BP'], int(p[4] * D)) + rep(B['BP'], int(p[5] * D)) + rep(P['PP'], int(p[6] * D)) +
         rep(P['PS'], int(p[6] * D)) + rep(B['BS'], int(p[7] * D)) + rep(B['BS'], int(p[8] * D)) +
         rep(1.0, int((1 - (p[2] + p[4] + p[5] + p[6] + p[7] + p[8])) * D))][0]
    # observations for individual benign variant
    ObsB = \
        [rep(P['PM'], int(b[2] * D)) + rep(B['BP'], int(b[4] * D)) + rep(B['BP'], int(b[5] * D)) + rep(P['PP'], int(b[6] * D)) +
         rep(P['PS'], int(b[6] * D)) + rep(B['BS'], int(b[7] * D)) + rep(B['BS'], int(b[8] * D)) +
         rep(1.0, int((1 - (b[2] + b[4] + b[5] + b[6] + b[7] + p[8])) * D))][0]
    return Obs, ObsB

def graphLR(lrList, f, center, years, bins, x_label):
    fig, ax = plt.subplots()
    #ax.hist(lrList, bins)
    x = [math.log(lr) for lr in lrList]
    ax.hist(x)
    ax.set_xlabel(x_label)
    ax.set_ylabel('frequency of variant')
    if not center is None:
        ax.set_title('center = ' + center.name + '| f = ' + str(f) + ' | D = ' + str(center.currentSize) + ' | Dgrow = ' + str(center.testsPerYear) + ' | years = ' + str(years))
    else:
        ax.set_title(
            'center = all '  + '| f = ' + str(f) + ' | years = ' + str(years))
    fig.tight_layout()
    plt.show()

def graphGrowth(center, f):
    y = center.allSizes
    x=[i for i in range(1, len(y) + 1)]
    fig, ax = plt.subplots()
    ax.set_xlabel('year')
    ax.set_ylabel('size of db')
    ax.set_title('center = ' + center.name + '| f = ' + str(f) + ' | D = ' + str(center.currentSize) + ' | Dgrow = ' + str(center.testsPerYear))
    plt.scatter(x=x, y=y)
    plt.show()


def runSimulation(center, p, b, P, B, f, PSF, years ):
    Obs, ObsB = generateNewObservations(center.initialSize, p, b, P, B)
    center.addLRsToCenter(labsamPoBl(Obs, ObsB, center.initialSize, f, PSF))
    lrpList = list()
    for i in range(years):
        # observations for individual pathogenic variant
        Obs_new, ObsB_new = generateNewObservations(center.testsPerYear, p, b, P, B)
        center.addLRsToCenter(labsamPoBl(Obs_new, ObsB_new, center.testsPerYear, f, PSF))

class testCenter:
    def __init__(self, name, initialSize, testsPerYear):
        self.name = name
        self.initialSize = initialSize
        self.testsPerYear = testsPerYear
        self.currentSize = initialSize
        self.allSizes = [self.currentSize]
        self.benignLRs = list()
        self.pathogenicLRs = list()
        self.productOfLRs = 0.0

    def addLRsToCenter(self, lrList):
        for lr in lrList:
            if lr < 1.0:
                self.benignLRs.append(lr)
            elif lr > 1.0:
                self.pathogenicLRs.append(lr)
        self.currentSize += len(list(filter(lambda a: a!= 1, lrList)))

    def calculateProductOfLRs(self):
        # sum the LRs because they are in log10
        productOfBenignLRs = 0.0
        productOfPathogenicLRs = 0.0
        for var in self.benignLRs:
            productOfBenignLRs += math.log(var, 10)
        for var in self.pathogenicLRs:
            productOfPathogenicLRs += math.log(var, 10)
        self.productOfLRs = productOfBenignLRs + productOfPathogenicLRs
        return productOfBenignLRs, productOfPathogenicLRs

def main():
    ### gene specific probablities for laboratory observations of pathogenic variants
    p0 = 0 # placeholder
    p1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive (PM3)
    p2_PM7 = 0.007  # probability case is assumed de novo (PM6)
    p3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance
    p4_BP2 = 0.001  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    p5_BP5 = 0.0001  # probability a case has an alternate molecular explanation
    p6_PP1 = 0.2  # probility of cosegregation supporting pathogenicity
    p7_PS2 = 0.003  # probabilty case is proven de novo (PS2)
    p8_BS4 = 0.0001  # probibility of strong cosegregation against pathogenicity (see b6)
    p = [p0, p1_PM3, p2_PM7, p3_BS2, p4_BP2, p5_BP5, p6_PP1, p7_PS2, p8_BS4]

    ### gene specific probablities for pathogenic observations of benign variats
    b0 = 0 # placeholder
    b1_PM3 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive
    b2_PM6 = 0.007  # probability case is assumed de novo
    b3_BS2 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance (BS2)
    b4_BP2 = 0.008  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    b5_BP5 = 0.07  # probability a case has an alternate molecular explanation (BP5)
    b6_PP1 = 0.01  # probility of cosegregation supporting pathogenicity (have bc and a benign brca1 variant; mom also has bc and inherited that brca1 variant from mom; similarly path variants will get benign evidence)
    b7_PS2 = 0.003  # probabilty case is proven de novo
    b8_BS4 = 0.1  # probibility of strong cosegregation against pathogenicity
    b = [b0, b1_PM3, b2_PM6, b3_BS2, b4_BP2, b5_BP5, b6_PP1, b7_PS2, b8_BS4]

    # straight from sean's paper
    PVS = 350
    PS = 18.7
    PM = 4.3
    PP = 2.08
    P = {'PVS': PVS, 'PS': PS, 'PM': PM, 'PP': PP}
    BS = 1 / 18.7
    BP = 1 / 2.08
    B = {'BS': BS, 'BP': BP}

    years = 20 # how long in future to project
    bins = 10 # number of bins for histogram plot
    # pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    PSF = 2  # how much more likely it is to see a pathogenic variant than a benign variant in the laboratory, roughly equal to likelihood ratio

    # each variant only has 1 piece of evidence
    ## determine number of times a variant seen in initial sample

    UW = testCenter('UW', 15000, 3000)
    ambry = testCenter('ambry', 300000, 60000)
    invitae = testCenter('invitae', 250000, 50000)
    arup = testCenter('arup', 75000, 15000)
    freq = 0.00001
    allbenignLRList = list()
    allpathogenicLRList = list()
    allbenignLRPList = list()
    allpathogenicLRPList = list()
    for center in [UW, ambry, invitae, arup]:
        runSimulation(center, p, b, P, B, freq, PSF, years )
        allbenignLRList += center.benignLRs
        allpathogenicLRList += center.pathogenicLRs
        graphLR(center.benignLRs, freq, center, years, bins, 'benign log likelihood ratio')
        graphLR(center.pathogenicLRs, freq, center, years, bins, 'pathogenic log likelihood ratio')
        '''graphLR(center.benignLRPs, freq, center, years, bins, 'benign combined  log likelihood ratios')
        graphLR(center.pathogenicLRPs, freq, center, years, bins, 'pathogenic combined  log likelihood ratios')
        graphGrowth(center, freq)'''
        print('benign LR: ' + str(center.benignLRs))
        print('path LR: ' + str(center.pathogenicLRs))
        productOfBenignLRs, productOfPathogenicLRs = center.calculateProductOfLRs()
        print('product of benign LR = ' + str(productOfBenignLRs))
        print('product of pathogenic LR = ' + str(productOfPathogenicLRs))
        print('product of all LR = ' + str(center.productOfLRs))

    graphLR(allbenignLRList, freq, None, years, bins, 'benign log likelihood ratio')
    graphLR(allpathogenicLRList, freq, None, years, bins, 'pathogenic log likelihood ratio')

if __name__ == "__main__":
    main()