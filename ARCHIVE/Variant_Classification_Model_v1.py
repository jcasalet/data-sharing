import numpy
import numbers

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
        return list(numpy.random.choice(a=theList, size=numSamples, replace=replace))[0]
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
    StartLRs = 1.0
    nVar = sum(rpois(LabD, Labf))
    if nVar > 0:
        VarBen = sample([1.0] + rep(0, labPSF), 1, replace=True)
        if VarBen == True:
            StartLRs = sample(ObsB, nVar, replace=True)
        else:
            StartLRs = sample(Obs, nVar, replace=True)
    return StartLRs

def labsamProd(Obs, ObsB, LabD, Labf, labPSF):
    ProdLR = 1.0
    nVar = sum(rpois(LabD, Labf))
    if nVar > 0:
        VarBen = sample([1.0] + rep(0, labPSF), 1, replace=True)
        if VarBen == True:
            ProdLR = prod(sample(ObsB, nVar, replace=True))
        else:
            ProdLR = prod(sample(Obs, nVar, replace=True))
    return ProdLR

def addPandB2Lab(Obs, ObsB, lrList):
    for LR in lrList:
        if LR == 1.0:
            continue
        elif LR < 1.0:
            ObsB.append(LR)
        else:
            Obs.append(LR)

def initialPopulation(D, p, b, P, B):
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

def main():
    ### gene specific probablities for laboratory observations of pathogenic variants
    p0 = 0 # placeholder
    p1 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive (PM3)
    p2 = 0.007  # probability case is assumed de novo (PS2)
    p3 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance
    p4 = 0.001  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    p5 = 0.0001  # probability a case has an alternate molecular explanation
    p6 = 0.2  # probility of cosegregation supporting pathogenicity
    p7 = 0.003  # probabilty case is proven de novo (PS2)
    p8 = 0.0001  # probibility of strong cosegregation against pathogenicity
    p = [p0, p1, p2, p3, p4, p5, p6, p7, p8]

    ### gene specific probablities for pathogenic observations of benign variats
    b0 = 0 # placeholder
    b1 = 0  # probability case with a variant has a pathogenic variant in trans - only non-zero for recessive
    b2 = 0.007  # probability case is assumed de novo
    b3 = 0  # probabilty a case is seen in a healthy individual - only informative for very high penetrance (BS2)
    b4 = 0.008  # probability a case is in trans (AD) or in cis (AD and AR) with a pathogenic variant
    b5 = 0.07  # probability a case has an alternate molecular explanation (BP5)
    b6 = 0.01  # probility of cosegregation supporting pathogenicity
    b7 = 0.003  # probabilty case is proven de novo
    b8 = 0.1  # probibility of strong cosegregation against pathogenicity
    b = [b0, b1, b2, b3, b4, b5, b6, b7, b8]

    # straight from sean's paper
    PVS = 350
    PS = 18.7
    PM = 4.3
    PP = 2.08
    P = {'PVS': PVS, 'PS': PS, 'PM': PM, 'PP': PP}
    BS = 1 / 18.7
    BP = 1 / 2.08
    B = {'BS': BS, 'BP': BP}

    years = 10
    n = 1000 # this was not initially a parameter -- brian had it fixed at 1000?
    f = 0.001  # frequency of variant in population
    D = 15000  # size of laboatory database (UW 15K, ambry 200K-400K, invitae 200-300K, arup labs 50-100k... # each will have their own)
    Dgrow = 3000  # number of tests run each year (diff b/w centers)
    # pathogenic selection factor, clinicians select patients whom they think have pathogenic variant
    PSF = 2  # how much more likely it is to see a pathogenic variant than a benign variant in the laboratory, roughly equal to likelihood ratio

    # each variant only has 1 piece of evidence
    '''for i in range(50):
        # observations for individual pathogenic variant
        Obs = [rep(PM, int(p2 * D)) + rep(BP, int(p4 * D)) + rep(BP, int(p5 * D)) +
               rep(PP, int(p6 * D)) + rep(PS, int(p6 * D))+ rep(BS, int(p7 * D)) +
               rep(BS, int(p8 * D)) + rep(1.0, int((1 - (p2 + p4 + p5 + p6 + p7 + p8)) * D))][0]
        # observations for individual benign variant
        ObsB = [rep(PM, int(b2 * D)) + rep(BP, int(b4 * D)) + rep(BP, int(b5 * D)) +
                rep(PP, int(b6 * D)) + rep(PS, int(b6 * D)) + rep(BS, int(b7 * D)) +
                rep(BS, int(b8 * D)) + rep(1.0, int((1 - (b2 + b4 + b5 + b6 + b7 + p8)) * D))][0]
        print('samPoBl(Obs) = ' + str(samPoBl(Obs)))
        print('samPoBl(ObsB) = ' + str(samPoBl(ObsB)))'''

        ## determine number of times a variant seen in initial sample

    Obs, ObsB = initialPopulation(D, p, b, P, B)
    for i in range(years):
        Obs += \
        [rep(PM, int(p2 * D)) + rep(BP, int(p4 * D)) + rep(BP, int(p5 * D)) + rep(PP, int(p6 * D)) +
         rep(PS, int(p6 * D)) + rep(BS, int(p7 * D)) + rep(BS, int(p8 * D)) +
         rep(1.0, int((1 - (p2 + p4 + p5 + p6 + p7 + p8)) * D))][0]
        # observations for individual benign variant
        ObsB += \
        [rep(PM, int(b2 * D)) + rep(BP, int(b4 * D)) + rep(BP, int(b5 * D)) + rep(PP, int(b6 * D)) +
         rep(PS, int(b6 * D)) + rep(BS, int(b7 * D)) + rep(BS, int(b8 * D)) +
         rep(1.0, int((1 - (b2 + b4 + b5 + b6 + b7 + p8)) * D))][0]
        lrList = labsamPoBl(Obs, ObsB, D, f, PSF)
        #print('labsamPoBl(Obs, ObsB, D, f, PSF) = ' + str(labsamPoBl(Obs, ObsB, D, f, PSF)))
        D += Dgrow
        addPandB2Lab(Obs, ObsB, lrList)
        print('size of obs = ' + str(len(Obs)))
        print('size of obsB = ' + str(len(ObsB)))
    #print(Obs)
    #print(ObsB)


if __name__ == "__main__":
    main()