import numbers
import math
import pickle
from shutil import copyfile
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputDir', help='output directory for PNG plots', required=True)
    parser.add_argument('-c', '--confFile', help='path to JSON configuration file', required=True)
    parser.add_argument('-j', '--jobType', help='job type to execute: "simulate" or "analyze"', required=True)
    options = parser.parse_args()
    return options

def save_config(confFile, outputDir):
    nLevels = confFile.count('/')
    fileName = confFile.split('/')[nLevels]
    copyfile(confFile, outputDir + '/' + fileName)

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

def saveProbability(simulation, center, outputDir):

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)


    outFile = outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
              center.name + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs.dat'
    with open(outFile, 'wb') as output:
        pickle.dump(center, output, pickle.HIGHEST_PROTOCOL)

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

def unionProbability(centerListList, years):
    # P(A U B U C ... ) = (sum size 1 sets) - (sum size 2 sets) + (sum size 3 sets) - ...
    import itertools
    subsets = list()
    centers = set()
    for cl in centerListList:
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
    for year in range(years + 1):
        probabilityUnions.append(calculateProbabilityOfUnion(year, subsetByLenDict))

    return probabilityUnions

def calculateProbabilityOfUnion(year, subsetByLenDict):
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


def inclusionExclusionProbability(simulation):
    # https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle#In_probability
    # calculate for each year, store in list or dictionary
    LB = list()
    B = list()
    LP = list()
    P = list()
    for center in simulation.largeCenters:
        LB.append(center.likelyBenignProbabilities)
        B.append(center.benignProbabilities)
        LP.append(center.likelyPathogenicProbabilities)
        P.append(center.pathogenicProbabilities)
    for center in simulation.mediumCenters:
        LB.append(center.likelyBenignProbabilities)
        B.append(center.benignProbabilities)
        LP.append(center.likelyPathogenicProbabilities)
        P.append(center.pathogenicProbabilities)
    for center in simulation.smallCenters:
        LB.append(center.likelyBenignProbabilities)
        B.append(center.benignProbabilities)
        LP.append(center.likelyPathogenicProbabilities)
        P.append(center.pathogenicProbabilities)

    simulation.LBanyCenter = []
    simulation.BanyCenter = []
    simulation.LPanyCenter = []
    simulation.PanyCenter = []
    for year in range(simulation.years+1):
        LBprob = 1.0
        Bprob = 1.0
        LPprob = 1.0
        Pprob = 1.0
        for center in range(len(LB)):
            LBprob *= (1.0 - LB[center][year])
            Bprob *= (1.0 - B[center][year])
            LPprob *= (1.0 - LP[center][year])
            Pprob *= (1.0 - P[center][year])

        simulation.LBanyCenter.append(1 - LBprob)
        simulation.BanyCenter.append(1 - Bprob)
        simulation.LPanyCenter.append(1 - LPprob)
        simulation.PanyCenter.append(1 - Pprob)