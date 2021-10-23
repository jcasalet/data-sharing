import numbers
import math
import pickle
from shutil import copyfile
import argparse
from evidence import Evidence

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
    #allLRPs[t][p] = t + '_' + p + '_' + str(mySimulation.constants[p][t]) + '_' + \
                #str(mySimulation.allCenters.getYearNProbabilities(mySimulation.years))
    fileName = outputDir + '/allLRPs.csv'
    indices = ['benign', 'pathogenic']
    with open(fileName, 'w') as f:
        print('parameters: ', end=' ', file=f)
        for p in parameters:
            print(p, end=',', file=f)
        for t in types:
            print(t, file=f)
            for i in indices:
                print(t + '_' + i + ': ', end=',', file=f)
                for p in parameters:
                    #print(str(simulation.constants[p][t]) + '=' + str(allLRPs[t][p][i]), end=' ', flush=True, file=f)
                    print(allLRPs[t][p], end=' ', flush=True, file=f)
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
    if len(theList) == 0:
        return []
    if numSamples > 1:
        return list(rng.choice(a=theList, size=numSamples, replace=replace))
    elif numSamples == 1:
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
    initialEvidence = [Evidence(getRandomUniformPriorLR(rng), 1/len(observations))]
    #initialEvidence = [Evidence(getRandomUniformPriorLR(rng), 0.001)]
    sampleEvidence = sample(observations, int(expectedNum), replace=True, rng=rng)
    # pull out the lrs from the Evidence objects
    '''sampleLRs = list()
    for e in sampleEvidence:
        sampleLRs.append(e.lr)'''
    if len(sampleEvidence) == 0:
        return []
    else:
        return initialEvidence + sampleEvidence
        #return sampleLRs

def getExpectedNumsFromPSF(n, PSF):
    # for now, we've fixed PSF at a number but could in the future make it a distribution from which we sample
    fractionBenign = 1.0 / (PSF +1)
    # numBenign = int(fractionBenign * n)
    numBenign = math.ceil(fractionBenign * n)
    numPathogenic = n - numBenign
    return numBenign, numPathogenic

def sampleNumberOfPeopleWithVariant(n, freq, rng):
    # here we're using the poisson dist b/c the sampling process satisfies:
    # 1. variant occurs randomly
    # 2. variant occurs independently
    # 3. variant counts are discrete (e.g. whole numbers only)
    # P(X=x) = lam^x * e^(-lam) / x! (where lam = mean, lam = variance)
    #return sum(rpois(n, freq, rng))
    return rng.binomial(n, freq)

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

