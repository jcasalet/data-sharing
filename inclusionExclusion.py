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