import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy

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

def plotAnyCenterProbability(simulation, outputDir, version):

    yearList = [i for i in range(0, simulation.years + 1)]
    plt.xlim(0, simulation.years)
    plt.ylim(0, 1)
    if version == 'new':
        PanyCenter = []
        BanyCenter = []
        LPanyCenter = []
        LBanyCenter = []
        for year in yearList:
            PanyCenter.append(simulation.probabilityUnions[year]['P'])
            BanyCenter.append(simulation.probabilityUnions[year]['B'])
            LPanyCenter.append(simulation.probabilityUnions[year]['LP'])
            LBanyCenter.append(simulation.probabilityUnions[year]['LB'])
    elif version == 'old':
        PanyCenter = simulation.PanyCenter
        LPanyCenter = simulation.LPanyCenter
        BanyCenter = simulation.BanyCenter
        LBanyCenter = simulation.LBanyCenter
    else:
        PanyCenter = [0]
        BanyCenter = [0]
        LPanyCenter = [0]
        LBanyCenter = [0]
        for year in range(0, simulation.years):
            pSum = 0
            bSum = 0
            lpSum = 0
            lbSum = 0
            for variant in simulation.benignVariantClassifications[year]:
                if simulation.benignVariantClassifications[year][variant] == 'B':
                    bSum += 1
                elif simulation.benignVariantClassifications[year][variant] == 'LB':
                    lbSum += 1
            BanyCenter.append(float(bSum) / float(simulation.numVariants))
            LBanyCenter.append(float(lbSum / float(simulation.numVariants)))
            for variant in simulation.pathogenicVariantClassifications[year]:
                if simulation.pathogenicVariantClassifications[year][variant] == 'P':
                    pSum += 1
                elif simulation.pathogenicVariantClassifications[year][variant] == 'LP':
                    lpSum += 1
            PanyCenter.append(float(pSum) / float(simulation.numVariants))
            LPanyCenter.append(float(lpSum / float(simulation.numVariants)))


    plt.plot(yearList, PanyCenter, marker='.', color='red', label='pathogenic')
    plt.plot(yearList, BanyCenter, marker='.', color='green', label='benign')
    plt.plot(yearList, LPanyCenter, marker='.', color='orange', label=' likely pathogenic', linestyle='dashed')
    plt.plot(yearList, LBanyCenter, marker='.', color='blue', label=' likely benign', linestyle='dashed')

    plt.ylabel('probability of classification', fontsize=18)
    plt.xlabel('year', fontsize=18)

    dist = str(simulation.nSmall) + '_' + str(simulation.nMedium) + '_' + str(simulation.nLarge)

    plt.savefig(outputDir + '/' + simulation.saType + '_' + str(simulation.saParam) + '_' + simulation.name + '_' + \
                "any-center" + '_' + version + '_' + str(simulation.years) + 'yrs_' + str(simulation.frequency) + '_' + dist + '_probs',
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
