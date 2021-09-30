import numpy as np
cimport numpy as np

DTYPE8 = np.int8
DTYPE32 = np.int32
ctypedef np.int8_t DTYPE8_t
ctypedef np.int32_t DTYPE32_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping

cdef inline unsigned int int_pos(int a): return a if a >= 0 else 0


cdef unsigned long sum1d_pos(np.ndarray[DTYPE32_t, ndim=1] a):
    """Sumiranje clanova matrice"""
    cdef size_t i
    cdef unsigned long total = 0
    for i in range(a.size):
        total += a[i]
    return total


cdef long double ispSum(np.ndarray[DTYPE32_t, ndim=1] pocZal,
                        np.ndarray[DTYPE32_t, ndim=1] zavZal):
    """Isporucena Suma u cijelom razdoblju"""
    cdef size_t i
    cdef unsigned long isp = 0
    for i in range(pocZal.size):
        isp += pocZal[i] - zavZal[i]
    return isp


def plotSurf(np.ndarray[DTYPE32_t, ndim=1] potr, unsigned long maxZalD,
             unsigned long maxZalG, unsigned long minZalD,
             unsigned long minNar, unsigned int daniDob, double pzptD,
             double pzptG, unsigned int pltDwnSmp):
    """Plot povrsine svih eksperimenata"""
    # Suma i velicina potraznje
    cdef unsigned long potrSum = sum1d_pos(potr)
    cdef unsigned long potrSize = potr.size

    # Pocetne zalihe, zavrsne zalihe, izdavanje narudzbe i velicina narudzbe
    cdef np.ndarray[DTYPE32_t, ndim=1] pocZal = np.zeros(potrSize,
                                                              dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] zavZal = np.zeros(potrSize,
                                                              dtype=DTYPE32)
    cdef np.ndarray[DTYPE8_t, ndim=1] izdNar = np.zeros(potrSize,
                                                             dtype=DTYPE8)
    cdef np.ndarray[DTYPE32_t, ndim=1] velNar = np.zeros(potrSize,
                                                              dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=1] pocZalArr
    cdef np.ndarray[DTYPE32_t, ndim=1] zavZalArr
    cdef np.ndarray[DTYPE8_t, ndim=1] izdNarArr
    cdef np.ndarray[DTYPE32_t, ndim=1] velNarArr

    # Liste za spremanje podataka za plot
    cdef list maxZalPltArr = []
    cdef list minZalPltArr = []
    cdef list pzptPltArr = []
    cdef list pzptArr = []
    cdef list minZalArr = []
    cdef list maxZalArr = []

    # Iteratori
    cdef unsigned int maxZal, minZal, dan
    cdef unsigned long brojacIter = 0

    # Trenutni PZPT
    cdef long double pzpt

    # Generiranje tocaka za plot
    # Generiranje max razina zaliha
    for maxZal in range(maxZalD, maxZalG + 1):
        print('Maksimalna razina zaliha: %d' % maxZal)

        # Generiranje min razina zaliha
        for minZal in range(minZalD, maxZal):
            if maxZal < (minZal + minNar):
                continue
            pocZal[0] = maxZal

            # Simulacijski eksperiment po danima
            for dan in range(potrSize):
                brojacIter += 1
                if dan > daniDob:
                    pocZal[dan] = zavZal[dan-1] + velNar[dan-daniDob-1]
                elif dan > 0:
                    pocZal[dan] = zavZal[dan-1]
                zavZal[dan] = int_pos(pocZal[dan] - potr[dan])
                if zavZal[dan] <= minZal:
                    izdNar[dan] = 1
                    velNar[dan] = maxZal - zavZal[dan]
                else:
                    izdNar[dan] = 0
                    velNar[dan] = 0
            pzpt = ispSum(pocZal, zavZal) / potrSum
            # print('PZPT: %.10f' % pzpt)

            # Spremanje točaka za plot zadovoljavajućih eksperimenata
            if pzpt >= pzptD and pzpt <= pzptG:
                # Spremanje najpovoljnijeg ekspremienta
                if not pzptArr:
                    pocZalArr = pocZal.copy()
                    zavZalArr = zavZal.copy()
                    izdNarArr = izdNar.copy()
                    velNarArr = velNar.copy()
                # Spremanje ostalih zadovoljavajućih eksperimenata
                pzptArr.append(pzpt)
                minZalArr.append(minZal)
                maxZalArr.append(maxZal)

            # Spremanje točaka za surface plot
            if minZal % pltDwnSmp == 0:
                maxZalPltArr.append(maxZal)
                minZalPltArr.append(minZal)
                pzptPltArr.append(pzpt)

    print('Broj iteracija : {}'.format(brojacIter))

    return (minZalPltArr, maxZalPltArr, pzptPltArr, minZalArr, maxZalArr,
            pzptArr, pocZalArr, zavZalArr, izdNarArr, velNarArr)
