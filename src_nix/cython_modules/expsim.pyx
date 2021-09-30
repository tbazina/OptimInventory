import numpy as np
cimport numpy as np

DTYPE = np.long
ctypedef np.long_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping

cdef inline long int_pos(long a): return a if a >= 0 else 0


cdef long sum1d_pos(long[:] summand):
    """Sumiranje clanova long matrice"""
    cdef size_t i
    cdef long total = 0
    for i in range(summand.size):
        total += summand[i]
    return total


"""
cdef long sum1d_int(int[:] summand):
    "Sumiranje clanova integer matrice"
    cdef size_t i
    cdef long total = 0
    for i in range(summand.size):
        total += summand[i]
    return total
"""

cdef long double pzptProFun(long[:] daniSync, long potrSize, long [:] pocZal,
                            long[:] zavZal, long[:] velNar, long double potrSum,
                            long daniDob):
    """PZPT po proizvodima"""
    cdef size_t i
    cdef long double sumVelNar = sum1d_pos(velNar)
    cdef long double pzptPro
    for i in range(potrSize-1, 0, -1):
        if velNar[i] != 0:
            break
    if sum1d_pos(daniSync[i+1:potrSize]) <= daniDob:
        sumVelNar -= velNar[i]
    sumVelNar += (pocZal[0] - int_pos(zavZal[potrSize-1]))
    potrSum = <long double>potrSum
    pzptPro = <long double>(<long double>sumVelNar / <long double>potrSum)
    return <long double>pzptPro


cdef long double zadDan(long[:] pocZal, long[:] potr, long potrSize):
    """Broj dana u kojima je zadovoljena potražnja"""
    cdef size_t i
    cdef long brDan = 0
    for i in range(potrSize):
        if pocZal[i] >= potr[i]:
            brDan += 1
    return brDan


cdef long provjPocZal(long maxZal, long pocZalExp):
    """Vraca maksimalnu razinu zaliha eksperimenta ili konstantnu pocetnu
    zalihu"""
    if pocZalExp > 0:
       return pocZalExp
    else:
       return maxZal


cdef void bruteForceSearch(long[:] potr, long pocZalExp, long maxZalD,
                           long maxZalG, long minZalD, long minZalG,
                           long minNar, long daniDob, long radVrDob,
                           long radVrSub, long double backLogPerc,
                           long double pzptDPerc, long double pzptGPerc,
                           long pzptProChk, long pzptDanChk, izlFol,
                           long sprNaj, long potrSum, long potrSize,
                           long[:] daniSync, long brojacZadExp):
    """Brute-Force pretraživanje"""
    # Pocetne zalihe, zavrsne zalihe, izdavanje narudzbe i velicina narudzbe
    # Matrice koje se mijenjaju tijekom eksperimenta
    cdef long[:] pocZal = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] zavZal = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] izdNar = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] velNar = np.zeros(potrSize, dtype=DTYPE)

    # Iteratori i brojači
    cdef long maxZal, minZal, dan, danUnatr, danIzdNar, danNerRad

    # Blokiranje narudžbe
    cdef long blockNar

    # Ukupna veličina narudžbe
    cdef long ukVelNar

    # Backlogging - broj proizvoda i koeficijent
    cdef long backLog
    cdef long double backLogKoef = backLogPerc / 100.

    # Trenutni PZPT
    cdef long double pzptPro
    cdef long double pzptDan

    # PZPT donja i gornja granica - koeficijenti
    cdef long double pzptDDec = pzptDPerc / 100.
    cdef long double pzptGDec = pzptGPerc / 100.

    # Početak eksperimenta
    # Generiranje max razina zaliha
    for maxZal in range(maxZalD, maxZalG + 1):
        print('Brute maksimalna razina zaliha: %d' % maxZal)
        # Provjera: konstantne pocetne zalihe ili jednake maksimumu
        pocZal[0] = provjPocZal(maxZal, pocZalExp)

        # Generiranje min razina zaliha
        for minZal in range(minZalD, maxZal + 1 - minNar):
            if minZal > minZalG:
                break
            # print('Minimalna razina zaliha: {}'.format(minZal))

            # Simulacijski eksperiment po danima
            # Prvi dan
            backLog = 0
            blockNar = 0
            if pocZal[0] >= potr[0]:
                zavZal[0] = pocZal[0] - potr[0]
            else:
                zavZal[0] = 0
                backLog += <long>(backLogKoef * (potr[0] - pocZal[0]))
            if zavZal[0] <= minZal and blockNar == 0:
                izdNar[0] = 1
                velNar[0] = ((maxZal - zavZal[0]) // minNar) * minNar
                blockNar = 1
            else:
                izdNar[0] = 0
                velNar[0] = 0

            # Dani bez dobave
            dan = 1
            while sum1d_pos(daniSync[:dan]) < (daniDob + 1):
                pocZal[dan] = zavZal[dan-1]
                if pocZal[dan] >= potr[dan]:
                    zavZal[dan] = pocZal[dan] - potr[dan]
                else:
                    zavZal[dan] = 0
                    backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
                if zavZal[dan] <= minZal and blockNar == 0:
                    izdNar[dan] = 1
                    velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
                    blockNar = 1
                else:
                    izdNar[dan] = 0
                    velNar[dan] = 0
                dan += 1

            # Prvi dan s mogućom dobavom
            if izdNar[0] == 1:
                if (zavZal[dan-1] + velNar[0]) >= backLog:
                    pocZal[dan] = zavZal[dan-1] + velNar[0] - backLog
                    backLog = 0
                    blockNar = 0
                else:
                    pocZal[dan] = 0
                    backLog -= zavZal[dan-1] + velNar[0]
                    blockNar = 0
            else:
                pocZal[dan] = zavZal[dan-1]
            if pocZal[dan] >= potr[dan]:
                zavZal[dan] = pocZal[dan] - potr[dan]
            else:
                zavZal[dan] = 0
                backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
            if zavZal[dan] <= minZal and blockNar == 0:
                izdNar[dan] = 1
                velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
                blockNar = 1
            else:
                izdNar[dan] = 0
                velNar[dan] = 0
            dan += 1

            # Svi ostali dani
            for dan in range(dan, potrSize):
                # Provjera izdane narudžbe
                if daniSync[dan] == 1:
                    danUnatr = 1
                    while (sum1d_pos(daniSync[dan-daniDob-danUnatr:dan]) <
                           (daniDob + 1)):
                        danUnatr += 1
                    danIzdNar = dan - daniDob - danUnatr
                    if daniSync[danIzdNar-1] == 1:
                        if izdNar[danIzdNar] == 1:
                            if (zavZal[dan-1] + velNar[danIzdNar]) >= backLog:
                                pocZal[dan] = (zavZal[dan-1] +
                                               velNar[danIzdNar] - backLog)
                                backLog = 0
                                blockNar = 0
                            else:
                                pocZal[dan] = 0
                                backLog -= zavZal[dan-1] + velNar[danIzdNar]
                                blockNar = 0
                        else:
                            pocZal[dan] = zavZal[dan-1]
                    else:
                        ukVelNar = sum1d_pos(
                            velNar[danIzdNar-(radVrSub-radVrDob):danIzdNar+1])
                        if ukVelNar > 0:
                            if (zavZal[dan-1] + ukVelNar) >= backLog:
                                pocZal[dan] = (zavZal[dan-1] + ukVelNar
                                               - backLog)
                                backLog = 0
                                blockNar = 0
                            else:
                                pocZal[dan] = 0
                                backLog -= zavZal[dan-1] + ukVelNar
                                blockNar = 0
                        else:
                            pocZal[dan] = zavZal[dan-1]
                else:
                    pocZal[dan] = zavZal[dan-1]
                # Računanje završnih zaliha i backlogginga
                if pocZal[dan] >= potr[dan]:
                    zavZal[dan] = pocZal[dan] - potr[dan]
                else:
                    zavZal[dan] = 0
                    backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
                # Odluka o izdavanu i veličini narudžbe
                if zavZal[dan] <= minZal and blockNar == 0:
                    izdNar[dan] = 1
                    velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
                    blockNar = 1
                else:
                    izdNar[dan] = 0
                    velNar[dan] = 0
            # Računanje pzpt po danima i proizvodima
            pzptPro = pzptProFun(daniSync, potrSize, pocZal, zavZal, velNar,
                                 potrSum, daniDob)
            pzptDan = zadDan(pocZal, potr, potrSize) / potrSize
            # Provjera kriterija eksperimenta
            if pzptProChk or (pzptDDec <= pzptPro <= pzptGDec):
                if pzptDanChk or (pzptDDec <= pzptDan <= pzptGDec):
                    brojacZadExp += 1
                    # Priprema podataka za spremanje na disk
                    csvArr = np.stack((pocZal, potr, zavZal, izdNar, velNar),
                                      axis=-1)
                    # Spremanje zadovoljavajućeg eksperimenta
                    np.savetxt(
                        (izlFol+'/{}-{}-{:.5f}P-{:.5f}D-MVN{}-DD{}-RV{}od{}-'+
                                '{}-BL{:.5f}-No{:04}.csv').format(
                            maxZal, minZal, pzptPro*100, pzptDan*100, minNar,
                            daniDob, radVrDob, radVrSub, pocZal[0],
                            backLogPerc, brojacZadExp), csvArr, fmt='%1u',
                        delimiter=';', newline='\n')
                    if brojacZadExp >= sprNaj:
                        return
    return


cdef void diagSearch(long[:] potr, long pocZalExp, long maxZalD, long maxZalG,
                     long minZalD, long minZalG, long minNar, long daniDob,
                     long radVrDob, long radVrSub, long double backLogPerc,
                     long double pzptDPerc, long double pzptGPerc,
                     long pzptProChk, long pzptDanChk, izlFol, long sprNaj,
                     long potrSum, long potrSize, long[:] daniSync,
                     long brojacZadExp, long double diagStopPerc):
    """Dijagonalno pretraživanje"""
    # Pocetne zalihe, zavrsne zalihe, izdavanje narudzbe i velicina narudzbe
    # Matrice koje se mijenjaju tijekom eksperimenta
    cdef long[:] pocZal = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] zavZal = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] izdNar = np.zeros(potrSize, dtype=DTYPE)
    cdef long[:] velNar = np.zeros(potrSize, dtype=DTYPE)

    # Iteratori i brojači
    cdef long maxZal, minZal, dan, danUnatr, danIzdNar, danNerRad

    # Blokiranje narudžbe
    cdef long blockNar

    # Ukupna veličina narudžbe
    cdef long ukVelNar

    # Backlogging - broj proizvoda i koeficijent
    cdef long backLog
    cdef long double backLogKoef = backLogPerc / 100.

    # Trenutni PZPT
    cdef long double pzptPro
    cdef long double pzptDan

    # PZPT donja i gornja granica - koeficijenti
    cdef long double pzptDDec = pzptDPerc / 100.
    cdef long double pzptGDec = pzptGPerc / 100.

    # Koeficijent zaustavljanja dijagonalnog pretraživanja:
    cdef long double diagStopKoef = (100. - diagStopPerc) / 100.

    # Početak eksperimenta
    # Generiranje max razina zaliha
    for maxZal in range(maxZalD, maxZalG + 1):
        print('Dijagonalno maksimalna razina zaliha: %d' % maxZal)
        # Provjera: konstantne pocetne zalihe ili jednake maksimumu
        pocZal[0] = provjPocZal(maxZal, pocZalExp)

        # Generiranje min razina zaliha
        if (maxZal - minNar) <= minZalG:
            minZal = maxZal - minNar
        else:
            minZal = minZalG
        print('Minimalna razina zaliha: {}'.format(minZal))

        # Simulacijski eksperiment po danima
        # Prvi dan
        backLog = 0
        blockNar = 0
        if pocZal[0] >= potr[0]:
            zavZal[0] = pocZal[0] - potr[0]
        else:
            zavZal[0] = 0
            backLog += <long>(backLogKoef * (potr[0] - pocZal[0]))
        if zavZal[0] <= minZal and blockNar == 0:
            izdNar[0] = 1
            velNar[0] = ((maxZal - zavZal[0]) // minNar) * minNar
            blockNar = 1
        else:
            izdNar[0] = 0
            velNar[0] = 0

        # Dani bez dobave
        dan = 1
        while sum1d_pos(daniSync[:dan]) < (daniDob + 1):
            pocZal[dan] = zavZal[dan-1]
            if pocZal[dan] >= potr[dan]:
                zavZal[dan] = pocZal[dan] - potr[dan]
            else:
                zavZal[dan] = 0
                backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
            if zavZal[dan] <= minZal and blockNar == 0:
                izdNar[dan] = 1
                velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
                blockNar = 1
            else:
                izdNar[dan] = 0
                velNar[dan] = 0
            dan += 1

        # Prvi dan s mogućom dobavom
        if izdNar[0] == 1:
            if (zavZal[dan-1] + velNar[0]) >= backLog:
                pocZal[dan] = zavZal[dan-1] + velNar[0] - backLog
                backLog = 0
                blockNar = 0
            else:
                pocZal[dan] = 0
                backLog -= zavZal[dan-1] + velNar[0]
                blockNar = 0
        else:
            pocZal[dan] = zavZal[dan-1]
        if pocZal[dan] >= potr[dan]:
            zavZal[dan] = pocZal[dan] - potr[dan]
        else:
            zavZal[dan] = 0
            backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
        if zavZal[dan] <= minZal and blockNar == 0:
            izdNar[dan] = 1
            velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
            blockNar = 1
        else:
            izdNar[dan] = 0
            velNar[dan] = 0
        dan += 1

        # Svi ostali dani
        for dan in range(dan, potrSize):
            # Provjera izdane narudžbe
            if daniSync[dan] == 1:
                danUnatr = 1
                while (sum1d_pos(daniSync[dan-daniDob-danUnatr:dan]) <
                       (daniDob + 1)):
                    danUnatr += 1
                danIzdNar = dan - daniDob - danUnatr
                if daniSync[danIzdNar-1] == 1:
                    if izdNar[danIzdNar] == 1:
                        if (zavZal[dan-1] + velNar[danIzdNar]) >= backLog:
                            pocZal[dan] = (zavZal[dan-1] +
                                           velNar[danIzdNar] - backLog)
                            backLog = 0
                            blockNar = 0
                        else:
                            pocZal[dan] = 0
                            backLog -= zavZal[dan-1] + velNar[danIzdNar]
                            blockNar = 0
                    else:
                        pocZal[dan] = zavZal[dan-1]
                else:
                    ukVelNar = sum1d_pos(
                        velNar[danIzdNar-(radVrSub-radVrDob):danIzdNar+1])
                    if ukVelNar > 0:
                        if (zavZal[dan-1] + ukVelNar) >= backLog:
                            pocZal[dan] = (zavZal[dan-1] + ukVelNar
                                           - backLog)
                            backLog = 0
                            blockNar = 0
                        else:
                            pocZal[dan] = 0
                            backLog -= zavZal[dan-1] + ukVelNar
                            blockNar = 0
                    else:
                        pocZal[dan] = zavZal[dan-1]
            else:
                pocZal[dan] = zavZal[dan-1]
            # Računanje završnih zaliha i backlogginga
            if pocZal[dan] >= potr[dan]:
                zavZal[dan] = pocZal[dan] - potr[dan]
            else:
                zavZal[dan] = 0
                backLog += <long>(backLogKoef * (potr[dan] - pocZal[dan]))
            # Odluka o izdavanu i veličini narudžbe
            if zavZal[dan] <= minZal and blockNar == 0:
                izdNar[dan] = 1
                velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
                blockNar = 1
            else:
                izdNar[dan] = 0
                velNar[dan] = 0
        # Računanje pzpt po danima i proizvodima
        pzptPro = pzptProFun(daniSync, potrSize, pocZal, zavZal, velNar,
                             potrSum, daniDob)
        pzptDan = zadDan(pocZal, potr, potrSize) / potrSize
        # Provjera kriterija eksperimenta
        if pzptProChk or ((pzptDDec*diagStopKoef) <= pzptPro):
            if pzptDanChk or ((pzptDDec*diagStopKoef) <= pzptDan):
                bruteForceSearch(potr, pocZalExp, maxZal, maxZalG, minZalD,
                                 minZalG, minNar, daniDob, radVrDob, radVrSub,
                                 backLogPerc, pzptDPerc, pzptGPerc, pzptProChk,
                                 pzptDanChk, izlFol, sprNaj, potrSum, potrSize,
                                 daniSync, brojacZadExp)
                return
    return


def simulateExperiment(long[:] potr, long pocZalExp, long maxZalD,
                       long maxZalG, long minZalD, long minZalG, long minNar,
                       long daniDob, long radVrDob, long radVrSub,
                       long double backLogPerc, long double pzptDPerc,
                       long double pzptGPerc, long pzptProChk, long pzptDanChk,
                       izlFol, long sprNaj, long algType,
                       long double diagStopPerc):
    """Simulacijski eksperiment"""

    # Suma i velicina potraznje
    cdef long potrSum = sum1d_pos(potr)
    cdef long potrSize = potr.size

    # Sinkroniziranost radnih dana dobavljača i subjekta
    cdef long[:] daniSync = np.array((potrSize//radVrSub+1)*(
            [1]*radVrDob + [0]*(radVrSub-radVrDob)), dtype=DTYPE)

    # Iteratori i brojači
    cdef long brojacZadExp

    # Pocetak simulacije
    brojacZadExp = 0
    if algType == 0:
        bruteForceSearch(potr, pocZalExp, maxZalD, maxZalG, minZalD, minZalG,
                         minNar, daniDob, radVrDob, radVrSub, backLogPerc,
                         pzptDPerc, pzptGPerc, pzptProChk, pzptDanChk, izlFol,
                         sprNaj, potrSum, potrSize, daniSync, brojacZadExp)
    else:
        diagSearch(potr, pocZalExp, maxZalD, maxZalG, minZalD, minZalG, minNar,
                   daniDob, radVrDob, radVrSub, backLogPerc, pzptDPerc,
                   pzptGPerc, pzptProChk, pzptDanChk, izlFol, sprNaj, potrSum,
                   potrSize, daniSync, brojacZadExp, diagStopPerc)
    return


