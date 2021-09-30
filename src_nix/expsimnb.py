import numpy as np
import numba as nb


@nb.jit(
    [
        nb.int64(nb.int64),
        nb.float64(nb.float64)
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def int_pos(a):
    """
    Truncate to zero

    :param a:
    float64 or int64

    :return:
    float64 or int64 positive value or zero

    """
    return a if a >= 0 else 0


@nb.jit(
    [
        nb.int64(nb.int64[:]),
        nb.float64(nb.float64[:])
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def sum1d_pos(summand):
    """
    Sumiranje clanova long matrice

    :param summand:
    np.array int64 or float64

    :return:
    int64 or float64 elementwise sum
    """
    total = 0
    for i in range(summand.size):
        total += summand[i]
    return total


@nb.jit(
    [
        nb.float64(
            nb.int64, nb.int64[:], nb.int64[:], nb.int64[:], nb.int64
            )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def pzptProFun(potrSize, pocZal, zavZal, velNar, potrSum):
    """PZPT po proizvodima"""
    sumVelNar = float(velNar.sum())
    for i in range(potrSize-1, 0, -1):
        if velNar[i] != 0:
            break
    sumVelNar -= velNar[i]
    for j in range(i, potrSize-1):
        if pocZal[j+1] > zavZal[j]:
            sumVelNar += velNar[i]
            break
    sumVelNar += pocZal[0] - int_pos(zavZal[potrSize-1])
    pzptPro = sumVelNar / potrSum
    return pzptPro


@nb.jit(
    [
        nb.int64(
            nb.int64[:], nb.int64[:], nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def zadDan(pocZal, potr, potrSize):
    """Broj dana u kojima je zadovoljena potražnja"""
    brDan = 0
    for i in range(potrSize):
        if pocZal[i] >= potr[i]:
            brDan += 1
    return brDan


@nb.jit(
    [
        nb.int64(
            nb.int64, nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def provjPocZal(maxZal, pocZalExp):
    """Vraca maksimalnu razinu zaliha eksperimenta ili konstantnu pocetnu
    zalihu"""
    if pocZalExp > 0:
       return pocZalExp
    else:
       return maxZal


@nb.jit(
    [
        nb.int64(
            nb.int64, nb.int64, nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def zavZalFun(pocZal, potr, blChk):
    """

    :param pocZal:
    :param potr:
    :param blChk:
    :return:
    """
    if blChk:
        return int_pos(pocZal - potr)
    else:
        return pocZal - potr


@nb.jit(
    [
        nb.float64(
            nb.int64[:], nb.int64[:], nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def blDecFun(pocZal, zavZal, potrSum):
    return float(
        abs(zavZal[zavZal < 0].sum()) - abs(pocZal[pocZal < 0].sum())
    ) / potrSum


@nb.jit(
    [
        nb.boolean(
            nb.int64, nb.int64, nb.int64[:]
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def velDobChkFun(velDobD, velDobG, velNar):
    if np.any(velNar):
        return (velDobD <= velNar[velNar > 0].min() <= velDobG) and\
               (velDobD <= velNar[velNar > 0].max() <= velDobG)
    else:
        return (velDobD <= 0 <= velDobG)


@nb.jit(
    nopython=False,
    cache=True,
    fastmath=True,
)
def expSave(csvArr, old_name, prepend, izlFol, maxZal, minZal, pzptPro,
            pzptDan, minNar, daniDob, radVrDob, radVrSub, pocZal, blDec,
            brojacZadExp):
    # Spremanje zadovoljavajućeg eksperimenta
    full_res = np.concatenate(
        (prepend, csvArr.astype(np.str_)), axis=-1
    ) if prepend.size else csvArr.astype(np.str_)
    full_path = (
            izlFol + '/' + old_name +
            '{}-{}-{:.5f}P-{:.5f}D-MVN{}-DD{}-RV{}od{}-' +
            '{}-BL{:.5f}-No{:04}.csv').format(
                maxZal, minZal, pzptPro*100, pzptDan*100, minNar, daniDob,
                radVrDob, radVrSub, pocZal[0], blDec*100, brojacZadExp
            )
    np.savetxt(
        full_path, full_res,
        fmt='%s',
        delimiter=';', newline='\n'
    )


@nb.jit(
    [
        nb.types.UniTuple(nb.int64[:], 4)(
            nb.int64, nb.int64, nb.int64, nb.int64, nb.int64[:], nb.int64,
            nb.int64, nb.int64[:], nb.int64, nb.int64, nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def simExpRun(minZal, maxZal, potrSize, pocZalExp, potr, blChk,
              minNar, daniSync, daniDob, radVrSub, radVrDob):
    # Pocetne zalihe, zavrsne zalihe, izdavanje narudzbe i velicina narudzbe
    # Matrice koje se mijenjaju tijekom eksperimenta
    pocZal = np.zeros(potrSize, np.int64)
    zavZal = np.zeros(potrSize, np.int64)
    izdNar = np.zeros(potrSize, np.int64)
    velNar = np.zeros(potrSize, np.int64)

    # Simulacijski eksperiment po danima
    # Prvi dan
    # Provjera: konstantne pocetne zalihe ili jednake maksimumu
    pocZal[0] = provjPocZal(maxZal, pocZalExp)
    blockNar = 0
    zavZal[0] = zavZalFun(pocZal[0], potr[0], blChk)
    if zavZal[0] <= minZal and blockNar == 0:
        izdNar[0] = 1
        velNar[0] = ((maxZal - zavZal[0]) // minNar) * minNar
        blockNar = 1
    else:
        izdNar[0] = 0
        velNar[0] = 0

    # Dani bez dobave
    dan = 1
    while daniSync[:dan].sum() < (daniDob + 1):
        pocZal[dan] = zavZal[dan-1]
        zavZal[dan] = zavZalFun(pocZal[dan], potr[dan], blChk)
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
        pocZal[dan] = zavZal[dan-1] + velNar[0]
        blockNar = 0
    else:
        pocZal[dan] = zavZal[dan-1]
    zavZal[dan] = zavZalFun(pocZal[dan], potr[dan], blChk)
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
            while daniSync[dan-daniDob-danUnatr:dan].sum() < (daniDob + 1):
                danUnatr += 1
            danIzdNar = dan - daniDob - danUnatr
            if daniSync[danIzdNar-1] == 1:
                if izdNar[danIzdNar] == 1:
                    pocZal[dan] = zavZal[dan-1] + velNar[danIzdNar]
                    blockNar = 0
                else:
                    pocZal[dan] = zavZal[dan-1]
            else:
                ukVelNar = \
                    velNar[danIzdNar-(radVrSub-radVrDob):danIzdNar+1].sum()
                if ukVelNar > 0:
                    pocZal[dan] = zavZal[dan-1] + ukVelNar
                    blockNar = 0
                else:
                    pocZal[dan] = zavZal[dan-1]
        else:
            pocZal[dan] = zavZal[dan-1]
        # Računanje završnih zaliha i backlogginga
        zavZal[dan] = zavZalFun(pocZal[dan], potr[dan], blChk)
        # Odluka o izdavanu i veličini narudžbe
        if zavZal[dan] <= minZal and blockNar == 0:
            izdNar[dan] = 1
            velNar[dan] = ((maxZal - zavZal[dan]) // minNar) * minNar
            blockNar = 1
        else:
            izdNar[dan] = 0
            velNar[dan] = 0
    return pocZal, zavZal, izdNar, velNar


@nb.jit(
    [
        nb.int64(
            nb.int64[:], nb.types.string, nb.types.string[:, :], nb.int64,
            nb.int64, nb.int64, nb.int64, nb.int64, nb.int64, nb.int64,
            nb.int64, nb.int64, nb.int64, nb.float64, nb.float64, nb.float64,
            nb.float64, nb.float64, nb.float64, nb.int64,
            nb.int64, nb.types.string, nb.int64, nb.int64, nb.int64,
            nb.int64[:], nb.int64, nb.int64, nb.int64
        )
    ],
    nopython=False,
    cache=True,
    fastmath=True,
)
def bruteForceSearch(potr, old_name, prepend, pocZalExp,
                     maxZalD, maxZalG, minZalD, minZalG, minNar, daniDob,
                     radVrDob, radVrSub, blChk, blDDec, blGDec, pzptProDDec,
                     pzptProGDec, pzptDanDDec, pzptDanGDec, pzptProChk,
                     pzptDanChk, izlFol, sprNaj, potrSum, potrSize,
                     daniSync, velDobChk, velDobD, velDobG):
    """Brute-Force pretraživanje"""
    # Brojac zadovoljavajucih eksperimenata
    brojacZadExp = 0

    # Početak eksperimenta
    # Generiranje max razina zaliha
    for maxZal in range(maxZalD, maxZalG + 1):
        # print('{} Brute maksimalna razina zaliha: {}'.format(old_name[:6], maxZal))

        # Generiranje min razina zaliha
        for minZal in range(minZalD, maxZal + 1 - minNar):
            if minZal > minZalG:
                break
            # print('Minimalna razina zaliha: {}'.format(minZal))

            # Simulacijski eksperiment po danima
            pocZal, zavZal, izdNar, velNar = simExpRun(
                minZal, maxZal, potrSize, pocZalExp, potr, blChk, minNar,
                daniSync, daniDob, radVrSub, radVrDob)

            # Računanje pzpt po danima i proizvodima
            pzptPro = pzptProFun(potrSize, pocZal, zavZal, velNar, potrSum)
            pzptDan = zadDan(pocZal, potr, potrSize) / float(potrSize)
            blDec = blDecFun(pocZal, zavZal, potrSum)
            # Provjera kriterija eksperimenta
            if pzptProChk or (pzptProDDec <= pzptPro <= pzptProGDec):
                if pzptDanChk or (pzptDanDDec <= pzptDan <= pzptDanGDec):
                    if blChk or (blDDec <= blDec <= blGDec):
                        if velDobChk or velDobChkFun(velDobD, velDobG, velNar):
                            brojacZadExp += 1
                            # Priprema podataka za spremanje na disk
                            csvArr = np.stack(
                                (pocZal, potr, zavZal, izdNar, velNar),
                                axis=-1
                            )
                            expSave(
                                csvArr, old_name, prepend, izlFol, maxZal,
                                minZal, pzptPro, pzptDan, minNar, daniDob,
                                radVrDob, radVrSub, pocZal, blDec,
                                brojacZadExp
                            )
                            if brojacZadExp >= sprNaj:
                                return brojacZadExp
        # print(pzptPro)
        # print(pzptDan)
        # print(blDec)
    return brojacZadExp


@nb.jit(
    [
        nb.int64(
            nb.int64[:], nb.types.string, nb.types.string[:, :], nb.int64,
            nb.int64, nb.int64, nb.int64, nb.int64, nb.int64, nb.int64,
            nb.int64, nb.int64, nb.int64, nb.float64, nb.float64, nb.float64,
            nb.float64, nb.float64, nb.float64, nb.int64, nb.int64,
            nb.types.string, nb.int64, nb.int64, nb.int64, nb.int64[:],
            nb.float64, nb.int64, nb.int64, nb.int64
        )
    ],
    nopython=False,
    cache=True,
    fastmath=True,
)
def diagSearch(potr, old_name, prepend, pocZalExp,
               maxZalD, maxZalG, minZalD, minZalG, minNar, daniDob,
               radVrDob, radVrSub, blChk, blDDec, blGDec, pzptProDDec,
               pzptProGDec, pzptDanDDec, pzptDanGDec, pzptProChk, pzptDanChk,
               izlFol, sprNaj, potrSum, potrSize, daniSync,
               diagStopDec, velDobChk, velDobD, velDobG):
    """Dijagonalno pretraživanje"""

    # Početak eksperimenta
    brojacZadExp = 0
    # Generiranje max razina zaliha
    for maxZal in range(maxZalD, maxZalG + 1):
        # print('Dijagonalno maksimalna razina zaliha: %d' % maxZal)

        # Generiranje min razina zaliha
        if (maxZal - minNar) <= minZalG:
            minZal = maxZal - minNar
        else:
            minZal = minZalG
        # print('Minimalna razina zaliha: {}'.format(minZal))

        # Simulacijski eksperiment po danima
        pocZal, zavZal, izdNar, velNar = simExpRun(
            minZal, maxZal, potrSize, pocZalExp, potr, blChk, minNar,
            daniSync, daniDob, radVrSub, radVrDob)

        # Računanje pzpt po danima i proizvodima
        pzptPro = pzptProFun(potrSize, pocZal, zavZal, velNar, potrSum)
        pzptDan = zadDan(pocZal, potr, potrSize) / float(potrSize)
        # print(pzptPro)
        # print(pzptDan)
        # Provjera kriterija eksperimenta za ukljucivanje brute-force
        if (not pzptProChk) and ((pzptProDDec-diagStopDec) <= pzptPro):
            brojacZadExp = bruteForceSearch(
                potr, old_name, prepend, pocZalExp, maxZal, maxZalG,
                minZalD, minZalG, minNar, daniDob, radVrDob, radVrSub,
                blChk, blDDec, blGDec, pzptProDDec, pzptProGDec,
                pzptDanDDec, pzptDanGDec, pzptProChk, pzptDanChk, izlFol,
                sprNaj, potrSum, potrSize, daniSync, velDobChk, velDobD,
                velDobG)
            return brojacZadExp
        elif (not pzptDanChk) and ((pzptDanDDec-diagStopDec) <= pzptDan):
            brojacZadExp = bruteForceSearch(
                potr, old_name, prepend, pocZalExp, maxZal, maxZalG,
                minZalD, minZalG, minNar, daniDob, radVrDob, radVrSub,
                blChk, blDDec, blGDec, pzptProDDec, pzptProGDec,
                pzptDanDDec, pzptDanGDec, pzptProChk, pzptDanChk, izlFol,
                sprNaj, potrSum, potrSize, daniSync, velDobChk, velDobD,
                velDobG)
            return brojacZadExp
    return brojacZadExp


def findOptExp(potr, old_name, prepend, pocZalExp, maxZalD, maxZalG, minZalD,
               minZalG, minNar, daniDob, radVrDob, radVrSub, blChk, blDPerc,
               blGPerc, pzptProDPerc, pzptProGPerc, pzptDanDPerc, pzptDanGPerc,
               pzptProChk, pzptDanChk, izlFol, sprNaj, diagStopPerc, velDobChk,
               velDobD, velDobG):
    """Simulacijski eksperiment"""

    # Suma i velicina potraznje
    potrSum = potr.sum()
    potrSize = potr.size

    # Sinkroniziranost radnih dana dobavljača i subjekta
    daniSync = np.array((potrSize//radVrSub+1)*(
            [1]*radVrDob + [0]*(radVrSub-radVrDob)), np.int64)

    # Backlogging - broj proizvoda i koeficijent
    # cdef long backLog
    blDDec = blDPerc / 100.
    blGDec = blGPerc / 100.

    # PZPT donja i gornja granica - koeficijenti
    pzptProDDec = pzptProDPerc / 100.
    pzptProGDec = pzptProGPerc / 100.
    pzptDanDDec = pzptDanDPerc / 100.
    pzptDanGDec = pzptDanGPerc / 100.

    # Koeficijent zaustavljanja dijagonalnog pretraživanja:
    diagStopDec = diagStopPerc / 100.

    # Pocetak simulacije
    if potrSum > 0 and potrSize > 0 and minNar > 0 and radVrSub > 0:
        brojacZadExp = diagSearch(
            potr, old_name, prepend, pocZalExp, maxZalD, maxZalG, minZalD,
            minZalG, minNar, daniDob, radVrDob, radVrSub, blChk, blDDec,
            blGDec, pzptProDDec, pzptProGDec, pzptDanDDec, pzptDanGDec,
            pzptProChk, pzptDanChk, izlFol, sprNaj, potrSum, potrSize,
            daniSync, diagStopDec, velDobChk, velDobD, velDobG
        )
    else:
        brojacZadExp = 0
    return brojacZadExp


