import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path


def wybierz_csv() -> str:
    pliki = sorted(Path.cwd().glob("*.csv"))
    if not pliki:
        sys.exit("Błąd: brak plików .csv w bieżącym katalogu.")
    if len(pliki) == 1:
        print(f"[Auto] Wybrany plik: {pliki[0].name}")
        return str(pliki[0])
    print("Dostępne pliki CSV:")
    for i, p in enumerate(pliki, 1):
        print(f"  {i:2d}. {p.name}")
    while True:
        try:
            wybor = int(input("Numer pliku: "))
            if 1 <= wybor <= len(pliki):
                return str(pliki[wybor - 1])
        except ValueError:
            pass


def wczytaj_dane(sciezka: str) -> pd.DataFrame:
    for kodowanie in ("utf-8", "utf-8-sig", "latin-1", "cp1250"):
        try:
            naglowek = open(sciezka, encoding=kodowanie).read(4096)
            separator = max(",;|\t", key=naglowek.count)
            t0 = time.perf_counter()
            dane = pd.read_csv(sciezka, sep=separator, encoding=kodowanie, skipinitialspace=True)
            dane.dropna(how="all", axis=1, inplace=True)
            dane.dropna(how="all", axis=0, inplace=True)
            for kolumna in dane.columns:
                try:
                    dane[kolumna] = pd.to_numeric(dane[kolumna])
                except (ValueError, TypeError):
                    pass
            print(f"[Wczytano]  {Path(sciezka).name}  sep={separator!r}  enc={kodowanie}  "
                  f"shape={dane.shape}  ({time.perf_counter() - t0:.3f}s)")
            print(f"[Kolumny]   {dane.columns.tolist()}")
            print(f"[Decyzyjna] '{dane.columns[-1]}'")
            return dane
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Nie można odczytać pliku: {sciezka}")


def _zakoduj_wiersze(macierz: np.ndarray) -> np.ndarray:
    n, k = macierz.shape
    maks = macierz.max(axis=0).astype(np.int64) + 2
    if np.sum(np.log2(maks.astype(float))) < 62:
        skrot, mnoznik = np.zeros(n, np.int64), np.int64(1)
        for j in range(k):
            skrot += macierz[:, j].astype(np.int64) * mnoznik
            mnoznik *= maks[j]
        return skrot
    return (pd.DataFrame(macierz)
            .groupby(list(range(k)), sort=False)
            .ngroup()
            .values.astype(np.int64))


def licz_konflikty(macierz: np.ndarray, decyzje: np.ndarray) -> int:
    n = macierz.shape[0]
    if macierz.shape[1] == 0:
        return n if np.unique(decyzje).size > 1 else 0

    skroty = _zakoduj_wiersze(macierz)
    kolejnosc = np.argsort(skroty, kind="stable")
    posortowane_skroty = skroty[kolejnosc]
    posortowane_decyzje = decyzje[kolejnosc]

    poczatki = np.concatenate(([0], np.where(posortowane_skroty[1:] != posortowane_skroty[:-1])[0] + 1))
    rozmiary = np.diff(np.concatenate((poczatki, [n])))

    dec_min = np.minimum.reduceat(posortowane_decyzje, poczatki)
    dec_maks = np.maximum.reduceat(posortowane_decyzje, poczatki)

    return int(((dec_min != dec_maks) * rozmiary).sum())


def kandydaci_ciec(wartosci: np.ndarray, decyzje: np.ndarray) -> np.ndarray:
    kolejnosc = np.argsort(wartosci, kind="stable")
    w, d = wartosci[kolejnosc], decyzje[kolejnosc]
    maska = (d[:-1] != d[1:]) & (w[:-1] != w[1:])
    return np.unique((w[:-1][maska] + w[1:][maska]) / 2.0)


def zastosuj_ciecia(wartosci: np.ndarray, ciecia: np.ndarray) -> np.ndarray:
    if ciecia.size == 0:
        return np.zeros(len(wartosci), np.int32)
    return np.searchsorted(ciecia, wartosci, side="right").astype(np.int32)


def na_tuple(wartosci: np.ndarray, ciecia: np.ndarray) -> list:
    granice = np.concatenate(([-np.inf], ciecia, [np.inf]))
    indeksy = zastosuj_ciecia(wartosci, ciecia)
    return [(float(granice[i]), float(granice[i + 1])) for i in indeksy]


def discretize(dane: pd.DataFrame, maks_ciec: int = 10) -> pd.DataFrame:
    kolumna_dec = dane.columns[-1]
    kolumny_num = [k for k in dane.columns[:-1] if pd.api.types.is_numeric_dtype(dane[k])]
    kolumny_kat = [k for k in dane.columns[:-1] if k not in kolumny_num]

    enc_decyzje = pd.factorize(dane[kolumna_dec].values)[0].astype(np.int32)
    enc_kat = {k: pd.factorize(dane[k].values)[0].astype(np.int32) for k in kolumny_kat}

    surowe = {k: dane[k].values.astype(np.float64) for k in kolumny_num}
    ciecia = {k: np.empty(0, np.float64) for k in kolumny_num}
    kandydaci = {k: kandydaci_ciec(surowe[k], enc_decyzje) for k in kolumny_num}

    etykiety = {k: zastosuj_ciecia(surowe[k], ciecia[k]) for k in kolumny_num}

    def zbuduj_macierz() -> np.ndarray:
        kolumny = [etykiety[k] for k in kolumny_num] + [enc_kat[k] for k in kolumny_kat]
        return np.column_stack(kolumny).astype(np.int32) if kolumny \
            else np.empty((len(dane), 0), np.int32)

    t0 = time.perf_counter()

    for iteracja in range(10_000):
        macierz = zbuduj_macierz()
        konflikty = licz_konflikty(macierz, enc_decyzje)
        print(f"Iter {iteracja:4d}  konflikty={konflikty}")
        if konflikty == 0:
            break

        najlepszy_zysk, najlepsza_kol, najlepsze_ciecie = -1, None, None

        for kol in kolumny_num:
            if ciecia[kol].size >= maks_ciec:
                continue
            uzyte = set(ciecia[kol].tolist())
            oryginalne_etykiety = etykiety[kol]

            for ciecie in kandydaci[kol]:
                if ciecie in uzyte:
                    continue
                etykiety[kol] = zastosuj_ciecia(surowe[kol], np.sort(np.append(ciecia[kol], ciecie)))
                zysk = konflikty - licz_konflikty(zbuduj_macierz(), enc_decyzje)
                etykiety[kol] = oryginalne_etykiety

                if zysk > najlepszy_zysk:
                    najlepszy_zysk, najlepsza_kol, najlepsze_ciecie = zysk, kol, ciecie

        if najlepsza_kol is None or najlepszy_zysk <= 0:
            print("Brak poprawy – algorytm zatrzymany.")
            break

        ciecia[najlepsza_kol] = np.sort(np.append(ciecia[najlepsza_kol], najlepsze_ciecie))
        etykiety[najlepsza_kol] = zastosuj_ciecia(surowe[najlepsza_kol], ciecia[najlepsza_kol])
        print(f"  {najlepsza_kol} @ {najlepsze_ciecie:.6f}  (konflikty o {najlepszy_zysk})")

    czas = time.perf_counter() - t0
    koncowe_konflikty = licz_konflikty(zbuduj_macierz(), enc_decyzje)
    suma_ciec = sum(ciecia[k].size for k in kolumny_num)

    print(f"\n[Wynik algorytmu]")
    print(f"  Iteracje           : {iteracja + 1}")
    print(f"  Konflikty końcowe  : {koncowe_konflikty}")
    print(f"  Łączna liczba cięć : {suma_ciec}")
    print(f"  Czas algorytmu     : {czas:.4f}s")
    for kol in kolumny_num:
        if ciecia[kol].size:
            print(f"    {kol}: {ciecia[kol].tolist()}")

    dane_wyjsciowe = {k: na_tuple(surowe[k], ciecia[k]) for k in kolumny_num}
    for k in kolumny_kat:
        dane_wyjsciowe[k] = dane[k].tolist()
    dane_wyjsciowe[kolumna_dec] = dane[kolumna_dec].tolist()

    return pd.DataFrame(dane_wyjsciowe, columns=dane.columns, index=dane.index)


def raport_konfliktow(dane: pd.DataFrame) -> None:
    kolumny_war = dane.columns[:-1].tolist()
    kolumna_dec = dane.columns[-1]

    grupy = [
        sorted(g.index.tolist())
        for _, g in dane.groupby(kolumny_war, sort=False)
        if g[kolumna_dec].nunique() > 1
    ]

    if not grupy:
        print("Brak konfliktów.")
        return

    for grupa in grupy:
        print(", ".join(map(str, grupa)))
    print(f"\nLiczba konfliktujących wierszy: {sum(len(g) for g in grupy)}")


if __name__ == "__main__":
    sciezka_wej = sys.argv[1] if len(sys.argv) > 1 else wybierz_csv()
    sciezka_wyj = (sys.argv[2] if len(sys.argv) > 2
                   else str(Path(sciezka_wej).stem) + "_discretized.csv")

    dane = wczytaj_dane(sciezka_wej)

    czas_start = time.perf_counter()
    dane_wyj = discretize(dane)
    czas_calkowity = time.perf_counter() - czas_start

    kolumny_war = dane_wyj.columns[:-1].tolist()
    liczba_przedzialow = sum(dane_wyj[k].nunique() for k in kolumny_war)

    print(f"\n{'═' * 55}")
    print(f"  Łączna liczba przedziałów : {liczba_przedzialow}")
    print(f"  Czas całkowity            : {czas_calkowity:.4f}s")

    print(f"\n--- Raport konfliktów ---")
    raport_konfliktow(dane_wyj)

    # dane_wyj.to_csv(sciezka_wyj, index=False)
    # print(f"\n[Zapisano] → {sciezka_wyj}")

    print(dane_wyj.head(10).to_string())