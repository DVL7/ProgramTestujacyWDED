# Dokumentacja programu `ztest.py`

`tester.py` - program używany na ostatnich zajęciach, NIEAKTUALNY

## Opis projektu

`ztest.py` to skrypt testujący program dyskretyzujący dane numeryczne. Jego zadaniem jest wczytanie zewnętrznego pliku Python zawierającego funkcję `discretize(df)`, uruchomienie tej funkcji na danych z pliku CSV, a następnie sprawdzenie, czy wynik dyskretyzacji spełnia wymagania formalne.

Program kontroluje między innymi:

- czy funkcja `discretize(df)` istnieje w podanym pliku,
- czy funkcja zwraca obiekt typu `pandas.DataFrame`,
- czy liczba wierszy i kolumn w wyniku jest poprawna,
- czy nazwy i kolejność kolumn nie zostały zmienione,
- czy kolumna decyzyjna pozostała bez zmian,
- czy wartości atrybutów warunkowych zostały zastąpione poprawnymi przedziałami,
- czy każda oryginalna wartość należy do przypisanego jej przedziału,
- czy dla każdej kolumny numerycznej istnieją przedziały zaczynające się od `-inf` i kończące się na `inf`,
- ile cięć zostało wykonanych,
- ile rekordów powoduje konflikty decyzyjne po dyskretyzacji.

## Wymagania

Do uruchomienia programu wymagane są:

- Python 3,
- biblioteka `pandas`.

Instalacja wymaganej biblioteki:

```bash
pip install pandas
```

## Struktura wejścia

Program przyjmuje dwa argumenty uruchomieniowe:

```bash
python ztest.py plik.py data.csv
```

Gdzie:

- `plik.py` — plik Python zawierający funkcję `discretize(df)`,
- `data.csv` — plik CSV z danymi wejściowymi.

### Wymagania dla pliku z algorytmem

Plik z algorytmem musi mieć rozszerzenie `.py` i musi zawierać funkcję:

```python
def discretize(df):
    ...
    return wynikowy_dataframe
```

Funkcja powinna:

1. Przyjmować jako argument obiekt `pandas.DataFrame`.
2. Zwracać obiekt `pandas.DataFrame`.
3. Zachować tę samą liczbę wierszy co dane wejściowe.
4. Zachować te same nazwy i kolejność kolumn.
5. Nie zmieniać ostatniej kolumny, która jest traktowana jako kolumna decyzyjna.
6. Zamienić wartości w kolumnach warunkowych na przedziały.

### Wymagania dla pliku CSV

Plik CSV powinien zawierać dane tabelaryczne. Program zakłada, że:

- ostatnia kolumna jest kolumną decyzyjną,
- wszystkie wcześniejsze kolumny są kolumnami warunkowymi,
- wartości w kolumnach warunkowych można przekonwertować na liczby typu `float`.

Przykład pliku `data.csv`:

```csv
wiek,dochód,decyzja
23,3000,tak
45,7000,nie
31,4500,tak
```

## Format wyniku funkcji `discretize(df)`

Funkcja `discretize(df)` powinna zwrócić ramkę danych, w której wartości w kolumnach warunkowych są zastąpione przedziałami.

Akceptowane formaty przedziałów:

```text
[0, 10)
(0, 10]
[0; 10)
(-inf, 5)
[5, inf)
```

Program akceptuje zarówno przecinek, jak i średnik jako separator granic przedziału.

Przykład poprawnego wyniku:

```csv
wiek,dochód,decyzja
[-inf,30),[-inf,5000),tak
[30,inf),[5000,inf),nie
[30,inf),[-inf,5000),tak
```

> Uwaga: program sprawdza przynależność wartości według reguły `lewa_granica <= wartość < prawa_granica`. Oznacza to, że prawa granica przedziału jest traktowana jako otwarta, niezależnie od nawiasu użytego w zapisie tekstowym.

## Sposób działania programu

### 1. Wczytanie funkcji dyskretyzującej

Funkcja `load_discretize(algo_path)` dynamicznie ładuje wskazany plik Python i sprawdza, czy znajduje się w nim funkcja `discretize`.

Jeżeli funkcji nie ma, program zgłasza błąd:

```text
plik.py nie ma funkcji discretize(df)
```

### 2. Normalizacja wyniku

Funkcja `normalize_output(raw_df, out_df)` sprawdza liczbę kolumn w wyniku.

Dopuszczalne są dwa przypadki:

- wynik ma dokładnie tyle samo kolumn co dane wejściowe,
- wynik ma jedną dodatkową kolumnę na początku, która wygląda jak kolumna indeksu, np. `index`, `unnamed` lub `unnamed: 0`.

W drugim przypadku pierwsza kolumna jest usuwana.

### 3. Parsowanie granic przedziałów

Funkcja `parse_boundary(value)` zamienia granice przedziałów na wartości liczbowe.

Obsługiwane są specjalne wartości:

- `-inf`, `-infinity` — minus nieskończoność,
- `inf`, `+inf`, `infinity`, `+infinity` — plus nieskończoność.

Pozostałe wartości są konwertowane na `float`.

### 4. Parsowanie przedziałów

Funkcja `parse_interval(value)` odczytuje pojedynczy przedział.

Przedział może być przekazany jako:

- lista lub krotka dwuelementowa, np. `(0, 10)`,
- napis tekstowy, np. `[0, 10)`.

Jeżeli lewa granica jest większa lub równa prawej, program zgłasza błąd.

### 5. Uruchomienie testu

Funkcja `run_test(discretize, raw_df)` wykonuje główną logikę programu:

1. Uruchamia funkcję `discretize` na kopii danych wejściowych.
2. Mierzy czas wykonania.
3. Sprawdza typ zwróconego wyniku.
4. Normalizuje ewentualną kolumnę indeksu.
5. Sprawdza zgodność liczby wierszy i kolumn.
6. Sprawdza, czy kolumna decyzyjna nie została zmieniona.
7. Sprawdza poprawność każdego przedziału.
8. Sprawdza, czy oryginalne wartości należą do przypisanych przedziałów.
9. Sprawdza, czy każda kolumna warunkowa ma pokrycie od `-inf` do `inf`.
10. Liczy liczbę cięć.
11. Wykrywa konflikty decyzyjne.

## Liczba cięć

Liczba cięć jest obliczana jako suma liczby unikalnych przedziałów w każdej kolumnie warunkowej pomniejszonej o 1.

Dla każdej kolumny:

```text
liczba_cięć_w_kolumnie = liczba_unikalnych_przedziałów - 1
```

Całkowita liczba cięć:

```text
suma cięć ze wszystkich kolumn warunkowych
```

Przykład:

- kolumna `wiek` ma 3 unikalne przedziały, czyli 2 cięcia,
- kolumna `dochód` ma 2 unikalne przedziały, czyli 1 cięcie,
- razem: 3 cięcia.

## Konflikty decyzyjne

Konflikt występuje wtedy, gdy po dyskretyzacji dwa lub więcej obiektów mają identyczne wartości atrybutów warunkowych, ale różne wartości w kolumnie decyzyjnej.

Program grupuje rekordy według kolumn warunkowych. Następnie sprawdza, czy w danej grupie występuje więcej niż jedna wartość decyzji.

Jeżeli tak, wszystkie rekordy z tej grupy są liczone jako konfliktowe.

## Wynik programu

Po uruchomieniu program wypisuje raport w konsoli.

### Przykład wyniku pozytywnego

```text
Wynik testu:
Czas wykonania programu dyskretyzujacego: 0.002315 s
ZALICZONE
Liczba cięć: 3
Konflikty: 2
Raport Konfliktów:
2, 5
```

Znaczenie pól:

- `Czas wykonania programu dyskretyzujacego` — czas działania funkcji `discretize(df)`,
- `ZALICZONE` — wynik spełnia wymagania programu,
- `Liczba cięć` — liczba granic podziału utworzonych przez dyskretyzację,
- `Konflikty` — liczba rekordów należących do grup konfliktowych,
- `Raport Konfliktów` — numery wierszy, które należą do konfliktów.

### Przykład wyniku negatywnego

```text
Wynik testu:
Czas wykonania programu dyskretyzujacego: 0.001204 s
NIEZALICZONE
- niezgodna liczba wierszy: wejscie=10, wyjscie=9
- kolumna decyzyjna zostala zmieniona
```

Jeżeli wystąpią błędy, program kończy działanie z kodem wyjścia `1`.

## Obsługiwane błędy

Program może zgłosić między innymi następujące błędy:

| Błąd | Znaczenie |
|---|---|
| `Nie mozna zaladowac programu` | Nie udało się załadować pliku Python. |
| `nie ma funkcji discretize(df)` | Plik z algorytmem nie zawiera wymaganej funkcji. |
| `discretize() musi zwracac pandas.DataFrame` | Funkcja zwróciła obiekt innego typu niż `DataFrame`. |
| `Niepoprawna liczba kolumn` | Wynik ma nieprawidłową liczbę kolumn. |
| `niezgodna liczba wierszy` | Wynik ma inną liczbę wierszy niż dane wejściowe. |
| `niezgodne nazwy lub kolejnosc kolumn` | Kolumny w wyniku mają inne nazwy albo inną kolejność. |
| `kolumna decyzyjna zostala zmieniona` | Ostatnia kolumna została zmodyfikowana. |
| `niepoprawny format przedzialu` | Komórka nie zawiera poprawnego zapisu przedziału. |
| `wartosc ... nie nalezy do przedzialu` | Oryginalna wartość nie mieści się w przypisanym przedziale. |
| `kolumna ... nie ma przedzialu zaczynajacego sie od -inf` | W kolumnie brakuje przedziału rozpoczynającego się od minus nieskończoności. |
| `kolumna ... nie ma przedzialu konczacego sie na inf` | W kolumnie brakuje przedziału kończącego się na plus nieskończoności. |

## Kody zakończenia programu

| Kod | Znaczenie |
|---|---|
| `0` | Test zakończył się powodzeniem. |
| `1` | Wystąpił błąd walidacji lub błąd działania programu. |
| `2` | Podano niepoprawną liczbę argumentów. |

## Przykład minimalnego pliku z funkcją `discretize`

```python
import pandas as pd


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Zakładamy, że ostatnia kolumna jest decyzyjna,
    # dlatego dyskretyzujemy tylko kolumny wcześniejsze.
    for col in result.columns[:-1]:
        result[col] = result[col].apply(
            lambda x: "[-inf, 50)" if float(x) < 50 else "[50, inf)"
        )

    return result
```

Uruchomienie testu:

```bash
python ztest.py algorytm.py data.csv
```

## Uwagi implementacyjne

- Program wycisza standardowe wyjście funkcji `discretize(df)` podczas testu. Oznacza to, że komunikaty wypisywane przez testowany algorytm za pomocą `print()` nie będą widoczne.
- Dane wejściowe są przekazywane do funkcji jako kopia `raw_df.copy()`, więc testowany algorytm nie modyfikuje bezpośrednio oryginalnej ramki danych używanej przez tester.
- Program nie sprawdza, czy przedziały są rozłączne ani czy tworzą pełny, ciągły podział osi liczbowej. Sprawdza tylko, czy wartości wejściowe należą do przypisanych przedziałów oraz czy w każdej kolumnie pojawia się przedział od `-inf` i do `inf`.
- Nawiasy `()`, `[]` są rozpoznawane składniowo, ale logika sprawdzania przynależności zawsze używa warunku `left <= value < right`.

## Autor / przeznaczenie

Skrypt może być używany jako automatyczny tester rozwiązań implementujących dyskretyzację danych numerycznych, np. w zadaniach laboratoryjnych, projektach z eksploracji danych lub systemów decyzyjnych.
