# Program Testujący Dyskretyzację

Prosty zestaw narzędzi do uruchamiania programu dyskretyzującego, mierzenia czasu jego działania i sprawdzania poprawności wygenerowanego wyniku.

## Uruchomienie

### macOS / Linux

1. Utwórz środowisko wirtualne.

```bash
python -m venv venv
```

2. Aktywuj środowisko.

```bash
source venv/bin/activate
```

3. Zainstaluj zależności.

```bash
pip install -r requirements.txt
```

4. Uruchom tester.

```bash
python3 tester.py
```

### Windows

1. Utwórz środowisko wirtualne.

```powershell
python -m venv venv
```

2. Aktywuj środowisko.

```powershell
venv\Scripts\activate
```

3. Zainstaluj zależności.

```powershell
pip install -r requirements.txt
```

4. Uruchom tester.

```powershell
python tester.py
```

## Pliki

- `iris.csv` - plik wejściowy z surowymi danymi.
- `test_discretizer.py` - przykładowy program dyskretyzujący.
- `tester.py` - program testujący.
- `tester_config.json` - parametry wejściowe programu `tester.py`.
- `requirements.txt` - lista bibliotek do zainstalowania.
