# tree_v2

Model wykrywa symulowane błędy katalogowe. Konfiguracja wybrana na pięciu foldach
rozdzielonych po seller_id. Dane: edycja 2, 16000 ofert treningowych i 10000 testowych.

## Zapisane wyniki

- Pooled OOF ROC-AUC: 0.9137317258013836.
- OOF Average Precision: 0.8318584459533912.
- Publiczny ROC-AUC pierwszego zgłoszenia: 0.92212 (26 października 2026 r.).
- Ponowne wykonanie w nowym notebooku Kaggle: wersja 1 zakończona jako Successful.
- Opis zgłoszenia: tree_v2 | seller-disjoint CV 0.913732.

Publiczny wynik dotyczy 30% testu. Końcowy wynik prywatny pozostaje nieznany.
OOF wykorzystano do wyboru metody; nie jest niezależną końcową oceną.
Odnośniki do obecnego zgłoszenia są zapisane poniżej. Ostateczny wybór zgłoszeń
należy potwierdzić przed końcem konkursu.

## Odnośniki do notebooka i zgłoszenia

- Notebook z ponownym wykonaniem tree_v2: [https://www.kaggle.com/code/s35559/notebookea46d8fc4a](https://www.kaggle.com/code/s35559/notebookea46d8fc4a).
- Udokumentowana zapisana wersja tego notebooka: **1**, status **Successful**.
  Link prowadzi do notebooka; numer wersji zapisujemy osobno, ponieważ zwykły
  adres może w przyszłości wskazywać nowszą wersję.
- Zgłoszenie znajduje się w zakładce [Submissions konkursu](https://www.kaggle.com/competitions/dsc-pjatk-catalogue-quality-challenge/submissions).
- Plik: `submission.csv`.
- Data wysłania: 26 października 2026 r.
- Opis w Kaggle: `tree_v2 | seller-disjoint CV 0.913732`.
- Status zgłoszenia: **Complete**.
- Publiczny ROC-AUC: **0.92212**.
- SHA-256 wysłanego pliku i zgodnego z nim pliku z ponownego wykonania:
  `31d563767081227139413a9ab88dafc8ebdf151a94f646f50d5084c5fe05c733`.

Adres zakładki jest odnośnikiem do listy, a nie bezpośrednim adresem pojedynczego
zgłoszenia. Na przekazanym widoku nie było widocznego numeru zgłoszenia. Powyższe
dane oraz hash pliku wskazują udokumentowane zgłoszenie bez zgadywania jego numeru.
Źródła weryfikacji: przekazane pliki CSV i manifesty oraz widoki Kaggle.

Stan na 28 października 2026 r.: na przekazanym widoku wybrano ręcznie **0 z 2**
zgłoszeń do końcowej oceny. Ten opis dokumentuje obecne zgłoszenie tree_v2;
nie oznacza decyzji zespołu o finalnym wyborze. Przed końcem konkursu należy
potwierdzić wybór i zaktualizować ten opis, jeżeli konfiguracja lub zgłoszenie się zmienią.
Oryginalne manifesty uruchomień zachowujemy bez zmian; te odnośniki uzupełniają
ich dokumentację.

## Kaggle

Importuj plik notebooks/model_tree_v2.ipynb do Kaggle. Dołącz konkurs jako Input, ustaw CPU i Internet OFF,
następnie Run All. Zgłoszenie powstaje w /kaggle/working/submission.csv.
Wersja referencyjna środowiska jest przypięta do oryginalnego obrazu z 2026-10-22.

## Lokalnie

Referencja: Python 3.13.15. Biblioteki modelu są przypięte w requirements-model.txt.
Ogólny requirements.txt z repo może zawierać inne wersje do prac rozwojowych.
Wszystkie poniższe polecenia wykonuj w głównym folderze repozytorium,
tam gdzie są model_tree_v2.py i requirements-model.txt. Najpierw sprawdź
wersję interpretera: `python3 --version` na macOS/Linux lub `python --version`
na Windows. Środowisko venv korzysta z interpretera, którym je utworzysz;
samo venv nie instaluje ani nie zmienia wersji Pythona.

### macOS / Linux

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-model.txt
python model_tree_v2.py
```

### Windows — PowerShell

```powershell
python -m venv .venv-model
.\.venv-model\Scripts\Activate.ps1
python -m pip install -r requirements-model.txt
python model_tree_v2.py
```

Jeżeli PowerShell blokuje skrypt aktywacji, użyj bezpośrednio interpretera
utworzonego środowiska. Po wykonaniu pierwszej komendy powyżej uruchom:

```powershell
.\.venv-model\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv-model\Scripts\python.exe model_tree_v2.py
```

### Windows — wiersz polecenia CMD

```bat
python -m venv .venv-model
.venv-model\Scripts\activate.bat
python -m pip install -r requirements-model.txt
python model_tree_v2.py
```

Po pracy możesz wyłączyć aktywne środowisko poleceniem `deactivate`.
Polecenia aktywacji opisuje [dokumentacja Python venv](https://docs.python.org/3.13/library/venv.html).

Trzy pliki train.csv, test.csv i sample_submission.csv umieść w data/.
Skrypt zapisuje wyniki w outputs/ (na Kaggle w /kaggle/working).
Dane, OOF i lokalne środowisko pozostają poza repo; zachowaj wpis data/
oraz dodaj outputs/ i .venv-model/ do .gitignore.
Skrypt odtwarza ten sam algorytm co notebook. Zgodność przewidywań audytowano
w środowisku Kaggle. Polecenia Windows są zgodne z dokumentacją; nie były
sprawdzane przez wykonanie na Windows. Inny system lub wersje bibliotek mogą
wpływać na wynik numeryczny.

## Notebooki w repozytorium

- notebooks/model_tree_v2.ipynb — model, walidacja i predykcja.
- notebooks/02_Eksperyment_kategoria.ipynb — kontrolowane porównanie cech.

Na Kaggle importuj wybrany plik z folderu notebooks/ i dołącz dane konkursu.
Powyższa instrukcja lokalna dotyczy skryptu uruchamianego z głównego folderu repo.
Przeniesienie plików notebooków nie zmienia ich kodu ani wyników.

## Podział i model

Fold wynika z pierwszych 16 znaków SHA256 tekstu 42:seller_id modulo 5.
W pliku zapisujemy numery 1–5. Kod sprawdza hash danych oraz wspólnego podziału.
Słownik TF-IDF, SVD i kodowanie kategorii są dopasowywane osobno na treningu foldu.
ID i seller_id nie są wejściem modelu.

Cechy obejmują tekst i kategorie oraz cenę i wagę względem deklarowanej lub
odczytanej z tekstu liczby produktów. TF-IDF jest skracany do 64 składowych SVD.
HistGradientBoosting wykonuje 350 iteracji, maksymalnie 15 liści, learning_rate=0.06.
Pozostałe parametry i wersje są w model_manifest.json.
Zgłoszenie jest średnią pięciu modeli; nie wykonujemy ponownego fit na całym treningu.

## Odtwarzanie i wyniki wyjściowe

Kod zapisuje submission.csv, oof_tree_v2.csv, shared/folds.csv, model_manifest.json
oraz handoff_tree_v2.zip. OOF zawiera target treningowy i nie jest publikowany w repo.
Manifest w repo to zapis faktycznego uruchomienia Kaggle i pasuje do wysłanego CSV.

26 października 2026 r. porównano wcześniejszy plik zgłoszenia z wynikiem
uruchomienia modelu w nowo utworzonym notebooku Kaggle. Dla wszystkich 10000 ofert
ID i ich kolejność są zgodne ze wzorem zgłoszenia, przewidywania są skończone i w [0,1].
Maksymalna różnica przewidywań wynosi 0.0, czyli mieści się w tolerancji 1e-6.
Pliki CSV są identyczne bajtowo. Wersje bibliotek, parametry modelu oraz hashe danych
i foldów w manifestach obu uruchomień są zgodne. Szczegóły: reproduction_audit.json.

To porównanie wykonano poza notebookiem na pobranych plikach. Oryginalne manifesty
pozostają zapisem uruchomień: reference_not_available oznacza, że w czasie wykonania
nie było pliku w ścieżce oczekiwanej przez wewnętrzny test. Nie oznacza niezgodności
przewidywań. Audyt plików nie potwierdza samodzielnie ustawienia przełącznika Internet.
Przed oddaniem potwierdź wersję finalnego notebooka z Internet OFF i wybór zgłoszenia
w sekcji odnośników powyżej. Po zmianie wybranej konfiguracji ponów porównanie.
Na innych wersjach bibliotek wynik może się zmienić.

## Źródła

Funkcja cech bazowych pochodzi z baseline_solution.ipynb organizatorów DSC PJATK.
Rozszerzenia: parser ilości v2, dodatkowe cechy, model drzewiasty i kontrola artefaktów.
Notebook zawiera wyłącznie kod i opis; zapisane wyniki komórek zostały usunięte.
Wkład poszczególnych członków zespołu zostanie opisany w raporcie.
