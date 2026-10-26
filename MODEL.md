# tree_v2

Model wykrywa symulowane błędy katalogowe. Konfiguracja wybrana na pięciu foldach
rozdzielonych po seller_id. Dane: edycja 2, 16000 ofert treningowych i 10000 testowych.

## Zapisane wyniki

- Pooled OOF ROC-AUC: 0.9137317258013836.
- OOF Average Precision: 0.8318584459533912.
- Publiczny ROC-AUC pierwszego zgłoszenia: 0.92212 (26 października 2026 r.).
- Notebook Kaggle: 01_baseline, zapis modelu offline: wersja 5.
- Opis zgłoszenia: tree_v2 | seller-disjoint CV 0.913732.

Publiczny wynik dotyczy 30% testu. Końcowy wynik prywatny pozostaje nieznany.
OOF wykorzystano do wyboru metody; nie jest niezależną końcową oceną.
Numer/link wybranego zgłoszenia należy dopisać po ustaleniu finalnych submissions.

## Kaggle

Importuj model_tree_v2.ipynb. Dołącz konkurs jako Input, ustaw CPU i Internet OFF,
następnie Run All. Zgłoszenie powstaje w /kaggle/working/submission.csv.
Wersja referencyjna środowiska jest przypięta do oryginalnego obrazu z 2026-10-22.

## Lokalnie

Referencja: Python 3.13.15. Biblioteki modelu są przypięte w requirements-model.txt.
Ogólny requirements.txt z repo może zawierać inne wersje do prac rozwojowych.
W osobnym środowisku modelu uruchom:

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-model.txt
python model_tree_v2.py
```

Trzy pliki train.csv, test.csv i sample_submission.csv umieść w data/.
Zapisz wyniki w domyślnym outputs/. Pliki danych i OOF pozostają lokalne.
Przed dodaniem zmian do Git przejrzyj listę plików. data/ jest wykluczone w repo;
outputs/ należy dodać do .gitignore jeżeli jeszcze go tam nie ma.
Skrypt odtwarza ten sam algorytm co notebook.

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
Przed oddaniem zapisz finalną wersję notebooka z Internet OFF oraz dopisz dokładny
link/numer wybranego zgłoszenia Kaggle. Po zmianie wybranej konfiguracji ponów porównanie.
Na innych wersjach bibliotek wynik może się zmienić.

## Źródła

Funkcja cech bazowych pochodzi z baseline_solution.ipynb organizatorów DSC PJATK.
Rozszerzenia: parser ilości v2, dodatkowe cechy, model drzewiasty i kontrola artefaktów.
Notebook zawiera wyłącznie kod i opis; zapisane wyniki komórek zostały usunięte.
Wkład poszczególnych członków zespołu zostanie opisany w raporcie.
