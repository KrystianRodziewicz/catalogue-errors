# catalogue-errors

Projekt na rekrutację do DSC PJATK, konkurs na Kaggle DSC PJATK Catalogue Quality Challenge. Trzeba przewidzieć czy oferta w sklepie ma błąd. Metryka to ROC AUC.

## Wyniki

- model tree_v2: 0.9137 na walidacji (5 foldów po sprzedawcach), 0.9221 na public leaderboard
- cechy liczby sztuk z tekstu: z 0.8863 do 0.9150 (liczone w środowisku eda, inne wersje bibliotek niż referencyjne 0.9137)
- połączenia kategorii z tekstem: z 0.6177 do 0.7177 (regresja logistyczna)

## Pliki

- notebooks/eda.ipynb: eda, porównanie cech liczby sztuk i analiza 5 pomyłek
- notebooks/model_tree_v2.ipynb: model
- notebooks/02_Eksperyment_kategoria.ipynb: porównanie cech kategorii
- MODEL.md: opis modelu i jak go odpalić

## Jak odpalić

Model i eda mają osobne środowiska, żeby biblioteki się nie gryzły.

1. pobierz z Kaggle train.csv, test.csv i sample_submission.csv i wrzuć do data/ (danych nie ma w repo)
2. model w .venv-model (więcej w MODEL.md), zapisze outputs/oof_tree_v2.csv

```
python -m venv .venv-model
.venv-model\Scripts\activate
pip install -r requirements-model.txt
python model_tree_v2.py
deactivate
```

3. skopiuj outputs/oof_tree_v2.csv do data/handoff/
4. eda w .venv, potem odpal notebooks/eda.ipynb

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Na macOS/Linux zamiast .venv\Scripts\activate jest source .venv/bin/activate (tak samo dla .venv-model)

## Zespół

Team 3
- Krystian Rodziewicz: eda, porównanie liczby sztuk, analiza pomyłek, raport
- Piotr: model, porównanie kategorii, zgłoszenia na Kaggle
