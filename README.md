# catalogue-errors

Projekt na rekrutację do DSC PJATK, konkurs na Kaggle DSC PJATK Catalogue Quality Challenge. Trzeba przewidzieć czy oferta w sklepie ma błąd. Metryka to ROC AUC.

## Wyniki

- model tree_v2: 0.9137 na walidacji (5 foldów po sprzedawcach), 0.9221 na public leaderboard
- cechy liczby sztuk z tekstu: z 0.8863 do 0.9150
- połączenia kategorii z tekstem: z 0.6177 do 0.7177 (regresja logistyczna)

## Pliki

- notebooks/eda.ipynb: eda, porównanie cech liczby sztuk i analiza 5 pomyłek
- notebooks/model_tree_v2.ipynb: model
- notebooks/02_Eksperyment_kategoria.ipynb: porównanie cech kategorii
- MODEL.md: opis modelu i jak go odpalić

## Jak odpalić

1. pobierz z Kaggle train.csv, test.csv i sample_submission.csv i wrzuć do data/ (danych nie ma w repo)
2. odpal model jak w MODEL.md, zapisze outputs/oof_tree_v2.csv
3. skopiuj ten plik do data/handoff/
4. pip install -r requirements.txt i odpal notebooks/eda.ipynb

## Zespół

Team 3
- Krystian Rodziewicz: eda, porównanie liczby sztuk, analiza pomyłek, raport
- Piotr: model, porównanie kategorii, zgłoszenia na Kaggle
