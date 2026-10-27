# Materiał do raportu: model i eksperyment kategorii

Ten materiał opisuje sprawdzoną konfigurację tree_v2 oraz kontrolowane porównanie
cech kategorii i tekstu. Można włączyć go do sekcji „Podejście” oraz „Wyniki” raportu.
To część raportu, a nie kompletny raport zespołu. Nie zastępuje EDA, drugiego
kontrolowanego porównania ani pięciu analiz pomyłek.

## Cel i znaczenie etykiety

Przewidujemy prawdopodobieństwo, że generator wprowadził symulowany błąd do oferty.
Wartość target=1 oznacza wprowadzoną korupcję, a target=0 jej brak. Nie jest to
ludzka ocena poprawności oferty. Niska cena, brak opisu albo literówka same w sobie
nie wystarczają do uznania rekordu za błędny. Model ma ustawić oferty w kolejności
przydatnej do ręcznej kontroli; główną metryką konkursu jest ROC-AUC.

Cena price_pln dotyczy całego pakietu. shipping_weight_kg oznacza masę całej
przesyłki wraz z opakowaniem. pack_size określa liczbę głównych płatnych produktów
i nie obejmuje darmowych akcesoriów. Te rozróżnienia wpływają na konstrukcję cech.

## Walidacja po sprzedawcach

Zastosowano pięć wspólnych foldów. Numer foldu powstaje z pierwszych 16 znaków
zapisu szesnastkowego SHA-256 tekstu `42:seller_id`, następnie modulo 5 plus 1.
W efekcie ten sam sprzedawca zawsze trafia do tego samego foldu. Kod sprawdza
zgodność podziału i brak wspólnych sprzedawców między treningiem a walidacją.

Przy każdym z pięciu przebiegów model uczy się na czterech foldach i przewiduje
wyniki dla piątego. Zebrane przewidywania OOF obejmują wszystkie 16000 ofert.
Dla każdej z nich model nie miał dostępu do jej etykiety ani do ofert tego samego
sprzedawcy podczas danego dopasowania.

Podział po sprzedawcach odpowiada zadaniu konkursowemu, ponieważ sprzedawcy
treningowi i testowi są rozłączni. Losowy podział pojedynczych ofert mógłby
ułatwić wykorzystanie wspólnego stylu opisów i właściwości cenowych sprzedawcy.

Słownik TF-IDF, redukcja SVD i kodowanie kategorii są dopasowywane wewnątrz
treningowej części każdego foldu. W eksperymencie regresji logistycznej dotyczy
to także imputacji medianą i skalowania. Stałe reguły tekstowe nie uczą się
ze zbioru walidacyjnego. Kolumny id i seller_id nie są cechami predykcyjnymi.

Foldy wykorzystano do wyboru metody. Podane OOF nie stanowi niezależnej końcowej
oceny po całym procesie selekcji. Do takiej oceny potrzebny byłby osobny nietknięty
zbiór sprzedawców lub walidacja zagnieżdżona.

## Przygotowanie danych i cechy tree_v2

Model łączy 24 cechy liczbowe, trzy pola kategorialne i reprezentację tekstu.
Cechy liczbowe obejmują logarytmiczne przekształcenia ceny i wagi, wielkość
pakietu, cenę i masę na deklarowany produkt, długości tekstu oraz wskaźniki braków.
Dodatkowo wykorzystano wzmianki o promocji, akcesoriach i wariancie produktu.

Parser ilości v2 odczytuje ilości z tytułu i opisu. Rozpoznaje część zapisów
cyfrowych i słownych oraz pomija fragmenty zdań dotyczące gratisów i akcesoriów.
Zapisuje wskazówki ilości, brak wskazówki i konflikt między znalezionymi liczbami.
Na ich podstawie powstają różnica względem pack_size, stosunek ilości oraz cena
i masa przeliczone na ilość z tekstu. Parser jest heurystyką: może nie rozpoznać
literówki albo pominąć przydatną informację w zdaniu wspominającym akcesoria.
Żadna z jego wskazówek nie jest traktowana jak prawdziwa etykieta błędu.

Pola kategorialne to declared_category, brand i condition. Braki zastępuje
stała wartość, a nieznane kategorie są obsługiwane przez konfigurację kodowania.
Braki w cechach liczbowych pozostają brakami obsługiwanymi przez model drzewiasty.

Tekst tytułu i opisu jest łączony i zamieniany na małe litery. Przed budową
reprezentacji tekstowej usuwane są kody pasujące do wzorca litera i trzy cyfry.
TF-IDF wykorzystuje pojedyncze słowa oraz pary słów, min_df=3 i maksymalnie
20000 cech. TruncatedSVD redukuje tę reprezentację do 64 składowych.

## Model i tworzenie zgłoszenia

Konfiguracja tree_v2 korzysta z HistGradientBoostingClassifier. Model ten pozwala
uczyć zależności między ceną, wagą, ilością i pozostałymi cechami, zamiast narzucać
jedną liniową zależność. Wybrana konfiguracja ma 350 iteracji, learning_rate=0.06,
maksymalnie 15 liści, co najmniej 25 próbek w liściu i l2_regularization=2.
Wczesne zatrzymywanie jest wyłączone, a random_state wynosi 42.

Pięć modeli foldowych przewiduje wyniki dla zbioru testowego. Końcowa prognoza
to średnia tych pięciu prawdopodobieństw. Nie dopasowujemy dodatkowego modelu
na całym zbiorze treningowym. Zgłoszenie zawiera wyłącznie kolumny id,target,
bez progowania do zera i jedynki.

## Kontrolowane porównanie: połączenia kategorii ze słowami

Hipoteza: połączenie deklarowanej kategorii z tekstem pomaga rozpoznawać
nietypowe zestawienia. Przykładowo tekst `wireless headphones` w kategorii audio
daje tokeny `audio__wireless` i `audio__headphones`. Ten sam tekst w kategorii
kitchen daje inne tokeny. Model może uczyć się różnic między tymi zestawieniami.

W obu konfiguracjach zastosowano regresję logistyczną z C=2, solver=liblinear,
max_iter=600 i random_state=42. Pozostawiono te same wiersze, foldy i parser ilości
v1. W drugiej konfiguracji usunięto wyłącznie gałąź TF-IDF dla category_text.
Pozostałe cechy i ustawienia nie zmieniały się. To porównanie dotyczy regresji
logistycznej; nie jest ablacją modelu tree_v2.

| Konfiguracja | Pooled OOF ROC-AUC | Różnica względem wersji z cechą | Czas [s] |
|---|---:|---:|---:|
| Z połączeniami kategorii i tekstu | 0.717692 | 0.000000 | 14.835852 |
| Bez tych połączeń | 0.617712 | -0.099979 | 9.651587 |

Czas obejmuje budowę modelu, dopasowanie i predykcję walidacyjną. Nie obejmuje
wspólnego deterministycznego tworzenia cech ani zapisu plików. Są to czasy
konkretnego uruchomienia, a nie średnie z wielokrotnych pomiarów.

Wynik pogorszył się we wszystkich pięciu foldach. W tej konfiguracji gałąź
połączeń kategorii i tekstu pomagała w rankingu błędów. Nie dowodzi to, że każda
niezgodność kategorii jest błędem ani że ta sama cecha pomoże każdemu modelowi.
Eksperyment nie identyfikuje ukrytego typu korupcji konkretnego rekordu.

## Wyniki w poszczególnych foldach

| Fold | tree_v2 ROC-AUC | Regresja z category_text | Regresja bez category_text |
|---|---:|---:|---:|
| 1 | 0.913415 | 0.711263 | 0.612977 |
| 2 | 0.908780 | 0.695107 | 0.612291 |
| 3 | 0.916033 | 0.725747 | 0.608045 |
| 4 | 0.907208 | 0.723009 | 0.626070 |
| 5 | 0.923113 | 0.729921 | 0.628997 |

Tabela zawiera osobno model końcowy i dwie wersje kontrolowanego eksperymentu.
Porównanie tree_v2 z regresją nie izoluje pojedynczej zmiany: różnią się modelem
i zestawem cech. Dlatego nie przypisujemy całej różnicy jednej cesze.

## Ocena konfiguracji tree_v2

Pooled OOF ROC-AUC wynosi 0.913732, a OOF Average Precision
0.831858. ROC-AUC ocenia kolejność ofert, a nie procent
poprawnych decyzji. Average Precision uzupełnia ocenę rankingu przy rzadszej klasie
błędów. Żadna z tych wartości nie potwierdza kalibracji prawdopodobieństw.

Wynik publicznego zgłoszenia udokumentowany dla tej konfiguracji wyniósł 0.92212.
Dotyczy on publicznej części testu i nie zastępuje walidacji. Wynik prywatny,
który określa końcowe miejsce, nie jest jeszcze znany.

## Odtwarzalność

Sprawdzone środowisko: Python 3.13.15, NumPy 2.1.3,
pandas 2.3.3, scikit-learn 1.6.1,
SciPy 1.16.3 i threadpoolctl 3.6.0.
Ustawiono limit czterech wątków. Kod nie wymaga zewnętrznych danych,
pretrenowanych modeli ani pobierania zasobów z Internetu.

Dla pliku zgłoszenia z ponownego wykonania sprawdzono 10000 identyfikatorów,
kolejność zgodną ze sample_submission, zakres [0,1] i skończoność przewidywań.
Porównanie ze wcześniejszym plikiem dało maksymalną różnicę 0.0
przy tolerancji 1e-6. Pliki są identyczne bajtowo. Wersje bibliotek, parametry
oraz hashe danych i foldów w manifestach są zgodne. Zewnętrzne potwierdzenie
porównania znajduje się w reproduction_audit.json. Samo pole
reference_not_available w oryginalnym manifeście dotyczy braku pliku w ścieżce
wewnętrznego testu podczas wykonania, a nie wyniku tego późniejszego porównania.

## Ograniczenia i źródła

Zbiór jest syntetyczny. Wynik nie dowodzi skuteczności wykrywania rzeczywistych
nadużyć na rynku. Nawet poprawny ranking może zawierać niejednoznaczne przypadki
wynikające z nakładania się poprawnych i zaburzonych wartości. Pięć analiz błędów
powinno omawiać obserwowane dowody i możliwe wyjaśnienia, bez deklarowania wiedzy
o nieujawnionym typie korupcji.

Podstawą części cech jest baseline_solution.ipynb organizatorów DSC PJATK.
Kod użytej konfiguracji jest w model_tree_v2.ipynb i model_tree_v2.py.
Kontrolowane porównanie dokumentują 02_Eksperyment_kategoria.ipynb,
category_comparison.csv i category_metadata.json. OOF i dane ofert pozostają
prywatne. Instrukcję uruchomienia zawiera MODEL.md.

Przed złożeniem całego raportu należy dołączyć EDA, drugie porównanie cech,
pięć analiz pomyłek ze wspólną regułą decyzji oraz uzgodniony opis wkładu członków
zespołu. Trzeba również wskazać dokładny link lub numer wybranego zgłoszenia
Kaggle oraz finalną zapisaną wersję notebooka.
