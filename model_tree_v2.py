"""Reproduce tree_v2. Run from the repo root: python model_tree_v2.py."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '4'

import hashlib
import json
import platform
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_DIR = None
FOLDS_PATH = None
OUTPUT_ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
EXPECTED_TRAIN_SHA256 = 'aefc970e5efc4435cecfe76b901149fdd9ffd06bc62a3f62773379902b34195c'
EXPECTED_FOLDS_SHA256 = 'c6a234966e2b3107f3c24e9d3aa0d565430027cb6e16212ebf612a993ccc6956'

def find_data():
    if DATA_DIR is not None:
        candidates = [Path(DATA_DIR)]
    else:
        candidates = [p for p in [Path('data/public'), Path('data'), Path('.')]
                      if all((p / name).is_file() for name in
                             ['train.csv', 'test.csv', 'sample_submission.csv'])]
        if Path('/kaggle/input').exists():
            candidates += [p.parent for p in Path('/kaggle/input').rglob('train.csv')
                           if (p.parent / 'test.csv').is_file()
                           and (p.parent / 'sample_submission.csv').is_file()]
    candidates = sorted(set(p.resolve() for p in candidates))
    if len(candidates) != 1:
        raise ValueError('Dodaj dane konkursu do Input lub ustaw DATA_DIR.')
    return candidates[0]

def canonical_folds(train):
    values = [int(hashlib.sha256(f'{SEED}:{seller}'.encode()).hexdigest()[:16], 16) % 5 + 1
              for seller in train.seller_id.astype(str)]
    return pd.DataFrame({'id': train.id, 'seller_id': train.seller_id, 'fold': values})

def load_folds(train):
    output = OUTPUT_ROOT / 'shared' / 'folds.csv'
    output.parent.mkdir(exist_ok=True)
    if FOLDS_PATH is not None:
        fold_source = Path(FOLDS_PATH)
    elif Path('folds.csv').is_file():
        fold_source = Path('folds.csv')
    elif output.is_file():
        fold_source = output
    else:
        # Na Kaggle pierwszy notebook zapisze ten sam plik co w pakiecie.
        canonical_folds(train).to_csv(output, index=False, lineterminator='\n')
        fold_source = output
    table = pd.read_csv(fold_source)
    assert {'id', 'seller_id', 'fold'}.issubset(table.columns), 'Niepełny plik foldów.'
    assert table.id.is_unique and set(table.id) == set(train.id), 'Inne ID w foldach.'
    table = table.set_index('id').loc[train.id].reset_index()
    assert table.seller_id.tolist() == train.seller_id.tolist(), 'Inni sprzedawcy w foldach.'
    assert table.fold.isin([1, 2, 3, 4, 5]).all(), 'Nieprawidłowe numery foldów.'
    table['fold'] = table.fold.astype(int)
    assert table.groupby('seller_id').fold.nunique().eq(1).all(), 'Sprzedawca w kilku foldach.'
    assert table['fold'].tolist() == canonical_folds(train)['fold'].tolist(), 'Podział różni się od uzgodnionego startera.'
    encoded = table[['id', 'seller_id', 'fold']].to_csv(index=False, lineterminator='\n').encode()
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == EXPECTED_FOLDS_SHA256, 'Hash foldów nie pasuje do uzgodnionej edycji.'
    output.write_bytes(encoded)
    return table, digest



def features(df):
    x = df.copy()
    title = x.title.fillna('').astype(str).str.lower()
    desc = x.description.fillna('').astype(str).str.lower()
    x['text'] = title + ' ' + desc
    x['title_length'] = title.str.len()
    x['description_length'] = desc.str.len()
    # Missing observations remain missing; imputation is learned within each fold.
    for col in ['price_pln', 'shipping_weight_kg', 'pack_size']:
        x['log_' + col] = np.log1p(x[col].clip(lower=0))
    x['price_per_item'] = np.log1p(x.price_pln / x.pack_size.clip(lower=1))
    x['mass_per_item'] = np.log1p(x.shipping_weight_kg / x.pack_size.clip(lower=1))
    # Starter hint, not a guaranteed main-item count: free accessories and words
    # require a better parser. Evaluate improvements using held-out sellers.
    extracted = x.text.str.extract(r'(\d+)\s*pcs', expand=False)
    x['text_pack'] = pd.to_numeric(extracted, errors='coerce')
    x['pack_difference'] = (x.text_pack - x.pack_size).abs()
    # Useful text-category interactions; no vocabulary or statistics learned here.
    x['category_text'] = [
        ' '.join(f'{cat}__{token}' for token in re.findall(r'\w+', text))
        for cat, text in zip(x.declared_category.fillna('missing'), x.text)
    ]
    return x.replace({pd.NA: np.nan})

import numpy as np
import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

NUMBER_V2 = r'(?:\d+|one|two|three|four|five)'
WORD_V2 = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
QUANTITY_PATTERNS_V2 = [
    rf'\b({NUMBER_V2})\s+(?:individually\s+boxed\s+)?main\s+items?\b',
    rf'\bpack\s+of\s+({NUMBER_V2})\b',
    rf'\bpackage\s+contains\s+({NUMBER_V2})\s*(?:pcs|pieces|items?|units?)\b',
    rf'\bbundle\s+quantity\s*:\s*({NUMBER_V2})\s*(?:units?|pcs|pieces)?\b',
    rf'\bincluded\s+main\s+items?\s*:\s*({NUMBER_V2})\b',
    rf'\b({NUMBER_V2})\s*-\s*pack\b',
    rf'\bquantity\s*:?\s*({NUMBER_V2})\b',
]


def counts_v2(value):
    text = '' if pd.isna(value) else str(value).lower()
    parts = re.split(r'[.;!?\n]', text)
    # Ostrożna zasada: fragment o gratisach nie służy do liczenia produktów.
    clean = ' '.join(p for p in parts
                     if not re.search(r'\b(?:free|bonus|accessor\w*)\b', p))
    matches = [m for pattern in QUANTITY_PATTERNS_V2
               for m in re.findall(pattern, clean)]
    if not matches:
        matches = re.findall(rf'\b({NUMBER_V2})\s*(?:pcs|pieces|units?)\b', clean)
    values = {WORD_V2[m] if m in WORD_V2 else int(m) for m in matches}
    return {v for v in values if v > 0}


def rich_features(frame):
    x = features(frame)
    records = []
    for title, desc in zip(frame.title, frame.description):
        a, b = counts_v2(title), counts_v2(desc)
        both = a | b
        records.append({
            'quantity_hint_v2': float(next(iter(both))) if len(both) == 1 else np.nan,
            'quantity_title_v2': float(next(iter(a))) if len(a) == 1 else np.nan,
            'quantity_desc_v2': float(next(iter(b))) if len(b) == 1 else np.nan,
            'quantity_conflict_v2': float(len(both) > 1),
            'quantity_missing_v2': float(len(both) == 0),
        })
    q = pd.DataFrame(records, index=frame.index)
    for col in q:
        x[col] = q[col]
    x['text_pack'] = x.quantity_hint_v2
    x['pack_difference'] = (x.text_pack - x.pack_size).abs()
    x['pack_ratio_v2'] = x.pack_size / x.quantity_hint_v2
    x['price_per_text_item'] = np.log1p(x.price_pln / x.quantity_hint_v2)
    x['weight_per_text_item'] = np.log1p(x.shipping_weight_kg / x.quantity_hint_v2)
    x['log_price_weight_ratio'] = np.log1p(x.price_pln / x.shipping_weight_kg)
    x['title_missing'] = frame.title.isna().astype(float)
    x['description_missing'] = frame.description.isna().astype(float)
    x['promotion_mentioned'] = x.text.str.contains(
        r'\b(?:promotion|clearance|discount|sale)\b', regex=True).astype(float)
    x['accessories_mentioned'] = x.text.str.contains(
        r'\b(?:free|bonus|accessor\w*)\b', regex=True).astype(float)
    for variant in ['compact', 'standard', 'large']:
        x['variant_' + variant] = x.text.str.contains(
            rf'\b{variant}\b', regex=True).astype(float)
    # Kod produktu z tytułu nie jest potrzebny do uczenia słownika.
    x['semantic_text'] = x.text.str.replace(r'\b[a-z]\d{3}\b', ' ', regex=True)
    return x.replace([np.inf, -np.inf], np.nan)


RICH_NUMERIC = [
    'log_price_pln', 'log_shipping_weight_kg', 'log_pack_size',
    'price_per_item', 'mass_per_item', 'text_pack', 'pack_difference',
    'title_length', 'description_length', 'quantity_title_v2',
    'quantity_desc_v2', 'quantity_conflict_v2', 'quantity_missing_v2',
    'pack_ratio_v2', 'price_per_text_item', 'weight_per_text_item',
    'log_price_weight_ratio', 'title_missing', 'description_missing',
    'promotion_mentioned', 'accessories_mentioned',
    'variant_compact', 'variant_standard', 'variant_large',
]
RICH_CATEGORICAL = ['declared_category', 'brand', 'condition']


def make_tree_model():
    prep = ColumnTransformer([
        ('numeric', 'passthrough', RICH_NUMERIC),
        ('categories', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), RICH_CATEGORICAL),
        ('text_topics', Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                      max_features=20000, sublinear_tf=True,
                                      dtype=np.float32)),
            ('svd', TruncatedSVD(n_components=64, n_iter=5, random_state=42)),
        ]), 'semantic_text'),
    ], sparse_threshold=0)
    cat_indices = list(range(len(RICH_NUMERIC), len(RICH_NUMERIC) + len(RICH_CATEGORICAL)))
    return Pipeline([
        ('preprocess', prep),
        ('model', HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=25, l2_regularization=2.0,
            categorical_features=cat_indices, early_stopping=False,
            random_state=42,
        )),
    ])

def main():
    data_dir = find_data()
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    assert hashlib.sha256((data_dir / 'train.csv').read_bytes()).hexdigest() == EXPECTED_TRAIN_SHA256, 'Inna edycja treningu.'
    assert train.target.isin([0, 1]).all()
    assert train.id.is_unique and test.id.is_unique
    assert set(train.id).isdisjoint(test.id)
    assert set(train.seller_id).isdisjoint(test.seller_id)
    assert sample.columns.tolist() == ['id', 'target']
    assert sample.id.tolist() == test.id.tolist()
    fold_table, fold_sha256 = load_folds(train)
    fold_ids = fold_table.fold.to_numpy()
    splits = [(np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold))
              for fold in range(1, 6)]
    y = train.target.to_numpy()
    for fit, valid in splits:
        assert len(fit) and len(valid)
        assert set(train.iloc[fit].seller_id).isdisjoint(train.iloc[valid].seller_id)
        assert len(np.unique(y[valid])) == 2 and len(np.unique(y[fit])) == 2
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                'scipy': scipy.__version__, 'threadpoolctl': threadpoolctl.__version__}
    print('Dane:', train.shape, test.shape)
    print('Foldy:', fold_sha256)
    print('Wersje:', versions)

    started = time.perf_counter()
    raw = train.drop(columns='target')
    with threadpool_limits(limits=4):
        x = rich_features(raw)
        xt = rich_features(test)
        oof = np.full(len(train), np.nan)
        predictions = np.zeros(len(test))
        fold_scores = []
        for fold, (fit, valid) in enumerate(splits, 1):
            model = make_tree_model()
            model.fit(x.iloc[fit], y[fit])
            oof[valid] = model.predict_proba(x.iloc[valid])[:, 1]
            predictions += model.predict_proba(xt)[:, 1] / len(splits)
            auc = float(roc_auc_score(y[valid], oof[valid]))
            fold_scores.append(auc)
            print(f'Fold {fold}: ROC-AUC={auc:.6f}', flush=True)
            del model

    assert np.isfinite(oof).all() and np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
    oof_auc = float(roc_auc_score(y, oof))
    print('OOF ROC-AUC:', round(oof_auc, 6))
    print('Average Precision:', round(float(average_precision_score(y, oof)), 6))
    print('Czas [s]:', round(time.perf_counter() - started, 2))

    # W dotychczasowym notebooku ta kopia powstała przed utworzeniem tego pliku.
    reference_path = OUTPUT_ROOT / 'stage2' / 'submission_tree_v2.csv'
    reproduction = {'status': 'reference_not_available', 'tolerance': 1e-6}
    if reference_path.is_file():
        reference = pd.read_csv(reference_path)
        assert reference.id.is_unique and set(reference.id) == set(sample.id)
        reference = reference.set_index('id').loc[sample.id]
        assert np.isfinite(reference.target).all()
        difference = float(np.max(np.abs(reference.target.to_numpy() - predictions)))
        reproduction.update(max_absolute_difference=difference,
                            status='passed' if difference <= 1e-6 else 'failed')
        print('Maksymalna różnica względem poprzedniego zgłoszenia:', difference)
        if difference > 1e-6:
            raise ValueError('Wynik nie odtwarza referencji w tolerancji 1e-6. '
                             'Sprawdź wersje bibliotek i dane. Poprzedni plik nie został nadpisany.')
    else:
        print('Brak wcześniejszego pliku do porównania — zapisujemy nowe wyniki.')

    submission = sample[['id']].copy()
    submission['target'] = predictions
    submission_path = OUTPUT_ROOT / 'submission.csv'
    submission.to_csv(submission_path, index=False, float_format='%.12f')
    check = pd.read_csv(submission_path)
    assert check.columns.tolist() == ['id', 'target']
    assert check.id.is_unique and check.id.tolist() == sample.id.tolist()
    assert len(check) == len(test) and np.isfinite(check.target).all()
    assert check.target.between(0, 1).all()
    assert np.max(np.abs(check.target.to_numpy() - predictions)) < 1e-10

    oof_frame = pd.DataFrame({'id': train.id, 'seller_id': train.seller_id,
                              'fold': fold_ids, 'target': y, 'prediction': oof})
    oof_frame.to_csv(OUTPUT_ROOT / 'oof_tree_v2.csv', index=False, float_format='%.12f')
    manifest = {
        'configuration': 'tree_v2', 'seed': SEED, 'thread_limit': 4,
        'versions': versions, 'folds_sha256': fold_sha256,
        'fold_policy': 'SHA256(42:seller_id), first 16 hex digits, modulo 5, stored as 1..5',
        'data_sha256': {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                        for name in ['train.csv', 'test.csv', 'sample_submission.csv']},
        'fold_roc_auc': fold_scores, 'oof_roc_auc': oof_auc,
        'oof_average_precision': float(average_precision_score(y, oof)),
        'independent_final_estimate': False,
        'inference': 'Mean of five fold models; no full-data refit.',
        'tree_parameters': make_tree_model().named_steps['model'].get_params(),
        'tfidf': {'ngram_range': [1, 2], 'min_df': 3, 'max_features': 20000,
                  'sublinear_tf': True, 'dtype': 'float32'},
        'svd': {'n_components': 64, 'n_iter': 5, 'random_state': SEED},
        'elapsed_seconds': time.perf_counter() - started,
        'reproduction_check': reproduction,
        'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        'kaggle_submission_reference': None,
    }
    (OUTPUT_ROOT / 'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    import zipfile
    handoff_path = OUTPUT_ROOT / 'handoff_tree_v2.zip'
    with zipfile.ZipFile(handoff_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for local, stored in [
            (OUTPUT_ROOT / 'oof_tree_v2.csv', 'oof_tree_v2.csv'),
            (OUTPUT_ROOT / 'shared/folds.csv', 'folds.csv'),
            (OUTPUT_ROOT / 'model_manifest.json', 'model_manifest.json'),
        ]:
            archive.write(local, arcname=stored)
    print('\nGOTOWE — Model Notebook')
    print('Zgłoszenie:', submission_path)
    print('OOF dla zespołu:', OUTPUT_ROOT / 'oof_tree_v2.csv')
    print('Wspólne foldy:', OUTPUT_ROOT / 'shared' / 'folds.csv')
    print('Kontrola odtwarzania:', reproduction['status'])
    print('Pakiet dla drugiej osoby:', handoff_path)

if __name__ == '__main__':
    main()
