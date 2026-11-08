"""v0.24 P0: immutable, strictly chronological six-of-49 temporal dataset gate.

CSV history is authoritative for dates.  An explicitly requested catalog-only
mode permits research on frozen P1 draws, but records dates as unavailable.
Neither mode claims independent official-source verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import csv
import re

import numpy as np

SCHEMA = 'lotto7x7.temporal_dataset.v0.24'
YEAR_RE = re.compile(r'^lotto649_(\d{4})\.csv$')
CSV_FIELDS = ('date','draw_id','n1','n2','n3','n4','n5','n6','bonus')


@dataclass(frozen=True)
class TemporalDataset:
    ids: np.ndarray
    numbers: np.ndarray
    images: np.ndarray
    dates: np.ndarray | None
    sources: dict


def file_hash(path: Path) -> str:
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_arrays(ids: np.ndarray, numbers: np.ndarray,
                    dates: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reject duplicates, chronology conflicts, illegal rows; do not silently sort."""
    a = np.asarray(ids)
    b = np.asarray(numbers)
    if (a.ndim != 1 or b.shape != (len(a), 6)
            or len(a) < 3 or not np.issubdtype(a.dtype,np.integer)
            or not np.issubdtype(b.dtype,np.integer)):
        raise ValueError('expected 3+ ordered draw IDs and integer Nx6 numbers')
    if np.any(a < 1000000) or np.any(np.diff(a.astype('int64')) <= 0):
        raise ValueError('duplicated or out-of-order draw IDs')
    if np.any(b < 1) or np.any(b > 49):
        raise ValueError('six-number labels must be 1..49')
    b = np.sort(b.astype(np.uint8), axis=1)
    if np.any(np.diff(b.astype('int16'), axis=1) <= 0):
        raise ValueError('six-number labels must be distinct')
    d = None
    if dates is not None:
        d = np.asarray(dates)
        if d.shape != (len(a),) or not np.issubdtype(d.dtype,np.integer):
            raise ValueError('expected integer dates per draw')
        if np.any(np.diff(d.astype('int64')) <= 0):
            raise ValueError('duplicated or unordered chronological dates')
        for draw_id, day in zip(a,d):
            try:
                parsed = datetime.strptime(str(int(day)), '%Y%m%d')
            except ValueError as e:
                raise ValueError(f'invalid Gregorian date {day}') from e
            if parsed.year != int(draw_id)//1000000 + 1911:
                raise ValueError(f'draw ID/ROC year conflicts with date: {draw_id} / {day}')
    return a.astype(np.int64), b, (d.astype(np.int64) if d is not None else None)


def rasterize(numbers: np.ndarray) -> np.ndarray:
    """Sequential label mapping: number n corresponds to raster cell n-1."""
    a = np.asarray(numbers)
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError('expected Nx6')
    masks = np.zeros((len(a),49), dtype=np.uint8)
    masks[np.arange(len(a))[:,None], a.astype(np.intp)-1] = 1
    return masks.reshape(-1,7,7)


def load_csv_history(csv_root: Path, *, image_bin: Path | None = None,
                     catalog: Path | None = None) -> TemporalDataset:
    root = Path(csv_root)
    files = sorted(x for x in root.glob('lotto649_*.csv') if YEAR_RE.fullmatch(x.name))
    if not files:
        raise ValueError(f'no flat annual Lotto649 CSVs in {root}')
    years = [int(YEAR_RE.fullmatch(x.name).group(1)) for x in files]
    if years != list(range(years[0], years[-1]+1)):
        raise ValueError('missing or duplicate calendar-year CSV')
    rows = []
    for file, year in zip(files, years):
        with file.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not set(CSV_FIELDS).issubset(reader.fieldnames):
                raise ValueError(f'{file}: missing required header')
            found = 0
            for row in reader:
                origin = f'{file}:{reader.line_num}'
                try:
                    date = datetime.strptime(row['date'].strip(),'%Y/%m/%d')
                    did = int(row['draw_id'].strip())
                    nums = [int(row[f'n{i}'].strip()) for i in range(1,7)]
                    bonus = int(row['bonus'].strip())
                except (TypeError,ValueError,KeyError) as e:
                    raise ValueError(f'{origin}: invalid date/ID/number') from e
                if date.year != year or did//1000000 != year-1911:
                    raise ValueError(f'{origin}: draw ID/year/date mismatch')
                if not 1 <= bonus <= 49 or bonus in nums:
                    raise ValueError(f'{origin}: invalid bonus')
                rows.append((did, int(date.strftime('%Y%m%d')), nums))
                found += 1
            if not found:
                raise ValueError(f'{file}: no records')
    # Do not sort: a reversed record or file is an integrity problem.
    ids,nums,dates = validate_arrays(np.array([r[0] for r in rows],dtype=np.int64),
                                     np.array([r[2] for r in rows],dtype=np.int64),
                                     np.array([r[1] for r in rows],dtype=np.int64))
    images = rasterize(nums)
    evidence = {'mode':'csv','date_verification':'checked_from_csv',
                'annual_file_count':len(files), 'files_sha256':{str(p):file_hash(p) for p in files},
                'origin_note':'Repository CSV was exported from one third-party Biga archive; no independent official lottery verification.'}
    if image_bin is not None:
        # Existing immutable BIN v2 schema has verified fixed records and six-pixel invariants.
        from .bin_dataset import decode_bin
        blob = Path(image_bin).read_bytes()
        rec = decode_bin(blob)
        if not np.array_equal(rec['draw_id'], ids) or not np.array_equal(rec['image'],images):
            raise ValueError('BIN v2 images/IDs differ from annual CSV source')
        evidence['image_bin_sha256']=sha256(blob).hexdigest()
    if catalog is not None:
        with np.load(catalog,allow_pickle=False) as z:
            if not {'draw_ids','numbers'} <= set(z.files):
                raise ValueError('P1 catalog missing draw IDs / numbers')
            cat_ids=np.asarray(z['draw_ids'],dtype=np.int64)
            cat_nums=np.sort(np.asarray(z['numbers'],dtype=np.uint8),axis=1)
            if not np.array_equal(ids,cat_ids) or not np.array_equal(nums,cat_nums):
                raise ValueError('P1 catalog mismatches CSV draw IDs / six numbers')
        evidence['catalog_sha256']=file_hash(catalog)
    return TemporalDataset(ids,nums,images,dates,evidence)


def load_catalog_history(catalog: Path) -> TemporalDataset:
    """Explicit research-only mode when trustworthy dates are unavailable."""
    with np.load(catalog,allow_pickle=False) as z:
        if not {'draw_ids','numbers'} <= set(z.files):
            raise ValueError('P1 catalog missing expected arrays')
        ids,nums,_=validate_arrays(z['draw_ids'],z['numbers'])
    return TemporalDataset(ids,nums,rasterize(nums),None,{
        'mode':'catalog_only','catalog_sha256':file_hash(catalog),
        'date_verification':'UNAVAILABLE (ROC draw ID chronological order only)',
        'origin_note':'Catalog is a frozen third-party source; not a new official-source audit.'})


def chronological_split(n: int, *, minimum_training: int = 100,
                        warmup: int = 21) -> dict[str,int]:
    if n < max(3*minimum_training,warmup+minimum_training+20):
        raise ValueError('not enough draws for leakage-safe chronological split')
    train=int(n*0.6);val=int(n*0.8)
    if not (warmup < train < val < n) or train-warmup<minimum_training:
        raise ValueError('invalid chronological train/validation/test boundaries')
    return {'warmup':int(warmup),'train_end':int(train),
            'validation_end':int(val),'test_end':int(n),
            'train_target_range': [int(warmup+1),int(train-1)],
            'validation_target_range':[int(train),int(val-1)],
            'test_target_range':[int(val),int(n-1)]}


def dataset_report(data: TemporalDataset) -> dict:
    return {'schema':SCHEMA,'status':('PASS' if data.dates is not None else 'PARTIAL_NO_DATE_VERIFICATION'),'draws':len(data.ids),
            'first_id':int(data.ids[0]),'last_id':int(data.ids[-1]),
            'first_date':int(data.dates[0]) if data.dates is not None else None,
            'last_date':int(data.dates[-1]) if data.dates is not None else None,
            'unique_ticket_count':len(set(map(tuple,data.numbers.tolist()))),
            'image_pixels_per_draw':6,
            'splits':chronological_split(len(data.ids)),
            'source':data.sources}
