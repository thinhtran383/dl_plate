"""
Sua loi OCR bien so Viet Nam theo cau truc Thong tu 79/2024/TT-BCA.

Cau truc (khong dau gach):
  - Xe may / o to: RR + seri(1-2 ky tu) + 5 so thu tu  => 8-9 ky tu
  - Xe may dien:   RR + MD + 5 so                      => 9 ky tu (vd. 29MD86706)
  - Seri chuan:    chu + so 1-9 (vd. 29D1, 29Z1, 30A1)
  - Seri xe may:   20 chu cai gom ca Z (vd. 29Z, 29AA)
"""
from __future__ import annotations

import itertools
import re
from typing import FrozenSet, Iterable, List, Optional, Set

# Seri dang ky - chu cai dau (11 chu TT79 + bo sung xe may gom Z, N, P, S, T, U, V, X, Y)
_SERIES_LETTERS: FrozenSet[str] = frozenset('ABCDEFGHKLMNPSTUVXYZ')

# Seri xe may 2 chu (truoc 2025 / mot so loai dac biet)
_MOTO_SERIES_LETTERS: FrozenSet[str] = frozenset('ABCDEFGHKLMNPSTUVXYZ')

# Cap seri xe may bi cam
_FORBIDDEN_MOTO_PAIRS: FrozenSet[str] = frozenset({
    'CD', 'CT', 'DA', 'HC', 'LB', 'LD', 'MK',
})

# Ma vung dang ky xe (34 tinh sau sap nhap 7/2025 + ma cu pho bien)
_REGION_CODES: FrozenSet[str] = frozenset({
    '11', '12', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28',
    '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '43', '47', '48', '49',
    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
    '80', '81', '82', '83', '84', '85', '86', '88',
    '90', '92', '93', '94', '95', '97', '99',
})

# Nham ky tu OCR pho bien -> ky tu dung (theo vi tri)
_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6', 'B': '8',
}

_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '8': 'B',
}

# Goi y thay the khi tim kiem (BFS)
_DIGIT_ALTS: dict[str, tuple[str, ...]] = {
    '0': ('0', 'O', 'D', 'Q'),
    '1': ('1', 'I', 'L', '7'),
    '2': ('2', 'Z'),
    '3': ('3', '8'),
    '4': ('4', 'A'),
    '5': ('5', 'S'),
    '6': ('6', 'G'),
    '7': ('7', '1', 'Z'),
    '8': ('8', 'B', '6'),
    '9': ('9', 'G'),
    'G': ('6', '9', 'G'),
}

_SERIES_LETTER_ALTS: dict[str, tuple[str, ...]] = {
    'A': ('A', '4'),
    'B': ('B', '8', '6'),
    'C': ('C', 'G', '6'),
    'D': ('D', '0', 'O'),
    'E': ('E',),
    'F': ('F',),
    'G': ('G', '6', 'C'),
    'H': ('H',),
    'I': ('I', '1', 'L'),
    'K': ('K',),
    'L': ('L', '1', 'I'),
    'M': ('M', 'N', 'W'),
    'N': ('N', 'M'),
    'O': ('O', '0', 'D'),
    'P': ('P',),
    'S': ('S', '5'),
    'T': ('T', '7'),
    'U': ('U',),
    'V': ('V',),
    'W': ('W', 'M'),
    'X': ('X',),
    'Y': ('Y',),
    'Z': ('Z', '2'),
    '0': ('0', 'D', 'O'),
    '1': ('1',),
    '2': ('2', 'Z'),
    '5': ('5', 'S'),
    '6': ('6', 'G'),
    '8': ('8', 'B'),
}


def _clean(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper().replace('_', ''))


def _validate_series(series: str) -> bool:
    if not series:
        return False

    if series == 'MD':
        return True

    if len(series) == 2 and series[0] == 'M' and series[1] == 'D':
        return True

    # Seri chuan (Thong tu 79): 1 chu + 1 so — uu tien khi ky tu 2 la so
    if len(series) == 2 and series[1] in '123456789':
        return series[0] in _SERIES_LETTERS

    # Seri 1 chu (o to cu / mot so truong hop)
    if len(series) == 1 and series[0] in _SERIES_LETTERS:
        return True

    # Xe may 2 chu cai (chi khi ca hai deu la chu, vd. AA, BB)
    if (
        len(series) == 2
        and series[0].isalpha()
        and series[1].isalpha()
        and series[0] in _MOTO_SERIES_LETTERS
        and series[1] in _MOTO_SERIES_LETTERS
    ):
        return series not in _FORBIDDEN_MOTO_PAIRS

    return False


def validate_vn_plate_strict(
    plate: str,
    preferred_regions: Optional[Set[str]] = None,
) -> bool:
    plate = _clean(plate)
    if len(plate) not in (8, 9):
        return False

    region = plate[:2]
    if not region.isdigit() or region not in _REGION_CODES:
        return False
    if preferred_regions and region not in preferred_regions:
        return False

    tail = plate[-5:]
    if not tail.isdigit():
        return False

    series = plate[2:-5]
    return _validate_series(series)


def _quick_fix_by_position(plate: str) -> str:
    """Sua nhanh theo vi tri: 2 so dau, seri chu, 5 so cuoi."""
    if len(plate) < 8:
        return plate

    chars = list(plate)
    tail_len = 5
    series_len = len(chars) - 2 - tail_len
    if series_len < 1:
        return plate

    # Ma vung: bat buoc la so
    for i in range(2):
        ch = chars[i]
        if not ch.isdigit() and ch in _TO_DIGIT:
            chars[i] = _TO_DIGIT[ch]

    series_start = 2
    series_end   = 2 + series_len

    # Seri MD (xe may dien)
    if series_len >= 2 and ''.join(chars[series_start:series_start + 2]).upper() in ('MD', 'M0', 'M8'):
        if chars[series_start] in ('0', 'W'):
            chars[series_start] = 'M'
        second = chars[series_start + 1]
        if second in _TO_LETTER and second in ('D',):
            chars[series_start + 1] = 'D'
        elif second in _TO_DIGIT:
            chars[series_start + 1] = 'D' if second in ('0', 'O') else second

    # Seri chuan: chu + so (Z la seri hop le, khong doi thanh D)
    elif series_len == 2:
        ch0 = chars[series_start]
        ch1 = chars[series_start + 1]
        if ch0.isdigit() and ch0 in _TO_LETTER:
            chars[series_start] = _TO_LETTER[ch0]
        if not chars[series_start + 1].isdigit() and ch1 in _TO_DIGIT:
            chars[series_start + 1] = _TO_DIGIT[ch1]
    elif series_len == 1:
        ch0 = chars[series_start]
        if ch0.isdigit() and ch0 in _TO_LETTER:
            chars[series_start] = _TO_LETTER[ch0]

    # 5 so cuoi: bat buoc la so
    for i in range(series_end, len(chars)):
        ch = chars[i]
        if not ch.isdigit() and ch in _TO_DIGIT:
            chars[i] = _TO_DIGIT[ch]

    return ''.join(chars)


def _position_alternatives(plate: str, pos: int) -> tuple[str, ...]:
    ch = plate[pos]
    tail_len = 5
    series_len = len(plate) - 2 - tail_len

    if pos < 2:
        return _DIGIT_ALTS.get(ch, (ch,))

    if pos >= len(plate) - tail_len:
        return _DIGIT_ALTS.get(ch, (ch,))

    # Vung seri
    if series_len == 2 and pos == 2 and plate[2:4].upper() in ('MD', 'M0', 'M8', 'M6'):
        return _SERIES_LETTER_ALTS.get(ch, (ch,))
    if series_len == 2 and pos == 3 and plate[2] in ('M', 'N', 'W'):
        if ch in _DIGIT_ALTS:
            return ('D', '0', 'O') if ch in ('0', 'O', 'D') else _DIGIT_ALTS.get(ch, (ch,))
        return _SERIES_LETTER_ALTS.get(ch, (ch,))

    if series_len == 2 and pos == 2:
        return _SERIES_LETTER_ALTS.get(ch, (ch,))
    if series_len == 2 and pos == 3:
        return _DIGIT_ALTS.get(ch, (ch,))

    if series_len == 1 and pos == 2:
        return _SERIES_LETTER_ALTS.get(ch, (ch,))

    return (ch,)


def _search_corrections(
    plate: str,
    preferred_regions: Optional[Set[str]] = None,
    max_changes: int = 3,
) -> Optional[str]:
    """Tim bien so hop le gan nhat bang cach thu thay the ky tu nham."""
    plate = _clean(plate)
    if validate_vn_plate_strict(plate, preferred_regions=None):
        if not preferred_regions or plate[:2] in preferred_regions:
            return plate

    quick = _quick_fix_by_position(plate)
    if validate_vn_plate_strict(quick, preferred_regions=None):
        if not preferred_regions or quick[:2] in preferred_regions:
            return quick

    n = len(plate)
    if n not in (8, 9):
        return None

    best: Optional[tuple[int, int, int, str]] = None  # (region_penalty, series_penalty, edits, text)

    def _series_penalty(candidate: str) -> int:
        series = candidate[2:-5]
        if len(series) == 2 and series[1] in '123456789' and series[0] in _SERIES_LETTERS:
            return 0
        if series == 'MD':
            return 0
        return 1

    for n_changes in range(1, max_changes + 1):
        positions = list(range(n))
        for change_positions in itertools.combinations(positions, n_changes):
            option_lists = [
                _position_alternatives(plate, p) for p in change_positions
            ]
            for combo in itertools.product(*option_lists):
                chars = list(plate)
                for p, new_ch in zip(change_positions, combo):
                    chars[p] = new_ch
                candidate = ''.join(chars)
                if not validate_vn_plate_strict(candidate, preferred_regions=None):
                    continue

                edits = sum(
                    1 for a, b in zip(plate, candidate) if a != b
                )
                # Khong doi Z thanh chu khac (Z la seri hop le: 29Z, 29Z1...)
                z_penalty = sum(
                    2 for a, b in zip(plate, candidate)
                    if a == 'Z' and b != 'Z'
                )
                region_penalty = 0
                if preferred_regions and candidate[:2] not in preferred_regions:
                    region_penalty = 1

                score = (region_penalty, _series_penalty(candidate), edits + z_penalty)
                if best is None or score < (best[0], best[1], best[2]):
                    best = (region_penalty, _series_penalty(candidate), edits, candidate)

        if best is not None:
            return best[3]

    return None


def correct_vn_plate_text(
    text: str,
    preferred_regions: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Sua loi OCR bien so VN.
    Tra ve bien so da chuan hoa hoac None neu khong sua duoc.
    """
    cleaned = _clean(text)
    if not cleaned:
        return None

    pref: Optional[Set[str]] = None
    if preferred_regions:
        pref = {str(r).strip() for r in preferred_regions if str(r).strip()}

    # Thu nguyen ban
    if validate_vn_plate_strict(cleaned, preferred_regions=None):
        if not pref or cleaned[:2] in pref:
            return cleaned

    # Thu cat chuoi con (khi dinh tem khung)
    for length in (9, 8):
        if len(cleaned) < length:
            continue
        for start in range(len(cleaned) - length + 1):
            sub = cleaned[start:start + length]
            fixed = _search_corrections(sub, preferred_regions=pref)
            if fixed:
                return fixed

    return _search_corrections(cleaned, preferred_regions=pref)


def is_corrected_vn_plate(text: str, preferred_regions: Optional[Iterable[str]] = None) -> bool:
    corrected = correct_vn_plate_text(text, preferred_regions=preferred_regions)
    return corrected is not None
