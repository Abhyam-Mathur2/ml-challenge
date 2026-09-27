import re
import unicodedata

LEGAL = {
    "corporation": "corp", "corporate": "corp", "corp.": "corp",
    "incorporated": "inc", "inc.": "inc",
    "limited": "ltd", "ltd.": "ltd",
    "private": "pvt", "pvt.": "pvt",
    "company": "co", "co.": "co",
    "llc": "llc", "l.l.c": "llc",
}

ADDRESS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "highway": "hwy", "lane": "ln", "drive": "dr", "parkway": "pkwy",
    "apartment": "apt", "building": "bldg", "floor": "fl",
}


def unicode_clean(s):
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).casefold()
    s = s.replace("&", " and ")
    s = s.replace("’", "'")
    s = s.replace("-", " ")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(s):
    x = unicode_clean(s)
    toks = [LEGAL.get(t, t) for t in x.split()]
    return " ".join(toks)


def normalize_address(s):
    x = unicode_clean(s)
    toks = [ADDRESS.get(t, t) for t in x.split()]
    return " ".join(toks)


def normalize_country(s):
    return unicode_clean(s)


def compact(value):
    value = unicode_clean(value)
    return value.replace(" ", "")


def add_normalized(df):
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_compact"] = out["name_norm"].map(compact)
    out["name_tokens"] = out["name_norm"].map(lambda value: tuple(value.split()))
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["address_compact"] = out["address_norm"].map(compact)
    out["address_tokens"] = out["address_norm"].map(lambda value: tuple(value.split()))
    out["country_norm"] = out["country"].map(normalize_country)
    return out
