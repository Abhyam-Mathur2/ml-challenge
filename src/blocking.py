from collections import defaultdict
from itertools import combinations
import re
import unicodedata


LEGAL_TOKENS = {"corp", "inc", "ltd", "pvt", "co", "llc", "llp", "plc", "limited", "private", "company"}
ADDRESS_STOP_TOKENS = {"road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd", "lane", "ln", "drive", "dr", "floor", "fl", "building", "bldg"}
POSTING_LIMIT = 256
EXACT_POSTING_LIMIT = 1024


def _tokens(value):
    return {token for token in str(value or "").split() if token}


def _prefixes(value, width):
    return {token[:width] for token in _tokens(value) if len(token) >= width}


def _compact(value):
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(
        character for character in normalized
        if not unicodedata.combining(character) and character.isalnum()
    )


def _compact_prefixes(value, width):
    compact = _compact(value)
    return {compact[:width]} if len(compact) >= width else set()


def _significant_name_tokens(value):
    return {token for token in _tokens(value) if token not in LEGAL_TOKENS and len(token) >= 2}


def _address_numbers(value):
    numbers = set()
    for value_part in re.findall(r"\d+[a-z]?", str(value or "").casefold()):
        digits = re.match(r"\d+", value_part).group()
        suffix = value_part[len(digits):]
        numbers.add((digits.lstrip("0") or "0") + suffix)
    return numbers


def _significant_address_tokens(value):
    return {token for token in _tokens(value) if token not in ADDRESS_STOP_TOKENS and len(token) >= 3 and not token.isdigit()}


class TargetIndex:
    """Reusable inverted indexes for one target source."""

    def __init__(self, target, prefix_length=4, posting_limit=POSTING_LIMIT):
        self.target = target
        self.prefix_length = prefix_length
        self.posting_limit = posting_limit
        self.exact_name = defaultdict(list)
        self.exact_address = defaultdict(list)
        self.exact_name_country = defaultdict(list)
        self.exact_address_country = defaultdict(list)
        self.exact_compact_name = defaultdict(list)
        self.exact_compact_address = defaultdict(list)
        self.name_token = defaultdict(list)
        self.significant_name_token = defaultdict(list)
        self.significant_name_token_pair = defaultdict(list)
        self.address_token = defaultdict(list)
        self.address_number = defaultdict(list)
        self.address_number_token = defaultdict(list)
        self.country_name_token = defaultdict(list)
        self.country_address_token = defaultdict(list)
        self.name_prefix = defaultdict(list)
        self.address_prefix = defaultdict(list)
        self.compact_name_prefix = defaultdict(list)
        self.compact_address_prefix = defaultdict(list)
        self._disabled_postings = defaultdict(set)

        def add_posting(index_name, index, key, label, limit):
            if key in self._disabled_postings[index_name]:
                return
            postings = index[key]
            if len(postings) >= limit:
                del index[key]
                self._disabled_postings[index_name].add(key)
                return
            postings.append(label)

        for label, row in target[["name_norm", "address_norm", "country_norm"]].iterrows():
            name = row["name_norm"]
            address = row["address_norm"]
            country = row["country_norm"]
            if name:
                add_posting("exact_name", self.exact_name, name, label, EXACT_POSTING_LIMIT)
                add_posting("exact_name_country", self.exact_name_country, (name, country), label, EXACT_POSTING_LIMIT)
                add_posting("exact_compact_name", self.exact_compact_name, (country, _compact(name)), label, EXACT_POSTING_LIMIT)
            if address:
                add_posting("exact_address", self.exact_address, address, label, EXACT_POSTING_LIMIT)
                add_posting("exact_address_country", self.exact_address_country, (address, country), label, EXACT_POSTING_LIMIT)
                add_posting("exact_compact_address", self.exact_compact_address, (country, _compact(address)), label, EXACT_POSTING_LIMIT)
            for token in _tokens(name):
                add_posting("name_token", self.name_token, token, label, self.posting_limit)
                add_posting("country_name_token", self.country_name_token, (country, token), label, self.posting_limit)
            significant_tokens = sorted(_significant_name_tokens(name))
            for token in significant_tokens:
                add_posting("significant_name_token", self.significant_name_token, (country, token), label, self.posting_limit)
            for pair in combinations(significant_tokens[:6], 2):
                add_posting("significant_name_token_pair", self.significant_name_token_pair, (country, pair), label, self.posting_limit)
            for token in _tokens(address):
                add_posting("address_token", self.address_token, token, label, self.posting_limit)
                add_posting("country_address_token", self.country_address_token, (country, token), label, self.posting_limit)
            numbers = _address_numbers(address)
            address_tokens = sorted(_significant_address_tokens(address))[:6]
            for number in numbers:
                add_posting("address_number", self.address_number, (country, number), label, self.posting_limit)
                for token in address_tokens:
                    add_posting("address_number_token", self.address_number_token, (country, number, token), label, self.posting_limit)
            for prefix in _prefixes(name, prefix_length):
                add_posting("name_prefix", self.name_prefix, prefix, label, self.posting_limit)
            for prefix in _prefixes(address, prefix_length):
                add_posting("address_prefix", self.address_prefix, prefix, label, self.posting_limit)
            for prefix in _compact_prefixes(name, prefix_length + 2):
                add_posting("compact_name_prefix", self.compact_name_prefix, (country, prefix), label, self.posting_limit)
            for prefix in _compact_prefixes(address, prefix_length + 2):
                add_posting("compact_address_prefix", self.compact_address_prefix, (country, prefix), label, self.posting_limit)


def build_inverted_index(df, column, min_token_len=2):
    index = defaultdict(list)
    for label, value in df[column].items():
        for token in _tokens(value):
            if len(token) >= min_token_len:
                index[token].append(label)
    return index


def _take(postings, limit):
    return postings if limit is None else postings[:limit]


def generate_candidates(
    s1,
    target,
    max_candidates=80,
    tfidf_top_k=None,
    index=None,
    return_provenance=False,
    exact_limit=None,
    token_limit=20,
    prefix_limit=10,
    max_name_tokens=5,
    max_address_tokens=6,
    exact_family_fraction=0.20,
    name_family_fraction=0.25,
    address_family_fraction=0.25,
    number_family_fraction=0.20,
):
    """Generate bounded candidates without a target-wide similarity matrix."""
    del tfidf_top_k
    target_index = index or TargetIndex(target)
    if exact_limit is None:
        exact_limit = max_candidates

    result = {}
    provenance = {}
    for s1_label, row in s1[["name_norm", "address_norm", "country_norm"]].iterrows():
        name = row["name_norm"]
        address = row["address_norm"]
        country = row["country_norm"]

        family_caps = {
            "exact": max(1, int(max_candidates * exact_family_fraction)),
            "name": max(1, int(max_candidates * name_family_fraction)),
            "address": max(1, int(max_candidates * address_family_fraction)),
            "number": max(1, int(max_candidates * number_family_fraction)),
        }
        family_caps["prefix"] = max(1, max_candidates - sum(family_caps.values()))
        family_labels = {family: [] for family in family_caps}
        family_streams = {family: [] for family in family_caps}
        rules_by_label = defaultdict(list)

        def add(family, rule, postings, limit):
            values = _take(postings, limit)
            if values:
                family_streams[family].append((rule, values))

        if name:
            add("exact", "exact_name", target_index.exact_name.get(name, ()), exact_limit)
            add("name", "country_name", target_index.exact_name_country.get((name, country), ()), exact_limit)
            add("exact", "exact_compact_name", target_index.exact_compact_name.get((country, _compact(name)), ()), exact_limit)
        if address:
            add("exact", "exact_address", target_index.exact_address.get(address, ()), exact_limit)
            add("address", "country_address", target_index.exact_address_country.get((address, country), ()), exact_limit)
            add("exact", "exact_compact_address", target_index.exact_compact_address.get((country, _compact(address)), ()), exact_limit)

        name_tokens = sorted(
            _tokens(name), key=lambda token: len(target_index.name_token.get(token, ()))
        )[:max_name_tokens]
        for token in name_tokens:
            add("name", "name_token", target_index.name_token.get(token, ()), token_limit)
            add("name", "country_name", target_index.country_name_token.get((country, token), ()), token_limit)

        significant_tokens = sorted(
            _significant_name_tokens(name),
            key=lambda token: len(target_index.significant_name_token.get((country, token), ())),
        )[:max_name_tokens]
        for token in significant_tokens:
            add("name", "significant_name_token", target_index.significant_name_token.get((country, token), ()), token_limit)
        for pair in combinations(sorted(significant_tokens), 2):
            add("name", "significant_name_token_pair", target_index.significant_name_token_pair.get((country, pair), ()), token_limit)

        address_tokens = sorted(
            _tokens(address), key=lambda token: len(target_index.address_token.get(token, ()))
        )[:max_address_tokens]
        for token in address_tokens:
            add("address", "address_token", target_index.address_token.get(token, ()), token_limit)
            add("address", "country_address", target_index.country_address_token.get((country, token), ()), token_limit)

        numbers = _address_numbers(address)
        significant_address_tokens = sorted(_significant_address_tokens(address))[:max_address_tokens]
        for number in numbers:
            add("number", "address_number", target_index.address_number.get((country, number), ()), token_limit)
            for token in significant_address_tokens:
                add("number", "address_number_token", target_index.address_number_token.get((country, number, token), ()), token_limit)

        for prefix in _prefixes(name, target_index.prefix_length):
            add("prefix", "prefix_name", target_index.name_prefix.get(prefix, ()), prefix_limit)
        for prefix in _prefixes(address, target_index.prefix_length):
            add("prefix", "prefix_address", target_index.address_prefix.get(prefix, ()), prefix_limit)
        for prefix in _compact_prefixes(name, target_index.prefix_length + 2):
            add("prefix", "compact_name_prefix", target_index.compact_name_prefix.get((country, prefix), ()), prefix_limit)
        for prefix in _compact_prefixes(address, target_index.prefix_length + 2):
            add("prefix", "compact_address_prefix", target_index.compact_address_prefix.get((country, prefix), ()), prefix_limit)

        for family, streams in family_streams.items():
            cursors = [0] * len(streams)
            while len(family_labels[family]) < family_caps[family] and streams:
                progressed = False
                for stream_index, (rule, postings) in enumerate(streams):
                    cursor = cursors[stream_index]
                    if cursor >= len(postings):
                        continue
                    label = postings[cursor]
                    cursors[stream_index] += 1
                    progressed = True
                    if label not in family_labels[family]:
                        family_labels[family].append(label)
                    if rule not in rules_by_label[label]:
                        rules_by_label[label].append(rule)
                    if len(family_labels[family]) >= family_caps[family]:
                        break
                if not progressed:
                    break

        ordered = []
        for family in ("exact", "name", "address", "number", "prefix"):
            for label in family_labels[family]:
                if label not in ordered:
                    ordered.append(label)
        ordered = ordered[:max_candidates]
        result[s1_label] = ordered
        provenance[s1_label] = {label: tuple(rules_by_label[label]) for label in ordered}

    return (result, provenance) if return_provenance else result


def build_target_index(target, prefix_length=4, posting_limit=POSTING_LIMIT):
    return TargetIndex(target, prefix_length=prefix_length, posting_limit=posting_limit)