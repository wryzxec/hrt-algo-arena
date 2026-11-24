"""Ridge on observable headline templates, event timing and price response."""
from collections import defaultdict
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# Stable fragments from the training partition, independent of company and region.
_CUES = {
    1: ("secures", "margin improvement", "wins industry award", "record quarterly",
        "share buyback", "increase in customer", "strategic acquisition", "breakthrough",
        "strong demand", "expands operations", "multi-year partnership", "joint venture",
        "launches next-generation", "raises full-year", "beats analyst", "achieves key"),
    -1: ("regulatory review", "recalls products", "delays product", "drop in new customer",
         "rising costs", "misses quarterly", "class action", "decline in operating",
         "warns of supply", "loses key", "unexpected decline", "steps down",
         "withdraws from", "restructuring plan", "lowers full-year"),
    0: ("names new head", "mixed results", "strategic alternatives", "revises long-term",
        "files for regulatory", "host investor", "routine patent", "scheduled maintenance",
        "planned facility", "capital expenditure", "opens new office", "confirms participation",
        "to present at", "addresses investor", "organizational", "in talks",
        "annual shareholder", "annual sustainability", "board meeting"),
}
_SIGNS = {cue: sign for sign, cues in _CUES.items() for cue in cues}
_TEMPLATE = re.compile(r"\b(?:" + "|".join(map(re.escape, _SIGNS)) + r")\b")
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)(%|m\b|b\b)")
_NEGATED = re.compile(r"\b(?:not|never|no|denies|without)\b(?:\s+[\w-]+){0,2}\s*$")


def _features(frame):
    rows = []
    for row in frame[["headlines", "headline_bars", "close"]].itertuples(index=False):
        features = defaultdict(float)
        features["news count"] = 0
        latest = -1
        for bar, headline in zip(row.headline_bars, row.headlines):
            if not 0 <= bar <= 49:
                continue
            text = headline.lower()
            match = _TEMPLATE.search(text)
            cue = match.group() if match else "other"
            sign = _SIGNS.get(cue, 0)
            if match and _NEGATED.search(text[:match.start()]):
                sign = -sign
            age = 49 - bar
            response = row.close[49] / row.close[bar] - 1
            features["news count"] += 1
            features[f"count/{cue}"] += 1
            for half_life in (5, 15):
                weight = 2 ** (-age / half_life)
                features[f"recent{half_life}/{cue}"] += weight
                features[f"signed{half_life}"] += sign * weight
            features[f"response/{cue}"] += response
            number = _NUMBER.search(text)
            if number:
                value, unit = number.groups()
                # Percent in percentage points, dollar amounts in millions.
                magnitude = float(value) * (1000 if unit == "b" else 1)
                scale = "percent" if unit == "%" else "million"
                features[f"magnitude/{cue}/{scale}"] += magnitude * 2 ** (-age / 15)
            if bar >= latest:
                latest, latest_cue = bar, cue
                features["latest response"] = response
                features["latest age"] = age
                features["latest sign"] = sign
        if latest >= 0:
            features[f"latest/{latest_cue}"] = 1
        rows.append(dict(features))
    return rows


def build():
    return make_pipeline(FunctionTransformer(_features), DictVectorizer(),
                         StandardScaler(with_mean=False), Ridge(alpha=500, solver="lsqr"))
