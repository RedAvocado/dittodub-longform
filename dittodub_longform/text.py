"""Text normalization for Ming 3.0 TTS (English and Mandarin).

Numbers, money, percentages, times, dates, units and common abbreviations become words, so the model never has to
guess how to read digits or symbols (raw "$3.50" or "15%" can come out garbled). Pure Python, no dependencies.
Text that needs nothing is returned unchanged.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- English numbers

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))


def _under_1000(n: int) -> str:
    words = []
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n or not words:
        words.append(_ONES[n])
    return " ".join(words)


def number(n: int) -> str:
    """123 -> 'one hundred twenty-three' (no 'and')."""
    if n < 0:
        return "minus " + number(-n)
    if n == 0:
        return "zero"
    words = []
    for value, name in _SCALES:
        if n >= value:
            words += [number(n // value), name]
            n %= value
    if n:
        words.append(_under_1000(n))
    return " ".join(words)


def year(n: int) -> str:
    """1999 -> 'nineteen ninety-nine', 2005 -> 'two thousand five', 1905 -> 'nineteen oh five'."""
    if 2000 <= n <= 2009 or n % 1000 == 0:
        return number(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{number(hi)} hundred"
    return f"{number(hi)} {'oh ' + _ONES[lo] if lo < 10 else number(lo)}"


_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def ordinal(n: int) -> str:
    words = number(n)
    head, sep, last = words.rpartition(" ") if "-" not in words.rsplit(" ", 1)[-1] else words.rpartition("-")
    if last in _ORD:
        last = _ORD[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s if c.isdigit())


def decimal(s: str) -> str:
    whole, _, frac = s.replace(",", "").partition(".")
    out = number(int(whole or 0))
    return f"{out} point {digits(frac)}" if frac else out


def _plural(word, n):
    return word if n == 1 else (word[:-1] + "ies" if word.endswith("y") and word != "day" else word + "s")


# ---------------------------------------------------------------- English rules

_CUR = {"$": ("dollar", "cent"), "£": ("pound", "penny"), "€": ("euro", "cent"), "¥": ("yen", None)}
_SCALE_WORDS = r"(thousand|million|billion|trillion)"
_UNITS = {"km": ("kilometer", "kilometers"), "kg": ("kilogram", "kilograms"), "cm": ("centimeter", "centimeters"),
          "mm": ("millimeter", "millimeters"), "mg": ("milligram", "milligrams"), "ml": ("milliliter", "milliliters"),
          "mph": ("mile per hour", "miles per hour"), "km/h": ("kilometer per hour", "kilometers per hour"),
          "kph": ("kilometer per hour", "kilometers per hour"), "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"),
          "oz": ("ounce", "ounces"), "ft": ("foot", "feet"), "°C": ("degree Celsius", "degrees Celsius"),
          "°F": ("degree Fahrenheit", "degrees Fahrenheit"), "°": ("degree", "degrees"),
          "GB": ("gigabyte", "gigabytes"), "MB": ("megabyte", "megabytes"), "TB": ("terabyte", "terabytes"),
          "kHz": ("kilohertz", "kilohertz"), "Hz": ("hertz", "hertz"), "kW": ("kilowatt", "kilowatts"),
          "g": ("gram", "grams")}
_MONTHS = "January February March April May June July August September October November December".split()
_MON_ABBR = {m[:3]: m for m in _MONTHS} | {"Sept": "September"}
_FRAC = {2: ("half", "halves"), 4: ("quarter", "quarters")}
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_TITLES = [(r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bMs\.", "Miz"), (r"\bDr\.(?=\s+[A-Z])", "Doctor"),
           (r"\bProf\.(?=\s+[A-Z])", "Professor"), (r"\bvs\.?(?=\s)", "versus"), (r"\be\.g\.,?", "for example,"),
           (r"\bi\.e\.,?", "that is,"), (r"\bNo\.\s*(?=\d)", "number ")]
# may end a sentence: keep the full stop when a capital letter or the end of the text follows
_ENDERS = [(r"\b([A-Z][a-z]+\s+)St\.", r"\1Street"), (r"\bSt\.(?=\s+[A-Z])", "Saint"), (r"\bAve\.", "Avenue"),
           (r"\bJr\.", "Junior"), (r"\bSr\.", "Senior"), (r"\betc\.", "et cetera")]


def _abbreviations(t):
    for pat, rep in _TITLES:
        t = re.sub(pat, rep, t)
    for pat, rep in _ENDERS:
        def sub(m, rep=rep):
            tail = m.group("tail")
            if tail is None:
                return m.expand(rep)
            return m.expand(rep) + ("" if rep == "Saint" else ".") + tail
        t = re.sub(pat + r"(?P<tail>\s+(?=[A-Z])|\s*$)?", sub, t)
    return t


def _money(m):
    sym, amount, scale = m.group(1), m.group(2).replace(",", ""), m.group(3)
    unit, sub = _CUR[sym]
    if scale:
        return f"{decimal(amount)} {scale} {unit}s" if sym != "¥" else f"{decimal(amount)} {scale} yen"
    whole, _, frac = amount.partition(".")
    w = int(whole or 0)
    parts = []
    if w or not frac or sub is None:
        parts.append(f"{number(w)} {unit if sym == '¥' else _plural(unit, w)}")
    if frac and sub is not None:
        c = int((frac + "0")[:2])
        if c:
            subw = "pence" if sub == "penny" and c != 1 else (sub if c == 1 else sub + "s")
            parts.append(f"{number(c)} {subw}")
    return " and ".join(parts)


def _time(m):
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    words = number(h) if mi == 0 else f"{number(h)} {'oh ' + _ONES[mi] if mi < 10 else number(mi)}"
    if ap:
        return f"{words} {'a.m.' if ap.lower().startswith('a') else 'p.m.'}"
    return words + (" o'clock" if mi == 0 else "")


def _frac(m):
    a, b = int(m.group(1)), int(m.group(2))
    if b in _FRAC:
        name = _FRAC[b][0 if a == 1 else 1]
    elif b == 1 or b > 20:
        return f"{number(a)} over {number(b)}"
    else:
        name = ordinal(b) + ("" if a == 1 else "s")
    return f"{number(a)} {name}"


def _unit(m):
    num = m.group(1)
    one, many = _UNITS[m.group(2)]
    return f"{decimal(num)} {one if num == '1' else many}"


def _decade(m):
    w = year(int(m.group(1)))
    return w[:-1] + "ies" if w.endswith("y") else w + "s"


def _date(m):
    mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    yy = int(y) + (2000 if len(y) == 2 else 0)
    return f"{_MONTHS[mo - 1]} {ordinal(d)}, {year(yy)}"


def _phone(m):
    return ", ".join(digits(g) for g in re.findall(r"\d+", m.group(0)))


def normalize_en(t: str) -> str:
    t = _abbreviations(t)
    t = re.sub(r"\b([A-Z][a-z]{2,3})\.(?=\s*\d)", lambda m: _MON_ABBR.get(m.group(1), m.group(0)), t)
    t = re.sub(r"(\S+)@(\S+)\.([a-z]{2,})\b", lambda m: f"{m.group(1)} at {m.group(2)} dot {m.group(3)}", t)
    t = re.sub(r"\(?\b\d{3}\)?[ -]\d{3}-\d{4}\b|\b\d{3}-\d{4}\b", _phone, t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", _date, t)
    t = re.sub(r"([$£€¥])(\s?(?:" + _NUM + r"))\s?[–-]\s?(?=[$£€¥]?\d)(?![$£€¥])", r"\1\2 to \1", t)  # $200-300
    t = re.sub(r"([$£€¥]\s?(?:" + _NUM + r"))\s?[–-]\s?(?=[$£€¥])", r"\1 to ", t)  # $200-$300
    t = re.sub(r"([$£€¥])\s?(" + _NUM + r")(?:\s?" + _SCALE_WORDS + r"\b)?", _money, t)
    t = re.sub(r"(" + _NUM + r")\s?%", lambda m: decimal(m.group(1)) + " percent", t)
    t = re.sub(r"(^|[\s(])[-−](?=\d)", r"\1minus ", t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?|[AP]\.?M\.?)?(?![\w])",
               lambda m: _time(m) + (" " if m.group(0).endswith(" ") else ""), t)
    t = re.sub(r"\b(\d{1,2})\s?([ap]\.?m\.?|[AP]\.?M\.?)(?![\w])",
               lambda m: f"{number(int(m.group(1)))} {'a.m.' if m.group(2)[0] in 'aA' else 'p.m.'}", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b((?:1[1-9]|20)\d0)s\b", _decade, t)
    t = re.sub(r"'(\d)0s\b", lambda m: _TENS[int(m.group(1))][:-1] + "ies" if int(m.group(1)) >= 2 else m.group(0), t)
    t = re.sub(r"(?<![\d.,$£€¥])\b(\d{1,3}(?:\.\d+)?|\d{1,3}(?:,\d{3})+)\s?(" +
               "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)) + r")(?![\w/])", _unit, t)
    t = re.sub(r"\b911\b", "nine one one", t)
    t = re.sub(r"\b(\d+):(\d)\b", lambda m: f"{m.group(1)} to {m.group(2)}", t)  # ratios such as 3:2
    t = re.sub(r"\b(\d+)/(\d+)\b", _frac, t)
    t = re.sub(r"\b(\d+)\s?[–-]\s?(\d+)\b", lambda m: f"{m.group(1)} to {m.group(2)}", t)
    t = re.sub(r"(?<![\d,.])\b(1\d\d\d|20\d\d)\b(?![,.]?\d)", lambda m: year(int(m.group(1))), t)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b", lambda m: decimal(m.group(0)), t)
    t = re.sub(r"\b([A-Za-z]+)(\d+)\b", r"\1 \2", t)  # Q3, mp3
    t = re.sub(r"\b0\d+\b", lambda m: digits(m.group(0)), t)
    t = re.sub(r"\b\d+\b", lambda m: number(int(m.group(0))), t)
    t = re.sub(r"\s*&\s*", " and ", t)
    t = re.sub(r"\s\+\s", " plus ", t)
    t = re.sub(r"\s=\s", " equals ", t)
    t = re.sub(r"#\s*(?=[a-z])", "number ", t)
    return t


# ---------------------------------------------------------------- Mandarin

_ZH_D = "零一二三四五六七八九"
_ZH_CLASSIFIERS = "个位只本次天张件条头份种名辆台杯碗双斤点岁周年间家层座首部句段米"
_ZH_MEASURES = ("小时", "分钟", "公里", "千米", "公斤", "千克", "星期", "个月")


def _zh_under_10000(n: int) -> str:
    out, zero = "", False
    for value, name in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // value % 10
        if d:
            if zero and out:
                out += "零"
            out += _ZH_D[d] + name
            zero = False
        else:
            zero = True
    return out


def zh_number(n: int) -> str:
    if n < 0:
        return "负" + zh_number(-n)
    if n == 0:
        return "零"
    out, prev = "", None
    for value, name in ((10 ** 8, "亿"), (10 ** 4, "万"), (1, "")):
        part = n // value % 10000 if value < 10 ** 8 else n // value
        if part:
            if prev is not None and part < 1000:
                out += "零"
            out += (zh_number(part) if part >= 10000 else _zh_under_10000(part)) + name
            prev = part
        elif out:
            prev = 0
    out = re.sub("零+", "零", out).rstrip("零")
    return out[1:] if out.startswith("一十") else out


def zh_decimal(s: str) -> str:
    whole, _, frac = s.replace(",", "").partition(".")
    out = zh_number(int(whole or 0))
    return out + ("点" + "".join(_ZH_D[int(c)] for c in frac) if frac else "")


def normalize_zh(t: str) -> str:
    t = re.sub(r"(\d{2,4})年", lambda m: "".join(_ZH_D[int(c)] for c in m.group(1)) + "年", t)
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: zh_number(int(m.group(1))) + "点" +
               ("整" if m.group(2) == "00" else ("零" if m.group(2)[0] == "0" else "") + zh_number(int(m.group(2))) + "分"), t)
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: "百分之" + zh_decimal(m.group(1)), t)
    t = re.sub(r"[¥￥]\s?(\d[\d,]*(?:\.\d+)?)", lambda m: zh_decimal(m.group(1)) + "元", t)
    t = re.sub(r"\d{7,}", lambda m: "".join(_ZH_D[int(c)] for c in m.group(0)), t)
    t = re.sub(r"(?<![\d.])2(?=[" + _ZH_CLASSIFIERS + "]|" + "|".join(_ZH_MEASURES) + ")", "两", t)
    t = re.sub(r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+|\d+", lambda m: zh_decimal(m.group(0)), t)
    return t


# ---------------------------------------------------------------- entry point

def normalize(text: str, language: str = "en") -> str:
    """Normalize one passage. language: 'en', 'zh', or anything else (returned with whitespace cleanup only)."""
    t = " ".join(text.split())
    if language == "zh" or re.search(r"[一-鿿]", t):
        return normalize_zh(t)
    if language in ("en", "english"):
        return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", normalize_en(t))).strip()
    return t
