
from dittodub_longform.text import normalize, number, ordinal, year, zh_number  # noqa: E402

EN = [
    ("In 1999, Dr. Smith paid $3.50 for 2 tickets at 7:30 p.m. on Jan. 5th.",
     "In nineteen ninety-nine, Doctor Smith paid three dollars and fifty cents for two tickets at seven thirty p.m. "
     "on January fifth."),
    ("The 3rd version weighs 2.5kg and costs £1,200, about 1/3 of the price.",
     "The third version weighs two point five kilograms and costs one thousand two hundred pounds, about one third "
     "of the price."),
    ("Call me at 555-0142. The team arrived at 9am, and 15% of them were tired.",
     "Call me at five five five, zero one four two. The team arrived at nine a.m., and fifteen percent of them "
     "were tired."),
    ("Mr. and Mrs. Lee moved to 42 St. James St. in the 1990s.",
     "Mister and Missus Lee moved to forty-two Saint James Street in the nineteen nineties."),
    ("It costs $5 million, or $0.99 each, and €1.",
     "It costs five million dollars, or ninety-nine cents each, and one euro."),
    ("Pages 10-20 cover 2005 to 2024; see No. 7.",
     "Pages ten to twenty cover two thousand five to twenty twenty-four; see number seven."),
    ("At 10:00 it was 72°F, or 22°C.", "At ten o'clock it was seventy-two degrees Fahrenheit, or twenty-two degrees Celsius."),
    ("She ran 1 km in 3/4 of an hour, on 3/5/2024.",
     "She ran one kilometer in three quarters of an hour, on March fifth, twenty twenty-four."),
    ("He scored 1,234,567 points, 0.5 more than me.",
     "He scored one million two hundred thirty-four thousand five hundred sixty-seven points, zero point five more "
     "than me."),
    ("Rock & roll in the '80s, e.g. the 21st century.",
     "Rock and roll in the eighties, for example, the twenty-first century."),
    ("A plain sentence stays exactly the same.", "A plain sentence stays exactly the same."),
    ("He was born in 1066. It hit -5°C; prices ran $200-$300. Call 911. The ratio is 3:2.",
     "He was born in ten sixty-six. It hit minus five degrees Celsius; prices ran two hundred dollars to three "
     "hundred dollars. Call nine one one. The ratio is three to two."),
    ("[Laughter] That is 2 funny!", "[Laughter] That is two funny!"),
]

ZH = [
    ("我们在2024年3月5日下午3:30见面，价格是12.5元，涨了15%。",
     "我们在二零二四年三月五日下午三点三十分见面，价格是十二点五元，涨了百分之十五。"),
    ("他买了2个苹果，花了¥108。", "他买了两个苹果，花了一百零八元。"),
    ("全市人口约1,050,000人，电话13812345678。", "全市人口约一百零五万人，电话一三八一二三四五六七八。"),
    ("清晨的菜市场总是最热闹的地方。", "清晨的菜市场总是最热闹的地方。"),
    ("每天走1.5公里，花了2小时，第1名。", "每天走一点五公里，花了两小时，第一名。"),
]


def test_numbers():
    assert number(0) == "zero"
    assert number(115) == "one hundred fifteen"
    assert number(2000305) == "two million three hundred five"
    assert year(1905) == "nineteen oh five" and year(1900) == "nineteen hundred" and year(2010) == "twenty ten"
    assert ordinal(1) == "first" and ordinal(22) == "twenty-second" and ordinal(40) == "fortieth" and ordinal(112) == "one hundred twelfth"
    assert zh_number(10) == "十" and zh_number(15) == "十五" and zh_number(105) == "一百零五"
    assert zh_number(10010) == "一万零一十" and zh_number(1050000) == "一百零五万" and zh_number(200000000) == "二亿"


def test_sentences():
    bad = []
    for src, want in EN:
        got = normalize(src, "en")
        if got != want:
            bad.append((src, got, want))
    for src, want in ZH:
        got = normalize(src, "zh")
        if got != want:
            bad.append((src, got, want))
    assert not bad, "\n".join(f"{s}\n  got  {g}\n  want {w}" for s, g, w in bad)


if __name__ == "__main__":
    test_numbers()
    test_sentences()
    print("ok")


def test_takes():
    from dittodub_longform import _units, takes
    para = " ".join(["\"Go now,\" she said. It is late and the road is long."] * 12)  # ~150 words
    out = takes(para + "\n\nShort one.")
    assert all(_units(t) <= 130 for t in out) and len(out) >= 2
    assert " ".join(out).replace("  ", " ") == para + " Short one."  # nothing lost or reordered, quotes kept
    assert takes("Hello there. How are you?") == ["Hello there. How are you?"]
