"""Fields (careers) for the breeding experiment: problem generators, prompts, answer parsing, exact checkers.

Every field has difficulty levels 1..4. A problem is a dict with "field", "level", "q" (the question text),
"gold" (the canonical answer, already normalised) and whatever the checker needs.
All answers are requested as a last line "Answer: ...".
"""
import datetime
import random
import re

TAIL = ' Work it out step by step, then write the final answer on the last line as "Answer: {fmt}".'
ANS_RE = re.compile(r"Answer:\s*\**\s*(.+?)\s*\**\s*$", re.M)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def last_answer(text):
    m = ANS_RE.findall(text)
    return m[-1].strip().strip("*`$ .") if m else None


def to_int(s):
    if s is None:
        return None
    s = re.sub(r"[,\s]", "", s)
    m = re.fullmatch(r"-?\d+", s)
    return int(s) if m else None


# ------------------------------------------------------------------ generators
def gen_mult(rng, level):
    a, b = {1: (2, 2), 2: (3, 3), 3: (4, 3), 4: (4, 4)}[level]
    x = rng.randint(10 ** (a - 1), 10 ** a - 1)
    y = rng.randint(10 ** (b - 1), 10 ** b - 1)
    return {"q": f"What is {x} × {y}?" + TAIL.format(fmt="<number>"), "gold": str(x * y)}


def gen_div(rng, level):
    nd, dd = {1: (3, 1), 2: (4, 2), 3: (5, 2), 4: (6, 3)}[level]
    d = rng.randint(max(2, 10 ** (dd - 1)), 10 ** dd - 1)
    n = rng.randint(10 ** (nd - 1), 10 ** nd - 1)
    q, r = divmod(n, d)
    return {"q": f"Divide {n} by {d}. What are the quotient and the remainder?"
                 + TAIL.format(fmt="<quotient> remainder <remainder>"), "gold": f"{q} r {r}"}


def gen_modpow(rng, level):
    bmax, emax, mmax = {1: (9, 6, 13), 2: (20, 12, 31), 3: (50, 25, 97), 4: (99, 60, 997)}[level]
    b, e, m = rng.randint(2, bmax), rng.randint(3, emax), rng.randint(5, mmax)
    return {"q": f"What is {b}^{e} mod {m}?" + TAIL.format(fmt="<number>"), "gold": str(pow(b, e, m))}


def gen_base(rng, level):
    hi = {1: 63, 2: 1023, 3: 65535, 4: 2 ** 24 - 1}[level]
    n = rng.randint(hi // 4 + 1, hi)
    kind = rng.choice(["d2b", "h2d", "d2h"])
    if kind == "d2b":
        return {"q": f"Convert the decimal number {n} to binary." + TAIL.format(fmt="<binary digits>"),
                "gold": bin(n)[2:], "kind": "bin"}
    if kind == "h2d":
        return {"q": f"Convert the hexadecimal number {n:X} to decimal." + TAIL.format(fmt="<number>"),
                "gold": str(n), "kind": "dec"}
    return {"q": f"Convert the decimal number {n} to hexadecimal." + TAIL.format(fmt="<hex digits>"),
            "gold": f"{n:x}", "kind": "hex"}


def gen_date(rng, level):
    span = {1: 10, 2: 45, 3: 200, 4: 1500}[level]
    start = datetime.date(2000, 1, 1) + datetime.timedelta(days=rng.randint(0, 365 * 50))
    k = rng.randint(2, span)
    end = start + datetime.timedelta(days=k)
    return {"q": f"What is the date {k} days after {start.strftime('%B')} {start.day}, {start.year}?"
                 + TAIL.format(fmt="<Month> <day>, <year>"), "gold": end.isoformat()}


def gen_sort(rng, level):
    n, hi = {1: (5, 99), 2: (8, 999), 3: (12, 9999), 4: (16, 99999)}[level]
    xs = [rng.randint(0, hi) for _ in range(n)]
    return {"q": f"Sort these numbers from smallest to largest: {', '.join(map(str, xs))}."
                 + TAIL.format(fmt="<numbers separated by commas>"), "gold": ",".join(map(str, sorted(xs)))}


def gen_letters(rng, level):
    words = ["banana", "parallel", "mississippi", "committee", "bookkeeper", "assessment", "possession",
             "referee", "coffee", "balloon", "occurrence", "necessary", "embarrass", "address", "succeed",
             "letter", "pepper", "tomorrow", "giraffe", "accessory", "difference", "millennium", "aggressive"]
    n = {1: 1, 2: 2, 3: 3, 4: 5}[level]
    sent = " ".join(rng.choice(words) for _ in range(n))
    ch = rng.choice(sorted(set(sent.replace(" ", ""))))
    return {"q": f'How many times does the letter "{ch}" appear in: "{sent}"?' + TAIL.format(fmt="<number>"),
            "gold": str(sent.count(ch))}


def gen_code(rng, level):
    """A small straight-line or looping Python program; the answer is what it prints (one integer)."""
    steps = {1: 3, 2: 5, 3: 7, 4: 9}[level]
    loop = level >= 2
    lines = [f"x = {rng.randint(1, 9)}", f"y = {rng.randint(1, 9)}"]
    ops = ["x = x + y", "y = y * 2", "x = x - 3", "y = y + x", "x = x * 3 % 17", "y = (y - x) % 11",
           "x, y = y, x", "x = x // 2 + y"]
    body = [rng.choice(ops) for _ in range(steps)]
    if loop:
        k = rng.randint(2, 3 if level < 4 else 4)
        lines.append(f"for i in range({k}):")
        lines += ["    " + b for b in body[: steps // 2]]
        lines += body[steps // 2:]
    else:
        lines += body
    lines.append("print(x + y)")
    src = "\n".join(lines)
    out = {}
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, {}, out)  # generated here from a fixed list of statements, never from model output
    return {"q": "What does this Python program print?\n\n```python\n" + src + "\n```\n"
                 + TAIL.format(fmt="<the printed number>").strip(), "gold": buf.getvalue().strip()}


def gen_clock(rng, level):
    span = {1: 59, 2: 400, 3: 3000, 4: 20000}[level]
    h, m, k = rng.randint(0, 23), rng.randint(0, 59), rng.randint(5, span)
    t = (h * 60 + m + k) % (24 * 60)
    return {"q": f"A 24-hour clock shows {h:02d}:{m:02d}. What time does it show {k} minutes later?"
                 + TAIL.format(fmt="HH:MM"), "gold": f"{t // 60:02d}:{t % 60:02d}"}


FIELDS = {
    "mult": gen_mult, "div": gen_div, "modpow": gen_modpow, "base": gen_base,
    "date": gen_date, "sort": gen_sort, "letters": gen_letters, "code": gen_code, "clock": gen_clock,
}
HELD_OUT = {"code", "clock"}  # never trained by anyone: they measure generality


# ------------------------------------------------------------------ checking
def normalise(field, p, ans):
    """Model's answer string -> canonical string comparable with p['gold'], or None if unreadable."""
    if ans is None:
        return None
    a = ans.lower()
    if field in ("mult", "modpow", "letters", "code"):
        v = to_int(ans)
        return None if v is None else str(v)
    if field == "div":
        ints = re.findall(r"\d[\d,]*", a)
        return f"{to_int(ints[0])} r {to_int(ints[1])}" if len(ints) == 2 else None
    if field == "base":
        a = re.sub(r"[\s_]", "", a)
        if p["kind"] == "bin":
            a = a.removeprefix("0b").lstrip("0") or "0"
            return a if re.fullmatch(r"[01]+", a) else None
        if p["kind"] == "hex":
            a = a.removeprefix("0x").lstrip("0") or "0"
            return a if re.fullmatch(r"[0-9a-f]+", a) else None
        v = to_int(a)
        return None if v is None else str(v)
    if field == "date":
        m = re.search(r"([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", a)
        if not m:
            m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", a)
            return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}" if m2 else None
        months = [datetime.date(2000, i, 1).strftime("%B").lower() for i in range(1, 13)]
        mon = next((i + 1 for i, nm in enumerate(months) if nm.startswith(m.group(1)[:3])), None)
        try:
            return datetime.date(int(m.group(3)), mon, int(m.group(2))).isoformat() if mon else None
        except ValueError:
            return None
    if field == "clock":
        m = re.search(r"(\d{1,2}):(\d{2})", a)
        return f"{int(m.group(1)) % 24:02d}:{m.group(2)}" if m else None
    if field == "sort":
        nums = [to_int(x) for x in re.split(r"[,\s]+", ans.strip()) if x]
        return None if any(v is None for v in nums) or not nums else ",".join(map(str, nums))
    raise KeyError(field)


def check(p, text):
    """(ok, readable) for a model's full response text."""
    ans = last_answer(text)
    n = normalise(p["field"], p, ans)
    return n == p["gold"], n is not None


def make(field, level, n, seed):
    rng = random.Random(f"{field}-{level}-{seed}")
    out = []
    for i in range(n):
        p = FIELDS[field](rng, level)
        p.update(field=field, level=level, pid=f"{field}{level}-{seed}-{i}")
        out.append(p)
    return out


def answer_of(p, text):
    """The model's final answer, normalised (None if unreadable). Two models agree if these are equal."""
    ans = last_answer(text)
    return normalise(p["field"], p, ans)
