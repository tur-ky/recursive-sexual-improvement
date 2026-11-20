"""Fields (careers) for the breeding experiment: problem generators, prompts, answer parsing, exact checkers.

Every field has difficulty levels 1..4. A problem is a dict with "field", "level", "q" (the question text),
"gold" (the canonical answer, already normalised) and whatever the checker needs.
All answers are requested as a last line "Answer: ...".

The first nine fields are from Phase 0. Eleven more were added on 20 November (NEW_FIELDS; see NEWFIELDS.md), picked
for diversity: logic, state tracking, spatial reasoning, graph search, set algebra, algebra, probability, games, pattern
induction and a symbol system. For those, a final \boxed{...} also counts as the answer when there is no "Answer:" line;
the first nine keep their exact Phase 0 rule.
"""
import datetime
import heapq
import itertools
import random
import re
from fractions import Fraction
from functools import lru_cache

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


# ================================================================== fields added 20 November (NEWFIELDS.md)
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
ITEMS = ["key", "coin", "book", "lamp", "ring", "map", "cup", "pen"]
WORDS = ["apple", "river", "garden", "silver", "window", "planet", "forest", "candle", "bridge", "pocket", "winter",
         "yellow", "market", "rabbit", "doctor", "simple", "travel", "orange", "castle", "summer", "button", "kitten",
         "mirror", "pencil", "dragon", "letter", "jungle", "basket", "rocket", "violin", "carpet", "shadow", "people",
         "moment", "number", "family", "island", "school", "friend", "animal", "cloud", "stone", "music", "light",
         "water", "night", "green", "happy", "tiger", "bread", "camera", "dinner", "engine", "flower", "guitar",
         "harbor", "insect", "jacket", "ladder", "meadow", "napkin", "office", "parrot", "quartz", "ribbon", "saddle",
         "tunnel", "valley", "walnut", "zipper", "anchor", "blanket", "cactus", "donkey", "falcon", "goblet", "helmet",
         "lizard", "muffin", "nickel", "pepper", "puzzle", "rescue", "spider", "teapot", "unicorn", "velvet", "wizard",
         "beach", "chair", "dream", "grape", "honey", "lemon", "ocean", "piano", "snake", "train"]
NORM = {}


def field(name):
    def reg(fns):
        gen, norm = fns
        FIELDS[name] = gen
        NORM[name] = norm
        return fns
    return reg


def ints(s):
    return [int(x) for x in re.findall(r"(?<![\d.])-?\d+(?![\d.]*\d*/)", s.replace(",", " ").replace("−", "-"))]


# ------------------------------------------------------------------ knights and knaves (logic)
def _claims(rng, names, i, level):
    n = len(names)
    others = [j for j in range(n) if j != i]
    ref = lambda j: "I" if j == i else names[j]  # noqa: E731
    pool = ["knight", "knave", "same", "diff"] + (["both", "notboth"] if level >= 2 else []) + (
        ["count"] if level >= 4 else [])
    kind = rng.choice(pool)
    if kind in ("knight", "knave"):
        j = rng.choice(others)
        return (f"{names[j]} is a {kind}.", (lambda a, j=j: a[j]) if kind == "knight" else (lambda a, j=j: not a[j]))
    if kind == "count":
        k = rng.randint(0, n)
        return (f"Exactly {k} of us {'is a knight' if k == 1 else 'are knights'}.", lambda a, k=k: sum(a) == k)
    j, k = rng.sample(range(n), 2)  # may include the speaker ("Bob and I"): needed for any unique solution with 2
    if k == i:
        j, k = k, j
    if j == i:
        j, k = k, j  # "Bob and I ..."
    pair = f"{ref(j)} and {ref(k)}"
    return {"both": (f"{pair} are both knights.", lambda a, j=j, k=k: a[j] and a[k]),
            "notboth": (f"At least one of {pair.replace(' and I', ' and me')} is a knave.",
                        lambda a, j=j, k=k: not (a[j] and a[k])),
            "same": (f"{pair} are the same type.", lambda a, j=j, k=k: a[j] == a[k]),
            "diff": (f"{pair} are different types.", lambda a, j=j, k=k: a[j] != a[k])}[kind]


def gen_knights(rng, level):
    n = {1: 2, 2: 3, 3: 4, 4: 5}[level]
    names = rng.sample(NAMES, n)
    while True:
        claims = [_claims(rng, names, i, level) for i in range(n)]
        sols = [a for a in itertools.product([True, False], repeat=n)
                if all(a[i] == c[1](a) for i, c in enumerate(claims))]
        if len(sols) == 1:
            break
    sol = sols[0]
    says = " ".join(f'{nm} says: "{c[0]}"' for nm, c in zip(names, claims))
    fmt = ", ".join(f"{nm}: <knight or knave>" for nm in names)
    return {"q": "On an island, knights always tell the truth and knaves always lie. " + says
                 + " Who is a knight and who is a knave?" + TAIL.format(fmt=fmt),
            "gold": ",".join(f"{nm.lower()}={'knight' if t else 'knave'}" for nm, t in zip(names, sol)),
            "show": ", ".join(f"{nm}: {'knight' if t else 'knave'}" for nm, t in zip(names, sol)), "names": names}


def norm_knights(p, ans):
    a, out = ans.lower(), []
    for nm in p["names"]:
        m = re.search(rf"\b{nm.lower()}\b\W*(?:is\s+)?(?:an?\s+)?(knight|knave)", a)
        if not m:
            return None
        out.append(f"{nm.lower()}={m.group(1)}")
    return ",".join(out)


field("knights")((gen_knights, norm_knights))


# ------------------------------------------------------------------ boolean evaluation (logic, nesting)
def _bexpr(rng, depth, vs):
    if depth == 0:
        v = rng.choice(vs)
        return f"NOT {v}" if rng.random() < 0.3 else v
    if rng.random() < 0.2:
        return f"NOT ({_bexpr(rng, depth - 1, vs)})"
    return f"({_bexpr(rng, depth - 1, vs)} {rng.choice(['AND', 'OR'])} {_bexpr(rng, depth - 1, vs)})"


def gen_boolean(rng, level):
    nv, depth = {1: (2, 1), 2: (3, 2), 3: (4, 3), 4: (5, 4)}[level]
    vs = ["P", "Q", "R", "S", "T"][:nv]
    env = {v: rng.random() < 0.5 for v in vs}
    exprs = [_bexpr(rng, depth, vs) for _ in range(3)]
    vals = [eval(e.replace("AND", "and").replace("OR", "or").replace("NOT", "not"), {}, dict(env))  # generated here
            for e in exprs]
    given = ", ".join(f"{v} is {'true' if env[v] else 'false'}" for v in vs)
    listing = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(exprs))
    return {"q": f"Given that {given}, evaluate each expression:\n\n{listing}\n\n"
                 + TAIL.format(fmt="<true or false>, <true or false>, <true or false>").strip(),
            "gold": "".join("t" if x else "f" for x in vals),
            "show": ", ".join("true" if x else "false" for x in vals)}


def norm_boolean(p, ans):
    toks = re.findall(r"\b(true|false|t|f)\b", ans.lower())
    return "".join(t[0] for t in toks) if len(toks) == 3 else None


field("boolean")((gen_boolean, norm_boolean))


# ------------------------------------------------------------------ grid walking (spatial)
DIRS = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
ORDER = ["north", "east", "south", "west"]


def gen_grid(rng, level):
    n, relative = {1: (3, False), 2: (6, False), 3: (10, False), 4: (4, True)}[level]
    x, y = rng.randint(-5, 5), rng.randint(-5, 5)
    start, steps = (x, y), []
    face = 0
    for _ in range(n):
        k = rng.randint(1, 20 if level == 3 else 9)
        if relative:
            turn = rng.choice(["left", "right", "around", "none"])
            face = (face + {"left": -1, "right": 1, "around": 2, "none": 0}[turn]) % 4
            steps.append((f"turn {turn} and " if turn != "none" else "") + f"walk {k} steps forward")
        else:
            face = rng.randrange(4)
            steps.append(f"walk {k} steps {ORDER[face]}")
        dx, dy = DIRS[ORDER[face]]
        x, y = x + k * dx, y + k * dy
    intro = (f"You stand at ({start[0]}, {start[1]}) on a grid where east increases x and north increases y"
             + (", facing north" if relative else "") + ". You " + ", then ".join(steps) + ".")
    return {"q": intro + " What are your final coordinates?" + TAIL.format(fmt="(x, y)"), "gold": f"{x},{y}",
            "show": f"({x}, {y})"}


def norm_grid(p, ans):
    v = ints(ans)
    return f"{v[0]},{v[1]}" if len(v) == 2 else None


field("grid")((gen_grid, norm_grid))


# ------------------------------------------------------------------ shortest path (graph search)
def gen_path(rng, level):
    n, extra = {1: (4, 2), 2: (6, 4), 3: (8, 6), 4: (10, 9)}[level]
    nodes = [chr(65 + i) for i in range(n)]
    while True:
        order = rng.sample(nodes, n)
        w = {}
        for i in range(1, n):
            w[frozenset((order[i], rng.choice(order[:i])))] = rng.randint(1, 9)
        pairs = [frozenset(c) for c in itertools.combinations(nodes, 2) if frozenset(c) not in w]
        for e in rng.sample(pairs, min(extra, len(pairs))):
            w[e] = rng.randint(1, 9)
        s, t = rng.sample(nodes, 2)
        dist, heap = {s: 0}, [(0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for e, c in w.items():
                if u in e:
                    v = next(iter(e - {u}))
                    if d + c < dist.get(v, 1e9):
                        dist[v] = d + c
                        heapq.heappush(heap, (d + c, v))
        direct = w.get(frozenset((s, t)))
        if level == 1 or direct is None or direct > dist[t]:  # from level 2 the best route is never one road
            break
    roads = ", ".join(f"{a}-{b} {c}" for (a, b), c in sorted((tuple(sorted(e)), c) for e, c in w.items()))
    return {"q": f"Two-way roads connect some towns, with these lengths: {roads}. What is the length of the shortest "
                 f"route from {s} to {t}?" + TAIL.format(fmt="<number>"), "gold": str(dist[t]), "show": str(dist[t])}


def norm_int(p, ans):
    v = to_int(ans)
    if v is None:
        v = ints(ans)
        v = v[0] if len(v) == 1 else None
    return None if v is None else str(v)


field("path")((gen_path, norm_int))


# ------------------------------------------------------------------ who has what (state tracking)
def gen_story(rng, level):
    npl, nit, nev = {1: (3, 2, 3), 2: (4, 3, 6), 3: (5, 4, 10), 4: (6, 5, 16)}[level]
    names, items = rng.sample(NAMES, npl), rng.sample(ITEMS, nit)
    holder = {it: rng.choice(names) for it in items}
    lines = [f"{holder[it]} has the {it}." for it in items]
    for _ in range(nev):
        r = rng.random()
        holders = sorted(set(holder.values()))
        if level == 1 or r < 0.6:
            it = rng.choice(items)
            b = rng.choice([x for x in names if x != holder[it]])
            lines.append(f"{holder[it]} gives the {it} to {b}.")
            holder[it] = b
        elif r < 0.8:
            a = rng.choice(holders)
            b = rng.choice([x for x in names if x != a])
            lines.append(f"{a} and {b} swap everything they are holding.")
            holder = {it: (b if h == a else a if h == b else h) for it, h in holder.items()}
        else:
            a = rng.choice(holders)
            b = rng.choice([x for x in names if x != a])
            lines.append(f"{a} gives everything they are holding to {b}.")
            holder = {it: (b if h == a else h) for it, h in holder.items()}
    it = rng.choice(items)
    return {"q": " ".join(lines) + f" Who has the {it} now?" + TAIL.format(fmt="<name>"),
            "gold": holder[it].lower(), "show": holder[it], "names": names}


def norm_story(p, ans):
    found = [nm.lower() for nm in p["names"] if re.search(rf"\b{nm.lower()}\b", ans.lower())]
    return found[0] if len(found) == 1 else None


field("story")((gen_story, norm_story))


# ------------------------------------------------------------------ Caesar cipher (letters, not tokens)
def _shift(s, k):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.isalpha() else c for c in s)


SPACED_FMT = "<the letters, written apart in the same way>"


def _spaced(s):
    """Letters written apart, so each is its own token; " / " between words."""
    return " / ".join(" ".join(w) for w in s.split())


def gen_caesar(rng, level):
    """The base model scored 0 to 2% on whole words (it sees tokens, not letters; 20 November), so letters are written
    apart. Level 1: two words, shift 1. Level 2: shift 2 to 5. Level 3: shift 6 to 25. Level 4: three words, decode
    or encode."""
    nw = {1: 2, 2: 1, 3: 1, 4: 3}[level]
    k = {1: 1, 2: rng.randint(2, 5), 3: rng.randint(6, 25), 4: rng.randint(1, 25)}[level]
    plain = " ".join(rng.sample(WORDS, nw))
    rule = (f"a Caesar cipher that shifts every letter {k} place{'s' if k > 1 else ''} forward in the alphabet "
            f"(z wraps around to a). Letters are written apart, with / between words")
    if level == 4 and rng.random() < 0.5:
        return {"q": f"Encode this text with {rule}: {_spaced(plain)}" + TAIL.format(fmt=SPACED_FMT),
                "gold": _shift(plain, k).replace(" ", ""), "show": _spaced(_shift(plain, k))}
    return {"q": f"This text was encoded with {rule}: {_spaced(_shift(plain, k))}. Decode it."
                 + TAIL.format(fmt=SPACED_FMT), "gold": plain.replace(" ", ""), "show": _spaced(plain)}


def norm_caesar(p, ans):
    """Letters only: spacing and word breaks are not part of the answer."""
    return re.sub(r"[^a-z]", "", ans.lower()) or None


# Not registered: the base model scored 0 to 2% at every level on 20 November, with whole words, with letters written
# apart, and with spaced answers (it shifts letters correctly but garbles them when writing the word out). Like
# Countdown in Phase 0, a field under 15% gives answer-key self-training nothing to learn from. Replaced by sets.
# field("caesar")((gen_caesar, norm_caesar))


# ------------------------------------------------------------------ set operations (replaces Caesar)
SETOPS = {"∪": lambda a, b: a | b, "∩": lambda a, b: a & b, "−": lambda a, b: a - b}


def gen_sets(rng, level):
    size, hi, nsets = {1: (5, 15, 2), 2: (7, 25, 2), 3: (7, 30, 3), 4: (9, 40, 3)}[level]
    while True:
        S = {n: set(rng.sample(range(1, hi + 1), size)) for n in "ABC"[:nsets]}
        o1, o2, o3 = (rng.choice(list(SETOPS)) for _ in range(3))
        if level <= 2:
            expr, val = f"A {o1} B", SETOPS[o1](S["A"], S["B"])
        elif level == 3:
            x, y, z = rng.sample("ABC", 3)
            expr, val = f"({x} {o1} {y}) {o2} {z}", SETOPS[o2](SETOPS[o1](S[x], S[y]), S[z])
        else:
            x, y, z = rng.sample("ABC", 3)
            w = rng.choice([x, y])
            expr = f"({x} {o1} {y}) {o2} ({z} {o3} {w})"
            val = SETOPS[o2](SETOPS[o1](S[x], S[y]), SETOPS[o3](S[z], S[w]))
        if 1 <= len(val) <= 12:
            break
    given = "; ".join(f"{n} = {{{', '.join(map(str, sorted(s)))}}}" for n, s in S.items())
    out = ", ".join(map(str, sorted(val)))
    return {"q": f"Let {given}. Here ∪ is union, ∩ is intersection and − is set difference. List the elements of "
                 f"{expr} from smallest to largest." + TAIL.format(fmt="<numbers separated by commas>"),
            "gold": ",".join(map(str, sorted(val))), "show": out}


def norm_sets(p, ans):
    v = sorted(set(ints(ans.replace("{", " ").replace("}", " "))))
    return ",".join(map(str, v)) if v else None


field("sets")((gen_sets, norm_sets))


# ------------------------------------------------------------------ linear equations (algebra)
def _lin(coefs, vs, const):
    out = ""
    for c, v in zip(coefs, vs):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        out += (f"-{mag}{v}" if c < 0 else f"{mag}{v}") if not out else (f" - {mag}{v}" if c < 0 else f" + {mag}{v}")
    if const:
        out += f" - {-const}" if const < 0 else f" + {const}"
    return out or "0"


def gen_lineq(rng, level):
    if level <= 2:
        x = rng.randint(-12, 12)
        a, b = rng.choice([i for i in range(-9, 10) if i not in (0,)]), rng.randint(-20, 20)
        if level == 1:
            a = abs(a)
            eq = f"{_lin([a], ['x'], b)} = {a * x + b}"
        else:
            c = rng.choice([i for i in range(-9, 10) if i not in (0, a)])
            d = a * x + b - c * x
            eq = f"{_lin([a], ['x'], b)} = {_lin([c], ['x'], d)}"
        return {"q": f"Solve for x: {eq}" + TAIL.format(fmt="x = <number>"), "gold": str(x), "show": f"x = {x}",
                "vars": ["x"]}
    vs = ["x", "y"] if level == 3 else ["x", "y", "z"]
    lo = 6 if level == 3 else 5
    while True:
        sol = [rng.randint(-9, 9) for _ in vs]
        rows = [[rng.randint(-lo, lo) for _ in vs] for _ in vs]
        det = round(_det(rows))
        if det != 0 and all(any(r) for r in rows):
            break
    eqs = "\n".join(f"{_lin(r, vs, 0)} = {sum(c * s for c, s in zip(r, sol))}" for r in rows)
    fmt = ", ".join(f"{v} = <number>" for v in vs)
    return {"q": f"Solve this system of equations:\n\n{eqs}\n\n" + TAIL.format(fmt=fmt).strip(),
            "gold": ",".join(map(str, sol)), "show": ", ".join(f"{v} = {s}" for v, s in zip(vs, sol)), "vars": vs}


def _det(m):
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return sum((-1) ** j * m[0][j] * _det([r[:j] + r[j + 1:] for r in m[1:]]) for j in range(3))


def norm_lineq(p, ans):
    a = ans.replace("−", "-")
    named = [re.search(rf"\b{v}\s*=\s*(-?\d+)\b", a) for v in p["vars"]]
    if all(named):
        return ",".join(str(int(m.group(1))) for m in named)
    v = ints(a)
    return ",".join(map(str, v)) if len(v) == len(p["vars"]) else None


field("lineq")((gen_lineq, norm_lineq))


# ------------------------------------------------------------------ probability as exact fractions
def _bag(rng):
    cols = rng.sample(["red", "blue", "green", "white", "black"], 3)
    return {c: rng.randint(1, 6) for c in cols}


def _bagtext(bag):
    items = [f"{n} {c}" for c, n in bag.items()]
    return ", ".join(items[:-1]) + f" and {items[-1]} marbles"


SIDES = [4, 6, 8, 10, 12, 20]


def _die_event(rng, n):
    """A condition on one roll of an n-sided die: (text, predicate)."""
    k = rng.randint(2, n - 1)
    m = rng.choice([3, 4, 5])
    return rng.choice([(f"greater than {k}", lambda x, k=k: x > k), (f"at most {k}", lambda x, k=k: x <= k),
                       ("even", lambda x: x % 2 == 0), (f"a multiple of {m}", lambda x, m=m: x % m == 0)])


def gen_prob(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            n = rng.choice(SIDES)
            txt, pred = _die_event(rng, n)
            q = f"A fair {n}-sided die (faces 1 to {n}) is rolled once. What is the probability that the result is {txt}?"
            ans = Fraction(sum(pred(x) for x in range(1, n + 1)), n)
        else:
            bag = _bag(rng)
            c = rng.choice(list(bag))
            q, ans = f"A bag holds {_bagtext(bag)}. One marble is drawn at random. What is the probability that it is " \
                     f"{c}?", Fraction(bag[c], sum(bag.values()))
    elif level == 2:
        n, m = rng.choice(SIDES[:4]), rng.choice(SIDES[:4])
        rolls = [(a, b) for a in range(1, n + 1) for b in range(1, m + 1)]
        dice = f"a fair {n}-sided die and a fair {m}-sided die" if n != m else f"two fair {n}-sided dice"
        s, k = rng.randint(3, n + m - 1), rng.randint(1, min(n, m))
        txt, pred = rng.choice([(f"the sum is exactly {s}", lambda a, b: a + b == s),
                                (f"the sum is at least {s}", lambda a, b: a + b >= s),
                                (f"at least one die shows {k}", lambda a, b: k in (a, b)),
                                ("the two dice show the same number", lambda a, b: a == b)])
        q = f"You roll {dice} (faces numbered from 1). What is the probability that {txt}?"
        ans = Fraction(sum(pred(a, b) for a, b in rolls), len(rolls))
    elif level == 3:
        bag = _bag(rng)
        tot = sum(bag.values())
        c, d = rng.sample(list(bag), 2)
        if rng.random() < 0.5:
            q = (f"A bag holds {_bagtext(bag)}. Two marbles are drawn at random without replacement. What is the "
                 f"probability that both are {c}?")
            ans = Fraction(bag[c] * (bag[c] - 1), tot * (tot - 1))
        else:
            q = (f"A bag holds {_bagtext(bag)}. Two marbles are drawn at random without replacement. What is the "
                 f"probability that one is {c} and the other is {d}?")
            ans = Fraction(2 * bag[c] * bag[d], tot * (tot - 1))
    else:
        if rng.random() < 0.5:
            sides, n = rng.choice(SIDES), rng.randint(2, 4)
            face = rng.randint(1, sides)
            q = (f"A fair {sides}-sided die (faces 1 to {sides}) is rolled {n} times. What is the probability of "
                 f"getting at least one {face}?")
            ans = 1 - Fraction(sides - 1, sides) ** n
        else:
            bag = _bag(rng)
            tot = sum(bag.values())
            c = rng.choice([x for x in bag if bag[x] >= 2])  if any(v >= 2 for v in bag.values()) else rng.choice(list(bag))
            q = (f"A bag holds {_bagtext(bag)}. Three marbles are drawn at random without replacement. What is the "
                 f"probability that exactly two of them are {c}?")
            ans = Fraction(_comb(bag[c], 2) * (tot - bag[c]), _comb(tot, 3))
    if ans in (0, 1):
        return gen_prob(rng, level)
    return {"q": q + TAIL.format(fmt="<fraction in lowest terms, like 3/8>"), "gold": str(ans), "show": str(ans)}


def _comb(n, k):
    from math import comb
    return comb(n, k)


def norm_prob(p, ans):
    m = re.search(r"\\[dt]?frac\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}", ans) or re.search(r"(\d+)\s*/\s*(\d+)", ans)
    if not m or int(m.group(2)) == 0:
        return None
    return str(Fraction(int(m.group(1)), int(m.group(2))))


field("prob")((gen_prob, norm_prob))


# ------------------------------------------------------------------ tic-tac-toe (game search)
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _winner(b):
    for i, j, k in LINES:
        if b[i] != "." and b[i] == b[j] == b[k]:
            return b[i]
    return None


@lru_cache(maxsize=None)
def _value(b, who):
    """Minimax value for X (+1 win, 0 draw, -1 loss) with `who` to move."""
    w = _winner(b)
    if w:
        return 1 if w == "X" else -1
    if "." not in b:
        return 0
    vals = [_value(b[:i] + who + b[i + 1:], "O" if who == "X" else "X") for i in range(9) if b[i] == "."]
    return max(vals) if who == "X" else min(vals)


def _wins(b, who):
    return [i for i in range(9) if b[i] == "." and _winner(b[:i] + who + b[i + 1:]) == who]


def gen_ttt(rng, level):
    """X moves first; whoever's turn it is (by piece count) is "you". Level 1: your only immediate win. Level 2: the
    only block of the opponent's single threat. (Levels 3-4 of the game field are other games: tic-tac-toe has only
    about 140 positions with a unique forcing move and 36 with a unique non-losing move, too few for exam + training.)"""
    while True:
        b, who = ["."] * 9, "X"
        for _ in range(rng.randint(3, 6)):
            free = [i for i in range(9) if b[i] == "."]
            b[rng.choice(free)] = who
            who = "O" if who == "X" else "X"
            if _winner("".join(b)):
                break
        b = "".join(b)
        if _winner(b) or "." not in b:
            continue
        opp = "O" if who == "X" else "X"
        free = [i for i in range(9) if b[i] == "."]
        mine, theirs = _wins(b, who), _wins(b, opp)
        if level == 1 and len(mine) == 1:
            ask, ans = f"{who} can win immediately. Which cell should {who} play to win?", mine[0]
        elif level == 2 and not mine and len(theirs) == 1:
            ask, ans = (f"{who} cannot win this turn, and {opp} threatens to win. Which cell must {who} play to stop "
                        f"{opp}?"), theirs[0]
        else:
            continue
        break
    rows = "\n".join(" " + " | ".join(b[r * 3 + c] if b[r * 3 + c] != "." else str(r * 3 + c + 1) for c in range(3))
                     for r in range(3))
    cells = lambda s: ", ".join(str(i + 1) for i in range(9) if b[i] == s) or "none"  # noqa: E731
    listing = (f"X is on cells {cells('X')}; O is on cells {cells('O')}. The eight winning lines are 1-2-3, 4-5-6, "
               f"7-8-9, 1-4-7, 2-5-8, 3-6-9, 1-5-9 and 3-5-7.")
    return {"q": f"Tic-tac-toe (X moved first). Empty cells show their number (1 to 9). It is {who}'s turn.\n\n{rows}"
                 f"\n\n{listing} {ask}" + TAIL.format(fmt="<cell number>"), "gold": str(ans + 1),
            "show": str(ans + 1), "kind": "int"}


def gen_game(rng, level):
    """Find the winning move. 1-2: tic-tac-toe. 3: a take-away game (the only winning number of stones to take).
    4: three-heap Nim, positions with exactly one winning move."""
    if level <= 2:
        return gen_ttt(rng, level)
    if level == 3:
        k = rng.randint(3, 9)
        n = rng.choice([x for x in range(20, 151) if x % (k + 1)])
        return {"q": f"There are {n} stones in a pile. Two players take turns removing between 1 and {k} stones; "
                     f"whoever takes the last stone wins. You move first. How many stones must you take now to be "
                     f"sure of winning?" + TAIL.format(fmt="<number>"), "gold": str(n % (k + 1)),
                "show": str(n % (k + 1)), "kind": "int"}
    while True:
        h = [rng.randint(1, 15) for _ in range(3)]
        x = h[0] ^ h[1] ^ h[2]
        moves = [(i, h[i] - (h[i] ^ x)) for i in range(3) if x and (h[i] ^ x) < h[i]]
        if len(moves) == 1:
            break
    i, take = moves[0]
    heaps = ", ".join(f"heap {'ABC'[j]} has {h[j]}" for j in range(3))
    return {"q": f"Nim: there are three heaps of stones ({heaps}). Two players take turns; on a turn a player removes "
                 f"any number of stones (at least one) from a single heap. Whoever takes the last stone wins. It is "
                 f"your turn, and exactly one move guarantees a win. Which heap, and how many stones do you take?"
                 + TAIL.format(fmt="<heap letter> <number of stones>"), "gold": f"{'ABC'[i]}:{take}",
            "show": f"{'ABC'[i]} {take}", "kind": "nim"}


def norm_game(p, ans):
    if p["kind"] == "nim":
        m = re.findall(r"\b([ABC])\b", ans.upper().replace("HEAP", " "))
        v = [x for x in ints(ans) if x > 0]
        return f"{m[0]}:{v[0]}" if len(set(m)) == 1 and len(v) == 1 else None
    v = [x for x in ints(ans) if x >= 0]
    return str(v[0]) if len(v) == 1 else None


field("game")((gen_game, norm_game))


# ------------------------------------------------------------------ sequence rules (induction)
def gen_seq(rng, level):
    kind = rng.choice({1: ["arith"], 2: ["geom", "quad", "quad"], 3: ["fib", "interleave", "affine"],
                       4: ["diffgeom", "alt", "cubeish"]}[level])
    a = rng.randint(-20, 60) if kind in ("arith", "quad", "interleave", "cubeish") else rng.randint(1, 12)
    if kind == "arith":
        d = rng.choice([i for i in range(-9, 13) if i])
        xs = [a + d * i for i in range(6)]
    elif kind == "geom":
        r = rng.choice([2, 3, 4, 5, -2, -3])
        xs = [a * r ** i for i in range(6)]
    elif kind == "quad":
        b, c = rng.randint(1, 4), rng.randint(-5, 5)
        xs = [b * i * i + c * i + a for i in range(6)]
    elif kind == "fib":
        xs = [a, rng.randint(1, 9)]
        while len(xs) < 7:
            xs.append(xs[-1] + xs[-2])
    elif kind == "interleave":
        d1, d2, b = rng.randint(2, 7), rng.randint(-6, -1), rng.randint(20, 60)
        xs = [a + d1 * (i // 2) if i % 2 == 0 else b + d2 * (i // 2) for i in range(8)]
    elif kind == "affine":
        m, c = rng.choice([2, 3]), rng.randint(-3, 4)
        xs = [a]
        while len(xs) < 6:
            xs.append(m * xs[-1] + c)
    elif kind == "diffgeom":
        d, r = rng.randint(1, 4), rng.choice([2, 3])
        xs = [a]
        for i in range(6):
            xs.append(xs[-1] + d * r ** i)
    elif kind == "alt":
        p_, m = rng.randint(2, 9), rng.choice([2, 3])
        xs = [a]
        for i in range(6):
            xs.append(xs[-1] + p_ if i % 2 == 0 else xs[-1] * m)
    else:
        c = rng.randint(-4, 4)
        xs = [i ** 3 + c * i + a for i in range(1, 8)]
    shown, nxt = xs[:-1], xs[-1]
    return {"q": f"What is the next number in this sequence: {', '.join(map(str, shown))}, ...?"
                 + TAIL.format(fmt="<number>"), "gold": str(nxt), "show": str(nxt), "kind": kind}


field("seq")((gen_seq, norm_int))


# ------------------------------------------------------------------ Roman numerals (a symbol system)
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
         (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
ROMAN_RE = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")


def to_roman(n):
    out = ""
    for v, s in ROMAN:
        while n >= v:
            out += s
            n -= v
    return out


def gen_roman(rng, level):
    if level == 1:
        n = rng.randint(4, 499)
        return {"q": f"Convert the Roman numeral {to_roman(n)} to an ordinary number." + TAIL.format(fmt="<number>"),
                "gold": str(n), "show": str(n), "kind": "int"}
    if level == 2:
        n = rng.randint(100, 999)
        return {"q": f"Write {n} as a Roman numeral." + TAIL.format(fmt="<Roman numeral>"), "gold": to_roman(n),
                "show": to_roman(n), "kind": "roman"}
    if level == 3:
        n = rng.randint(1000, 3999)
        return {"q": f"Convert the Roman numeral {to_roman(n)} to an ordinary number." + TAIL.format(fmt="<number>"),
                "gold": str(n), "show": str(n), "kind": "int"}
    x, y = rng.randint(40, 1999), rng.randint(40, 1999)
    return {"q": f"Add the Roman numerals {to_roman(x)} and {to_roman(y)}. Give the sum as a Roman numeral."
                 + TAIL.format(fmt="<Roman numeral>"), "gold": to_roman(x + y), "show": to_roman(x + y),
            "kind": "roman"}


def norm_roman(p, ans):
    if p["kind"] == "int":
        return norm_int(p, ans)
    a = re.sub(r"[^A-Za-z]", "", ans).upper()
    return a if a and ROMAN_RE.fullmatch(a) else None


field("roman")((gen_roman, norm_roman))

NEW_FIELDS = list(NORM)


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
    if field in NORM:
        return NORM[field](p, ans)
    raise KeyError(field)


def boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1].strip() if depth == 0 else None


def answer_text(p, text):
    """The "Answer:" line, or for the 20 November fields only, a final \boxed{} when there is no such line. The base
    model answered algebra and probability in \boxed{} 50 to 75% of the time, which would have measured the answer
    format rather than the skill. The Phase 0 fields keep their exact rule, so earlier results stay comparable."""
    a = last_answer(text)
    return boxed(text) if a is None and p["field"] in NORM else a


def check(p, text):
    """(ok, readable) for a model's full response text."""
    n = normalise(p["field"], p, answer_text(p, text))
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
    return normalise(p["field"], p, answer_text(p, text))
