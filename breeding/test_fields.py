"""Self-test: every field accepts its own gold answer in several phrasings and rejects a wrong one."""
import fields as F

bad = 0
for field in [f for f in F.FIELDS if f not in F.NEW_FIELDS]:
    for level in (1, 2, 3, 4):
        for p in F.make(field, level, 40, seed=0):
            gold = p.get("sol", p["gold"])
            if field == "div":
                q, r = p["gold"].split(" r ")
                goods = [f"{q} remainder {r}", f"**{q} remainder {r}**", f"quotient {q}, remainder {r}"]
            elif field == "date":
                import datetime
                d = datetime.date.fromisoformat(gold)
                goods = [f"{d.strftime('%B')} {d.day}, {d.year}", f"**{d.strftime('%B')} {d.day}, {d.year}**",
                         f"{d.strftime('%b')} {d.day} {d.year}", gold]
            elif field == "clock":
                goods = [gold, f"**{gold}**", gold.lstrip("0") if not gold.startswith("00") else gold]
            elif field == "base":
                goods = [gold, gold.upper()] + ([f"0b{gold}"] if p["kind"] == "bin" else [])
            elif field == "sort":
                goods = [gold, gold.replace(",", ", ")]
            else:
                goods = [gold, f"{int(gold):,}" if gold.lstrip("-").isdigit() else gold]
            for g in goods:
                ok, readable = F.check(p, f"some working...\n\nAnswer: {g}")
                if not ok:
                    bad += 1
                    print("REJECTED GOOD", field, level, repr(g), "gold", p["gold"])
            wrong = {"date": "January 1, 1999", "clock": "25:99", "div": "0 r 0"}.get(field, "7")
            if wrong != p["gold"] and F.check(p, f"Answer: {wrong}")[0]:
                bad += 1
                print("ACCEPTED WRONG", field, level, wrong, p["gold"])
            if F.check(p, "no answer line here")[0]:
                bad += 1
                print("ACCEPTED MISSING", field, level)
print("example questions:")
for field in [f for f in F.FIELDS if f not in F.NEW_FIELDS]:
    print(f"--- {field}\n{F.make(field, 2, 1, seed=1)[0]['q']}")
print("FAILURES", bad)
