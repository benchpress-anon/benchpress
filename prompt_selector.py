"""
prompt_selector.py - Extract optimal removal targets from NExT-QA questions.

For each video, analyzes all questions to find the entity whose removal
would make the most questions unanswerable (destroying the visual evidence
needed to answer). Outputs data.json for the batch inpainting pipeline.

Selection criteria:
  1. Subject frequency: entity most often the grammatical subject across questions
  2. Distinguishability: visual descriptor SAM3 can lock onto (color, clothing)
  3. Singularity: penalize plural/group references (can't target "people")
  4. Simplicity: final SAM3 prompt kept to ≤4 words

Usage:
    python prompt_selector.py                         # default: val_grouped.csv -> data.json
    python prompt_selector.py --input X.csv --output Y.json
"""

import argparse
import re
import csv
import json
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════════

ENTITY_NOUNS = {
    # People
    "man", "woman", "lady", "boy", "girl", "baby", "child", "kid",
    "person", "guy", "toddler", "infant",
    "mother", "father", "mom", "dad",
    "grandma", "grandmother", "grandpa", "grandfather",
    "cyclist", "driver", "player", "dancer", "singer", "skater",
    "teacher", "student", "referee", "goalkeeper",
    "cameraman", "spectator", "soldier", "chef", "waiter",
    "clown", "magician", "performer", "host",
    # Animals
    "dog", "cat", "horse", "bird", "monkey", "bear", "elephant",
    "rabbit", "fish", "hamster", "parrot", "puppy", "kitten",
    "chicken", "duck", "pig", "cow", "goat", "sheep", "lion",
    "tiger", "deer", "turtle", "lizard", "frog",
    "pug", "lioness", "penguin", "crab", "snake", "camel",
    # Vehicles / machines (removable foreground objects)
    "bus", "car", "truck", "bicycle", "bike", "motorcycle",
    "bulldozer", "tractor",
    # Roles with visual identity
    "rider", "climber", "swimmer", "runner", "gymnast",
    "animal",
}

PLURAL_TO_SINGULAR = {
    "men": "man", "women": "woman", "ladies": "lady",
    "boys": "boy", "girls": "girl", "babies": "baby",
    "children": "child", "kids": "kid", "people": "person",
    "guys": "guy", "toddlers": "toddler",
    "dogs": "dog", "cats": "cat", "horses": "horse", "birds": "bird",
    "monkeys": "monkey", "players": "player", "dancers": "dancer",
    "skaters": "skater", "adults": "person", "puppies": "puppy",
    "kittens": "kitten",
    "animals": "animal", "pugs": "pug", "lionesses": "lioness",
    "penguins": "penguin", "riders": "rider",
    "buses": "bus", "cars": "car", "trucks": "truck",
    "bicycles": "bicycle", "bikes": "bike",
    "bulldozers": "bulldozer",
}

ALL_ENTITY_WORDS = ENTITY_NOUNS | set(PLURAL_TO_SINGULAR.keys())

VISUAL_ADJS = {
    "white", "black", "red", "blue", "green", "yellow", "brown",
    "pink", "grey", "gray", "orange", "purple", "dark", "light",
    "shirtless", "bald", "small", "big", "little", "tall", "short",
    "old", "young", "older", "younger", "blonde", "male", "female",
    "striped", "spotted", "thin", "fat", "long", "curly",
}

CLOTHING = {
    "shirt", "dress", "vest", "jacket", "hat", "cap", "shorts",
    "pants", "skirt", "uniform", "stripes", "stripe", "top",
    "sweater", "hoodie", "coat", "suit", "tie", "scarf", "glasses",
    "sunglasses", "helmet", "apron", "jersey", "headband", "bandana",
    "mask", "gloves", "boots", "shoes", "socks", "tshirt",
}

DESCRIPTOR_PREPS = {"in", "with", "wearing"}
DESCRIPTOR_WORDS = VISUAL_ADJS | CLOTHING
QUANTITY_WORDS = {"two", "three", "four", "both", "several", "many", "few", "some"}
AUX_VERBS = {"did", "does", "do", "is", "are", "was", "were", "has", "have", "had"}


# ═══════════════════════════════════════════════════════════════════════
# Entity extraction
# ═══════════════════════════════════════════════════════════════════════

def _clean(tok):
    return tok.strip().rstrip(".,?!;:'\"").lstrip("'\"")


def find_entities(text):
    """Find all entity mentions in *text* with descriptor metadata."""
    tokens = text.lower().split()
    entities = []
    skip_until = -1

    for i, raw in enumerate(tokens):
        if i < skip_until:
            continue
        tok = _clean(raw)
        if tok not in ALL_ENTITY_WORDS:
            continue

        is_plural = tok in PLURAL_TO_SINGULAR
        singular = PLURAL_TO_SINGULAR.get(tok, tok)

        # ── Look backwards for visual adjectives / quantity ──
        pre_adjs = []
        has_quantity = False
        j = i - 1
        while j >= 0:
            prev = _clean(tokens[j])
            if prev in VISUAL_ADJS:
                pre_adjs.insert(0, prev)
                j -= 1
            elif prev in QUANTITY_WORDS:
                has_quantity = True
                j -= 1
            elif prev in ("the", "a", "an", "other", "another", "one"):
                j -= 1
                break
            else:
                break
        if has_quantity:
            is_plural = True

        # ── Look forwards for "in/with/wearing [descriptor]" ──
        descriptor = ""
        desc_key = ""
        end_idx = i + 1

        if i + 1 < len(tokens):
            nxt = _clean(tokens[i + 1])
            if nxt in DESCRIPTOR_PREPS:
                desc_parts = []
                k = i + 2
                while k < len(tokens) and k <= i + 5:
                    w = _clean(tokens[k])
                    if w in DESCRIPTOR_WORDS:
                        desc_parts.append(w)
                        k += 1
                    elif w == "the":
                        k += 1
                    else:
                        break
                if desc_parts:
                    descriptor = nxt + " " + " ".join(desc_parts)
                    desc_key = desc_parts[0]
                    end_idx = k

        # ── Build mention ──
        parts = pre_adjs + [tok]
        if descriptor:
            parts.append(descriptor)
        mention = " ".join(parts)

        if not desc_key and pre_adjs:
            desc_key = pre_adjs[0]

        entities.append({
            "mention": mention,
            "noun": tok,
            "singular": singular,
            "is_plural": is_plural,
            "position": i,
            "desc_key": desc_key,
        })
        skip_until = end_idx

    return entities


def extract_question_subject(question):
    """Return the first entity after the auxiliary verb (= grammatical subject)."""
    q = question.lower().strip()
    tokens = q.split()

    # Find auxiliary verb
    aux_pos = -1
    for i, tok in enumerate(tokens):
        if _clean(tok) in AUX_VERBS and i > 0:
            aux_pos = i
            break

    rest = " ".join(tokens[aux_pos + 1:]) if aux_pos >= 0 else q
    entities = find_entities(rest)
    return entities[0] if entities else None


# ═══════════════════════════════════════════════════════════════════════
# Per-video analysis
# ═══════════════════════════════════════════════════════════════════════

def _cluster_key(ent):
    return (ent["singular"], ent["desc_key"])


def _pick_prompt(mentions):
    """Choose the most specific, most frequent mention as the SAM3 prompt."""
    counter = Counter(e["mention"] for e in mentions)
    # Prefer mentions with descriptors
    specific = [(m, c) for m, c in counter.items() if len(m.split()) > 1]
    if specific:
        return max(specific, key=lambda x: x[1])[0]
    return counter.most_common(1)[0][0]


def analyze_video(qa_string):
    """
    Analyze a video's concatenated Q&As and return the best removal target.

    Returns dict or None.
    """
    qa_pairs = qa_string.split(" ||| ")

    subject_counts = Counter()           # cluster_key -> # questions as subject
    all_mentions = defaultdict(list)     # cluster_key -> entity dicts

    for qa in qa_pairs:
        m = re.match(r"\[(\w+)\]\s*(.+?)\s*->\s*(.+)", qa)
        if not m:
            continue
        _qtype, question, _answer = m.group(1), m.group(2), m.group(3)

        subj = extract_question_subject(question)
        if subj:
            key = _cluster_key(subj)
            subject_counts[key] += 1
            all_mentions[key].append(subj)

    if not subject_counts:
        return None

    # ── Merge bare-noun clusters into descriptor clusters ──
    # When "man" (generic) and "man in white" both exist, merge the bare
    # mentions into the descriptor cluster — they're likely the same entity.
    bare_keys = [k for k in subject_counts if not k[1]]
    for bk in bare_keys:
        noun = bk[0]
        # Find descriptor clusters for the same noun
        desc_keys = [k for k in subject_counts if k[0] == noun and k[1]]
        if len(desc_keys) == 1:
            # Only one descriptor variant — merge bare into it
            dk = desc_keys[0]
            subject_counts[dk] += subject_counts[bk]
            all_mentions[dk].extend(all_mentions[bk])
            del subject_counts[bk]
            del all_mentions[bk]

    # ── Score candidates ──
    total_qs = len(qa_pairs)
    scores = {}
    for key, count in subject_counts.items():
        singular, desc = key
        mentions = all_mentions[key]
        score = count * 10                                 # base: subject frequency
        if any(e["is_plural"] for e in mentions):
            score -= 15                                    # plural = hard to target one
        if desc:
            score += 5                                     # visual descriptor = SAM3 friendly
        avg_words = sum(len(e["mention"].split()) for e in mentions) / len(mentions)
        if avg_words > 1.5:
            score += 3                                     # specificity bonus
        if singular in ("person",):
            score -= 10                                    # too generic
        scores[key] = score

    # Tie-break: prefer descriptored entity, then higher coverage
    best_key = max(scores, key=lambda k: (scores[k], bool(k[1]), subject_counts[k]))
    best_mentions = all_mentions[best_key]
    prompt = _pick_prompt(best_mentions)
    coverage = subject_counts[best_key] / total_qs

    candidates = []
    for k in sorted(scores, key=lambda x: scores[x], reverse=True)[:5]:
        candidates.append({
            "entity": f"{k[0]}({k[1] or 'generic'})",
            "score": scores[k],
            "subject_count": subject_counts[k],
            "example": all_mentions[k][0]["mention"],
        })

    return {
        "prompt_word": prompt,
        "subject_question_count": subject_counts[best_key],
        "total_questions": total_qs,
        "coverage": round(coverage, 2),
        "score": scores[best_key],
        "candidates": candidates,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Select removal-target prompts from NExT-QA questions")
    ap.add_argument("--input", default="val_grouped.csv", help="Grouped CSV (one row per video)")
    ap.add_argument("--output", default="data.json", help="Output JSON for batch pipeline")
    args = ap.parse_args()

    with open(args.input) as f:
        videos = list(csv.DictReader(f))

    results = []
    no_prompt = []

    for v in videos:
        vid = v["video"]
        analysis = analyze_video(v["questions_and_answers"])
        if analysis:
            results.append({
                "id": vid,
                "prompt_word": analysis["prompt_word"],
            })
        else:
            no_prompt.append(vid)
            results.append({"id": vid, "prompt_word": None})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──
    valid = sum(1 for r in results if r["prompt_word"])
    print(f"Written {args.output}: {valid}/{len(results)} videos with prompts")
    if no_prompt:
        print(f"No prompt extracted for {len(no_prompt)} videos: {no_prompt[:20]}")

    # ── Distribution ──
    prompts = [r["prompt_word"] for r in results if r["prompt_word"]]
    base_nouns = []
    for p in prompts:
        words = p.split()
        for w in words:
            if w in ENTITY_NOUNS or w in PLURAL_TO_SINGULAR:
                base_nouns.append(PLURAL_TO_SINGULAR.get(w, w))
                break
    print(f"\nEntity noun distribution (top 15):")
    for noun, cnt in Counter(base_nouns).most_common(15):
        print(f"  {noun:15s} {cnt:4d}")

    print(f"\nPrompt examples (first 20):")
    for r in results[:20]:
        print(f"  {r['id']:>15s} -> {r['prompt_word']}")


if __name__ == "__main__":
    main()
