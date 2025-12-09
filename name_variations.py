import re
from typing import List

def name_variations(full_name: str) -> List[str]:
    # Split on whitespace first
    raw_parts = [p for p in full_name.strip().split() if p]
    if not raw_parts:
        return []

    # Helper: split CamelCase like "KokShiek" -> ["Kok", "Shiek"]
    def split_camel(part: str) -> list[str]:
        # If there is any extra capital letter, try to split
        if any(c.isupper() for c in part[1:]):
            pieces = re.findall(r"[A-Z][^A-Z]*", part)
            return pieces or [part]
        return [part]

    # We keep original tokens for matching, but ensure first letter is uppercase
    parts = []
    for p in raw_parts:
        if len(p) > 1:
            parts.append(p[0].upper() + p[1:])
        else:
            parts.append(p.upper())

    n = len(parts)
    if n == 1:
        return [parts[0].lower()]

    first = parts[0]
    last = parts[-1]
    middles = parts[1:-1]

    variants = set()

    def initials(names):
        return "".join(name[0].upper() for name in names if name)

    # 1) Full name as given
    variants.add(" ".join(parts))

    # 2) "Last First Middle..."
    variants.add(f"{last} {' '.join([first] + middles)}".strip())

    # Build given-block and expand CamelCase for initials
    given_block = [first] + middles

    expanded_given_block = []
    for token in given_block:
        expanded_given_block.extend(split_camel(token))

    given_initials = initials(expanded_given_block)  # e.g. "KS" from "KokShiek"
    last_initial = last[0].upper()
    first_initial = first[0].upper()

    # 3) "K Wong" style
    variants.add(f"{first_initial} {last}")

    # 4) "KS Wong" style (this is what you want for KokShiek Wong)
    variants.add(f"{given_initials} {last}")

    # 5) "Wong KS"
    variants.add(f"{last} {given_initials}")

    # 6) Pattern like "WK Kok Shiek" if there are middles, kept from your original idea
    if middles:
        middle_full = " ".join(middles)
        variants.add(f"{last_initial}{first_initial} {middle_full}")

    # Return everything lowercased for matching
    return [name.lower() for name in variants]
