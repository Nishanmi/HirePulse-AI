from backend.candidate.parser import parse_candidates

candidates = parse_candidates("data/raw/sample_candidates.json")

print(f"Loaded {len(candidates)} candidates")

if candidates:
    print(candidates[0])
