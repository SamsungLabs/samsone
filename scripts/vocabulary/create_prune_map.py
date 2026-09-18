import json
import re

from transformers import AutoTokenizer


def clean_token(token: str) -> str:
    # Keeps only standard ASCII letters and digits for the case-check
    return re.sub(r"[^A-Za-z0-9]", "", token)


def has_capital(token: str) -> bool:
    cleaned = clean_token(token)
    return any(c.isupper() for c in cleaned)


def create_prune_map(model_id, output_file="token_map.json"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    # Sort by ID to maintain correct indexing
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    special_token_ids = set(tokenizer.all_special_ids)
    pruned_map = {}
    reverse_pruned_map = {}
    new_id = 0

    def should_keep(token_str):
        # 1. Always keep special/control tokens
        if old_id in special_token_ids:
            return True

        # 2. Capitalization Rule
        if has_capital(token_str):
            return False

        # 3. Rule: No more than 3 white characters in a row
        # (Handles both literal space/tab and the BPE 'Ġ')
        if re.search(r"[\sĠĊĉ]{4,}", token_str):
            return False

        # 4. Rule: Latin-only (Basic ASCII + the 'Ġ' space marker)
        # This removes Cyrillic, Asian scripts, and Emojis
        if re.search(r"[^\x00-\x7FĠĊĉ\s]", token_str):
            return False
        if re.search(r"[-#]{3,}", token_str):
            return False
        return True

    removed = []
    kept = []
    for token_str, old_id in sorted_vocab:
        if should_keep(token_str):
            # If it passed all filters, add to map
            pruned_map[new_id] = old_id
            reverse_pruned_map[old_id] = new_id
            new_id += 1
            kept += [token_str]
        else:
            removed += [token_str]

    print("Extraction Complete!")
    print(f"New Vocab Size: {new_id} (Original: {len(vocab)})")
    return pruned_map, reverse_pruned_map, removed, kept


if __name__ == "__main__":
    pruned_map, reverse_pruned_map, removed, kept = create_prune_map(
        "HuggingFaceTB/SmolLM2-135M"
    )

    with open("outputs/prune_map.json", "w") as f:
        json.dump(pruned_map, f, indent=2)

    with open("outputs/reverse_prune_map.json", "w") as f:
        json.dump(reverse_pruned_map, f, indent=2)

    with open("outputs/removed.json", "w") as f:
        json.dump(removed, f, indent=2)

    with open("outputs/kept.json", "w") as f:
        json.dump(kept, f, indent=2)
