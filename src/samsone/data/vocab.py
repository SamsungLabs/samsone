import string

# Define the character set for the dummy text
CHARS = string.ascii_letters + string.digits + " .,!?"

# Create character to index mapping.
# Index 0 is reserved for padding/unknown tokens.
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}

# The vocabulary size is the number of unique characters plus one for the padding/unknown token.
VOCAB_SIZE = len(CHARS) + 1

# For convenience, create an index to character mapping
IDX_TO_CHAR = {i: ch for ch, i in CHAR_TO_IDX.items()}
