"""Compare a response and reference with traditional NLP metrics."""

from traditional_metrics import (
    bleu_score,
    gleu_score,
    meteor_style_score,
    rouge_l,
    rouge_n,
    token_f1_score,
)

RESPONSE = "Tokyo is the capital of Japan and has over 13 million residents."
REFERENCE = (
    "The capital of Japan is Tokyo, with a population exceeding 13 million people."
)


def main() -> None:
    print(f"BLEU:         {bleu_score(RESPONSE, REFERENCE):.3f}")
    print(f"GLEU:         {gleu_score(RESPONSE, REFERENCE):.3f}")
    print(f"METEOR-style: {meteor_style_score(RESPONSE, REFERENCE):.3f}")
    print(f"Token F1:     {token_f1_score(RESPONSE, REFERENCE):.3f}")
    print(f"ROUGE-1:      {rouge_n(RESPONSE, REFERENCE, 1)['f1']:.3f}")
    print(f"ROUGE-2:      {rouge_n(RESPONSE, REFERENCE, 2)['f1']:.3f}")
    print(f"ROUGE-L:      {rouge_l(RESPONSE, REFERENCE)['f1']:.3f}")


if __name__ == "__main__":
    main()
