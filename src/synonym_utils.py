import spacy
nlp = spacy.load("en_core_web_sm")

def expand_with_synonyms(text: str, top_k: int = 2) -> set[str]:
    """
    Liefert Menge von Wörtern im Text + ähnlichen Wörtern via spaCy.
    """
    doc = nlp(text)
    keywords = set()

    for token in doc:
        if not token.is_alpha or token.is_stop or len(token.text) < 3:
            continue
        keywords.add(token.text.lower())
        # Ähnliche Wörter aus dem Vokabular suchen
        similar = sorted(
            nlp.vocab,
            key=lambda w: token.similarity(w),
            reverse=True
        )
        # Nur sinnvolle Kandidaten mit ähnlichem Kontext
        for w in similar[:top_k + 5]:
            if w.has_vector and w.is_lower and w.is_alpha and w.text != token.text:
                keywords.add(w.text)
                if len(keywords) >= top_k:
                    break
    return keywords