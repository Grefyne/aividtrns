#!/usr/bin/env python3
"""
Multilingual tokenizer for XTTS-v2 text preprocessing.
Based on Sindhi-XTTSv2-Tokenizer by fahadmaqsood.
"""

import logging
import os
import re
import textwrap
from functools import cached_property
import torch
from num2words import num2words

# Conditional imports for optional dependencies
try:
    from spacy.lang.ar import Arabic
    from spacy.lang.en import English
    from spacy.lang.es import Spanish
    from spacy.lang.hi import Hindi
    from spacy.lang.ja import Japanese
    from spacy.lang.zh import Chinese
    from spacy.lang.ur import Urdu
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using fallback sentence splitting.")

try:
    from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words
    ZH_NUM2WORDS_AVAILABLE = True
except ImportError:
    ZH_NUM2WORDS_AVAILABLE = False
    print("Warning: Chinese num2words not available.")

logger = logging.getLogger(__name__)


def get_spacy_lang(lang):
    """Return Spacy language used for sentence splitting."""
    if not SPACY_AVAILABLE:
        return None
    
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "ur":
        return Urdu()
    elif lang == "es":
        return Spanish()
    elif lang == "hi":
        return Hindi()
    else:
        # For most languages, English does the job
        return English()


def split_sentence(text, lang, text_split_length=150):
    """
    Preprocess the input text and split into manageable chunks.
    Uses sentence-based splitting with reduced length to prevent TTS truncation.
    """
    text_splits = []
    
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        
        if SPACY_AVAILABLE:
            nlp = get_spacy_lang(lang)
            if nlp:
                nlp.add_pipe("sentencizer")
                doc = nlp(text)
                
                for sentence in doc.sents:
                    sentence_text = str(sentence).strip()
                    if not sentence_text:
                        continue
                        
                    # If current chunk is empty, start with this sentence
                    if not text_splits[-1]:
                        if len(sentence_text) <= text_split_length:
                            text_splits[-1] = sentence_text
                        else:
                            # Split long sentence by clauses or phrases
                            split_long_sentence(sentence_text, text_split_length, text_splits)
                    elif len(text_splits[-1]) + len(" " + sentence_text) <= text_split_length:
                        # Add sentence to current chunk if it fits
                        text_splits[-1] += " " + sentence_text
                    else:
                        # Start new chunk with this sentence
                        if len(sentence_text) <= text_split_length:
                            text_splits.append(sentence_text)
                        else:
                            # Split long sentence and add to new chunks
                            split_long_sentence(sentence_text, text_split_length, text_splits)
                
                # Clean up empty chunks
                text_splits = [chunk.strip() for chunk in text_splits if chunk.strip()]
            else:
                # Enhanced fallback when spaCy is not available
                text_splits = fallback_sentence_split(text, text_split_length)
        else:
            # Enhanced fallback when spaCy is not available
            text_splits = fallback_sentence_split(text, text_split_length)
    else:
        text_splits = [text.strip()]
    
    return text_splits


def split_long_sentence(sentence, max_length, text_splits):
    """Split a long sentence by clauses, phrases, or words."""
    import re
    
    # First try splitting by clauses (commas, semicolons)
    clause_pattern = r'([,;:])'
    parts = re.split(clause_pattern, sentence)
    
    # Rejoin with punctuation
    clauses = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in [',', ';', ':']:
            clauses.append(parts[i] + parts[i + 1])
            i += 2
        else:
            if parts[i].strip():
                clauses.append(parts[i])
            i += 1
    
    # If we got good clauses, use them
    if len(clauses) > 1:
        current_chunk = ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            if not current_chunk:
                current_chunk = clause
            elif len(current_chunk + " " + clause) <= max_length:
                current_chunk += " " + clause
            else:
                text_splits.append(current_chunk)
                current_chunk = clause
        
        if current_chunk:
            text_splits.append(current_chunk)
    else:
        # Fall back to word wrapping
        import textwrap
        for line in textwrap.wrap(
            sentence,
            width=max_length,
            drop_whitespace=True,
            break_on_hyphens=False,
            break_long_words=False,
        ):
            text_splits.append(line.strip())


def fallback_sentence_split(text, max_length):
    """Enhanced fallback sentence splitting when spaCy is not available."""
    import re
    
    # Split by sentence endings
    sentences = re.split(r'([.!?]+)', text)
    
    # Rejoin sentences with their punctuation
    clean_sentences = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and re.match(r'[.!?]+', sentences[i + 1]):
            clean_sentences.append(sentences[i] + sentences[i + 1])
            i += 2
        else:
            if sentences[i].strip():
                clean_sentences.append(sentences[i])
            i += 1
    
    # Build chunks from sentences
    text_splits = []
    current_chunk = ""
    
    for sentence in clean_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(sentence) > max_length:
            # Add current chunk if it exists
            if current_chunk:
                text_splits.append(current_chunk)
                current_chunk = ""
            
            # Split long sentence
            split_long_sentence(sentence, max_length, text_splits)
        elif not current_chunk:
            current_chunk = sentence
        elif len(current_chunk + " " + sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            text_splits.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        text_splits.append(current_chunk)
    
    return text_splits if text_splits else [text]


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "ur": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Urdu as in English.
        ]
    ],
    "zh": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),
            ("ing", "inženýr"),
            ("p", "pan"),
        ]
    ],
    "ru": [
        (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),
            ("г-н", "господин"),
            ("д-р", "доктор"),
        ]
    ],
    "nl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),
            ("mevr", "mevrouw"),
            ("dr", "dokter"),
            ("prof", "professor"),
            ("ing", "ingenieur"),
        ]
    ],
    "tr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),
            ("prof", "profesör"),
            ("b", "bay"),
            ("byn", "bayan"),
        ]
    ],
    "hu": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),
            ("prof", "professzor"),
            ("u", "úr"),
        ]
    ],
    "ko": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't use abbreviations in the same way
        ]
    ],
    "hi": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Hindi abbreviations are less common
        ]
    ],
}


# Symbol replacement mappings
_symbols = {
    "en": [
        (re.compile(r"%", re.IGNORECASE), " percent"),
        (re.compile(r"\$", re.IGNORECASE), " dollars"),
        (re.compile(r"£", re.IGNORECASE), " pounds"),
        (re.compile(r"€", re.IGNORECASE), " euros"),
        (re.compile(r"@", re.IGNORECASE), " at"),
        (re.compile(r"#", re.IGNORECASE), " hash"),
        (re.compile(r"&", re.IGNORECASE), " and"),
        (re.compile(r"\+", re.IGNORECASE), " plus"),
        (re.compile(r"°", re.IGNORECASE), " degrees"),
    ],
    "es": [
        (re.compile(r"%", re.IGNORECASE), " por ciento"),
        (re.compile(r"\$", re.IGNORECASE), " dólares"),
        (re.compile(r"£", re.IGNORECASE), " libras"),
        (re.compile(r"€", re.IGNORECASE), " euros"),
        (re.compile(r"@", re.IGNORECASE), " arroba"),
        (re.compile(r"#", re.IGNORECASE), " numeral"),
        (re.compile(r"&", re.IGNORECASE), " y"),
        (re.compile(r"\+", re.IGNORECASE), " más"),
        (re.compile(r"°", re.IGNORECASE), " grados"),
    ],
    "fr": [
        (re.compile(r"%", re.IGNORECASE), " pourcent"),
        (re.compile(r"\$", re.IGNORECASE), " dollars"),
        (re.compile(r"£", re.IGNORECASE), " livres"),
        (re.compile(r"€", re.IGNORECASE), " euros"),
        (re.compile(r"@", re.IGNORECASE), " arobase"),
        (re.compile(r"#", re.IGNORECASE), " dièse"),
        (re.compile(r"&", re.IGNORECASE), " et"),
        (re.compile(r"\+", re.IGNORECASE), " plus"),
        (re.compile(r"°", re.IGNORECASE), " degrés"),
    ],
    "de": [
        (re.compile(r"%", re.IGNORECASE), " prozent"),
        (re.compile(r"\$", re.IGNORECASE), " dollar"),
        (re.compile(r"£", re.IGNORECASE), " pfund"),
        (re.compile(r"€", re.IGNORECASE), " euro"),
        (re.compile(r"@", re.IGNORECASE), " klammeraffe"),
        (re.compile(r"#", re.IGNORECASE), " raute"),
        (re.compile(r"&", re.IGNORECASE), " und"),
        (re.compile(r"\+", re.IGNORECASE), " plus"),
        (re.compile(r"°", re.IGNORECASE), " grad"),
    ],
    "pt": [
        (re.compile(r"%", re.IGNORECASE), " por cento"),
        (re.compile(r"\$", re.IGNORECASE), " dólares"),
        (re.compile(r"£", re.IGNORECASE), " libras"),
        (re.compile(r"€", re.IGNORECASE), " euros"),
        (re.compile(r"@", re.IGNORECASE), " arroba"),
        (re.compile(r"#", re.IGNORECASE), " cardinal"),
        (re.compile(r"&", re.IGNORECASE), " e"),
        (re.compile(r"\+", re.IGNORECASE), " mais"),
        (re.compile(r"°", re.IGNORECASE), " graus"),
    ],
    # Add more languages as needed...
}


def expand_abbreviations_multilingual(text, lang="en"):
    """Expand abbreviations for the given language."""
    if lang not in _abbreviations:
        lang = "en"  # Default to English
    
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_symbols_multilingual(text, lang="en"):
    """Expand symbols for the given language."""
    if lang not in _symbols:
        lang = "en"  # Default to English
    
    for regex, replacement in _symbols[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers_multilingual(text, lang="en"):
    """Expand numbers to words for the given language."""
    try:
        # Handle decimal numbers first (before regular numbers to avoid conflicts)
        decimal_pattern = r'\b\d+\.\d+\b'
        
        def replace_decimal(match):
            try:
                number_str = match.group(0)
                if lang == "zh" and ZH_NUM2WORDS_AVAILABLE:
                    # Use Chinese-specific number conversion
                    normalizer = zh_num2words()
                    return normalizer.normalize(number_str)
                else:
                    # For other languages, convert using num2words
                    integer_part, decimal_part = number_str.split('.')
                    integer_words = num2words(int(integer_part), lang=lang)
                    
                    # Language-specific decimal separators
                    if lang == "es":
                        decimal_word = "coma"
                    elif lang == "fr":
                        decimal_word = "virgule"
                    elif lang == "de":
                        decimal_word = "komma"
                    elif lang == "pt":
                        decimal_word = "vírgula"
                    elif lang == "it":
                        decimal_word = "virgola"
                    else:
                        decimal_word = "point"
                    
                    # Convert each decimal digit separately
                    decimal_words = " ".join([num2words(int(d), lang=lang) for d in decimal_part])
                    return f"{integer_words} {decimal_word} {decimal_words}"
            except:
                return match.group(0)  # Return original if conversion fails
        
        text = re.sub(decimal_pattern, replace_decimal, text)
        
        # Handle ordinals (1st, 2nd, 3rd, etc.)
        ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
        
        def replace_ordinal(match):
            number = int(match.group(1))
            try:
                return num2words(number, ordinal=True, lang=lang)
            except:
                return match.group(0)  # Return original if conversion fails
        
        text = re.sub(ordinal_pattern, replace_ordinal, text, flags=re.IGNORECASE)
        
        # Handle regular numbers
        number_pattern = r'\b\d+\b'
        
        def replace_number(match):
            number = int(match.group(0))
            try:
                return num2words(number, lang=lang)
            except:
                return match.group(0)  # Return original if conversion fails
        
        text = re.sub(number_pattern, replace_number, text)
        
    except Exception as e:
        logger.warning(f"Error expanding numbers for language {lang}: {e}")
        # Return original text if number expansion fails
        pass
    
    return text


class MultilingualTokenizer:
    """Multilingual tokenizer for XTTS-v2 text preprocessing."""
    
    def __init__(self, language="en"):
        self.language = language
        self.text_split_length = 150
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline."""
        # Clean and normalize text
        text = text.strip()
        if not text:
            return []
        
        # Expand abbreviations
        text = expand_abbreviations_multilingual(text, self.language)
        
        # Expand numbers to words (before symbols to handle $20.50 correctly)
        text = expand_numbers_multilingual(text, self.language)
        
        # Expand symbols
        text = expand_symbols_multilingual(text, self.language)
        
        # Split into manageable chunks
        text_splits = split_sentence(text, self.language, self.text_split_length)
        
        # Clean up whitespace
        text_splits = [" ".join(split.split()) for split in text_splits if split.strip()]
        
        return text_splits
    
    def tokenize(self, text):
        """Tokenize text for XTTS-v2."""
        return self.preprocess_text(text)
    
    def set_language(self, language):
        """Set the language for tokenization."""
        self.language = language
    
    def set_text_split_length(self, length):
        """Set the maximum length for text splits."""
        self.text_split_length = length


def tokenize_for_xtts(text, language="en", max_length=150):
    """Convenience function to tokenize text for XTTS-v2."""
    tokenizer = MultilingualTokenizer(language)
    tokenizer.set_text_split_length(max_length)
    return tokenizer.tokenize(text)


def main():
    """Test the tokenizer with various languages."""
    test_cases = [
        ("Hello Mr. Smith. I have 14% battery and it costs $20.50.", "en"),
        ("Hola Sr. García. Tengo 14% de batería y cuesta $20.50.", "es"),
        ("Bonjour Mr. Dupont. J'ai 14% de batterie et ça coûte 20,50€.", "fr"),
        ("Hallo Dr. Müller. Ich habe 14% Akku und es kostet 20,50€.", "de"),
    ]
    
    for text, lang in test_cases:
        print(f"\nLanguage: {lang}")
        print(f"Original: {text}")
        
        tokenizer = MultilingualTokenizer(lang)
        processed = tokenizer.tokenize(text)
        
        print(f"Processed: {processed}")


if __name__ == "__main__":
    main() 