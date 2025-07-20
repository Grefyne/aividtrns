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


def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text and split into manageable chunks."""
    text_splits = []
    
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        
        if SPACY_AVAILABLE:
            nlp = get_spacy_lang(lang)
            if nlp:
                nlp.add_pipe("sentencizer")
                doc = nlp(text)
                
                for sentence in doc.sents:
                    if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                        # if the last sentence + the current sentence is less than the text_split_length
                        # then add the current sentence to the last sentence
                        text_splits[-1] += " " + str(sentence)
                        text_splits[-1] = text_splits[-1].lstrip()
                    elif len(str(sentence)) > text_split_length:
                        # if the current sentence is greater than the text_split_length
                        for line in textwrap.wrap(
                            str(sentence),
                            width=text_split_length,
                            drop_whitespace=True,
                            break_on_hyphens=False,
                            tabsize=1,
                        ):
                            text_splits.append(str(line))
                    else:
                        text_splits.append(str(sentence))
                
                if len(text_splits) > 1:
                    if text_splits[0] == "":
                        del text_splits[0]
            else:
                # Fallback to simple splitting
                text_splits = [text[i:i+text_split_length] for i in range(0, len(text), text_split_length)]
        else:
            # Fallback to simple splitting when spaCy is not available
            text_splits = [text[i:i+text_split_length] for i in range(0, len(text), text_split_length)]
    else:
        text_splits = [text.lstrip()]
    
    return text_splits


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
        self.text_split_length = 250
    
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


def tokenize_for_xtts(text, language="en", max_length=250):
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