import logging
import os
import re
import textwrap
from functools import cached_property

import torch
from num2words import num2words
from spacy.lang.ar import Arabic
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.hi import Hindi
from spacy.lang.ja import Japanese
from spacy.lang.zh import Chinese
from spacy.lang.ur import Urdu
from tokenizers import Tokenizer

from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words
from TTS.tts.utils.text.cleaners import collapse_whitespace, lowercase

logger = logging.getLogger(__name__)

def get_spacy_lang(lang):
    """Return Spacy language used for sentence splitting."""
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
        return English()
def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text with a robust fallback mechanism."""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        try:
            nlp = get_spacy_lang(lang)
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            current_chunk = ""
            for sentence in doc.sents:
                sentence_text = str(sentence).strip()
                if len(current_chunk) + len(sentence_text) <= text_split_length:
                    current_chunk = (current_chunk + " " + sentence_text).lstrip()
                else:
                    if current_chunk:
                        text_splits.append(current_chunk)
                    if len(sentence_text) > text_split_length:
                        # Split long sentences using fallback
                        for line in textwrap.wrap(
                            sentence_text,
                            width=text_split_length,
                            drop_whitespace=True,
                            break_on_hyphens=False,
                            tabsize=1,
                        ):
                            text_splits.append(line.strip())
                    else:
                        text_splits.append(sentence_text)
                    current_chunk = ""
            if current_chunk:
                text_splits.append(current_chunk)
        except Exception as e:
            logger.warning(f"spaCy sentencizer failed for lang {lang}: {e}. Using fallback splitting.")
            sentences = text.split('. ')
            current_chunk = ""
            for sentence in sentences:
                sentence_text = sentence.strip()
                if sentence_text:
                    if len(current_chunk) + len(sentence_text) <= text_split_length:
                        current_chunk = (current_chunk + " " + sentence_text).lstrip()
                    else:
                        if current_chunk:
                            text_splits.append(current_chunk)
                        if len(sentence_text) > text_split_length:
                            for line in textwrap.wrap(
                                sentence_text,
                                width=text_split_length,
                                drop_whitespace=True,
                                break_on_hyphens=False,
                                tabsize=1,
                            ):
                                text_splits.append(line.strip())
                        else:
                            text_splits.append(sentence_text)
                        current_chunk = ""
            if current_chunk:
                text_splits.append(current_chunk)
    else:
        text_splits = [text.strip()]
    
    if not text_splits or text_splits[0] == "":
        text_splits = [text.strip()]
    return [s for s in text_splits if s]

# Abbreviations (unchanged)
_abbreviations = {
    "en": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
            ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
            ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"),
            ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
            ("col", "colonel"), ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"), ("sr", "señor"), ("dr", "doctor"), ("dra", "doctora"),
            ("st", "santo"), ("co", "compañía"), ("jr", "junior"), ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"), ("mr", "monsieur"), ("dr", "docteur"), ("st", "saint"),
            ("co", "compagnie"), ("jr", "junior"), ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"), ("dr", "doktor"), ("st", "sankt"), ("co", "firma"), ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"), ("sr", "senhor"), ("dr", "doutor"), ("dra", "doutora"),
            ("st", "santo"), ("co", "companhia"), ("jr", "júnior"), ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("sig", "signore"), ("dr", "dottore"), ("st", "santo"), ("co", "compagnia"),
            ("jr", "junior"), ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"), ("m", "pan"), ("dr", "doktor"), ("sw", "święty"), ("jr", "junior"),
        ]
    ],
    "ar": [],
    "ur": [],
    "sd": [],
    "zh": [],
    "cs": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"), ("ing", "inženýr"), ("p", "pan"),
        ]
    ],
    "ru": [
        (re.compile(r"\b%s\b" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"), ("г-н", "господин"), ("д-р", "доктор"),
        ]
    ],
    "nl": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"), ("mevr", "mevrouw"), ("dr", "dokter"), ("jhr", "jonkheer"),
        ]
    ],
    "tr": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"), ("byk", "büyük"), ("dr", "doktor"),
        ]
    ],
    "hu": [
        (re.compile(r"\b%s\." % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"), ("b", "bácsi"), ("nőv", "nővér"),
        ]
    ],
    "ko": [],
    "hi": [],
}

# Symbols (unchanged)
_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "), ("@", " at "), ("%", " percent "), ("#", " hash "),
            ("$", " dollar "), ("£", " pound "), ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " y "), ("@", " arroba "), ("%", " por ciento "), ("#", " numeral "),
            ("$", " dolar "), ("£", " libra "), ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " et "), ("@", " arobase "), ("%", " pour cent "), ("#", " dièse "),
            ("$", " dollar "), ("£", " livre "), ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " und "), ("@", " at "), ("%", " prozent "), ("#", " raute "),
            ("$", " dollar "), ("£", " pfund "), ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "), ("@", " arroba "), ("%", " por cento "), ("#", " cardinal "),
            ("$", " dólar "), ("£", " libra "), ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "), ("@", " chiocciola "), ("%", " per cento "), ("#", " cancelletto "),
            ("$", " dollaro "), ("£", " sterlina "), ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " i "), ("@", " małpa "), ("%", " procent "), ("#", " krzyżyk "),
            ("$", " dolar "), ("£", " funt "), ("°", " stopnie "),
        ]
    ],
    "ar": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " و "), ("@", " على "), ("%", " في المئة "), ("#", " رقم "),
            ("$", " دولار "), ("£", " جنيه "), ("°", " درجة "),
        ]
    ],
    "ur": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " اور "), ("@", " پر "), ("%", " فیصد "), ("#", " ہیش "),
            ("$", " ڈالر "), ("£", " پاؤنڈ "), ("°", " ڈگری "), ("*", " ستارہ "),
            ("@", " ای میل "), ("!", " اچھا "),
        ]
    ],
    "sd": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ۽ "), ("@", " تَي "), ("%", " سيڪڙَو "), ("#", " هَيش "),
            ("$", " ڊالر "), ("£", " پائونڊ "), ("°", " ڊگري "), ("*", " تارَو "),
        ]
    ],
    "zh": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "), ("@", " 在 "), ("%", " 百分之 "), ("#", " 号 "),
            ("$", " 美元 "), ("£", " 英镑 "), ("°", " 度 "),
        ]
    ],
    "cs": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " a "), ("@", " na "), ("%", " procento "), ("#", " křížek "),
            ("$", " dolar "), ("£", " libra "), ("°", " stupně "),
        ]
    ],
    "ru": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " и "), ("@", " собака "), ("%", " процентов "), ("#", " номер "),
            ("$", " доллар "), ("£", " фунт "), ("°", " градус "),
        ]
    ],
    "nl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " en "), ("@", " bij "), ("%", " procent "), ("#", " hekje "),
            ("$", " dollar "), ("£", " pond "), ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "), ("@", " at "), ("%", " yüzde "), ("#", " diyez "),
            ("$", " dolar "), ("£", " sterlin "), ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " és "), ("@", " kukac "), ("%", " százalék "), ("#", " kettőskereszt "),
            ("$", " dollár "), ("£", " font "), ("°", " fok "),
        ]
    ],
    "ko": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "), ("@", " 에 "), ("%", " 퍼센트 "), ("#", " 번호 "),
            ("$", " 달러 "), ("£", " 파운드 "), ("°", " 도 "),
        ]
    ],
    "hi": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " और "), ("@", " ऐट दी रेट "), ("%", " प्रतिशत "), ("#", " हैश "),
            ("$", " डॉलर "), ("£", " पाउंड "), ("°", " डिग्री "),
        ]
    ],
}

# Regex patterns
_ordinal_re = {
    "en": re.compile(r"([0-9]+(?:\s*(?:million|billion|thousand))?\s*)(st|nd|rd|th)\b", re.IGNORECASE),
    "es": re.compile(r"([0-9]+(?:\s*(?:millón|mil))?\s*)(º|ª|er|o|a|os|as)\b", re.IGNORECASE),
    "fr": re.compile(r"([0-9]+(?:\s*(?:million|mille))?\s*)(º|ª|er|re|e|ème)\b", re.IGNORECASE),
    "de": re.compile(r"([0-9]+(?:\s*(?:Millionen|tausend))?\s*)(st|nd|rd|th|º|ª|\.(?=\s|$))\b", re.IGNORECASE),
    "pt": re.compile(r"([0-9]+(?:\s*(?:milhão|mil))?\s*)(º|ª|o|a|os|as)\b", re.IGNORECASE),
    "it": re.compile(r"([0-9]+(?:\s*(?:milione|mille))?\s*)(º|°|ª|o|a|i|e)\b", re.IGNORECASE),
    "pl": re.compile(r"([0-9]+(?:\s*(?:milion|tysiąc))?\s*)(º|ª|st|nd|rd|th)\b", re.IGNORECASE),
    "ar": re.compile(r"([0-9]+(?:\s*(?:مليون|ألف))?\s*)(ون|ين|ث|ر|ى)\b", re.IGNORECASE),
    "cs": re.compile(r"([0-9]+(?:\s*(?:milión|tisíc))?\s*)\.(?=\s|$)", re.IGNORECASE),
    "ru": re.compile(r"([0-9]+(?:\s*(?:миллион|тысяча))?\s*)(-й|-я|-е|-ое|-ье|-го)\b", re.IGNORECASE),
    "nl": re.compile(r"([0-9]+(?:\s*(?:miljoen|duizend))?\s*)(de|ste|e)\b", re.IGNORECASE),
    "tr": re.compile(r"([0-9]+(?:\s*(?:milyon|bin))?\s*)(\.|inci|nci|uncu|üncü|\.)\b", re.IGNORECASE),
    "hu": re.compile(r"([0-9]+(?:\s*(?:millió|ezres))?\s*)(\.|adik|edik|odik|edik|ödik|ödike|ik)\b", re.IGNORECASE),
    "ko": re.compile(r"([0-9]+(?:\s*(?:백만|천))?\s*)(번째|번|차|째)\b", re.IGNORECASE),
    "hi": re.compile(r"([0-9]+(?:\s*(?:लाख|हज़ार))?\s*)(st|nd|rd|th)\b", re.IGNORECASE),
    "ur": re.compile(r"([0-9]+(?:\s*(?:ملین|ہزار))?\s*)(ون|ين|ث|ر|ى)\b", re.IGNORECASE),
    "sd": re.compile(r"([0-9]+(?:\s*(?:ملین|ہزار))?\s*)(ون|ين|ث|ر|ى)\b", re.IGNORECASE),
}

# Fixed number regex with proper closing parenthesis
_number_re = re.compile(r"([0-9]+(?:\s*(?:million|billion|thousand|milion|tysiąc|milión|тысяча|duizend|milyon|ezres|백만|천|लाख|हज़ार|ملین|ہزار))?\b)", re.IGNORECASE)

_currency_re = {
    "USD": re.compile(r"(?:\$([0-9]+(?:\s*(?:million|billion|thousand))?)|([0-9]+(?:\s*(?:million|billion|thousand))?\s*\$))\b", re.IGNORECASE),
    "GBP": re.compile(r"(?:£([0-9]+(?:\s*(?:million|billion|thousand))?)|([0-9]+(?:\s*(?:million|billion|thousand))?\s*£))\b", re.IGNORECASE),
    "EUR": re.compile(r"(?:(€[0-9]+(?:\s*(?:million|billion|thousand))?)|([0-9]+(?:\s*(?:million|billion|thousand))?\s*€))\b", re.IGNORECASE),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)\b")

def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text

def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text

def _expand_decimal_point(m, lang="en"):
    try:
        amount = m.group(1).replace(",", ".")
        return num2words(float(amount), lang=lang if lang != "cs" else "cz")
    except ValueError:
        return m.group(0)

def _expand_currency(m, lang="en", currency="USD"):
    try:
        amount = float(re.sub(r"[^\d.]", "", m.group(0).replace(",", ".")))
        full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")
        and_equivalents = {
            "en": ", ", "es": " con ", "fr": " et ", "de": " und ", "pt": " e ",
            "it": " e ", "pl": ", ", "cs": ", ", "ru": ", ", "nl": ", ",
            "ar": ", ", "ur": ", ", "tr": ", ", "hu": ", ", "ko": ", ",
            "hi": ", ", "sd": ", ",
        }
        if amount.is_integer():
            last_and = full_amount.rfind(and_equivalents[lang])
            if last_and != -1:
                full_amount = full_amount[:last_and]
        return full_amount
    except ValueError:
        return m.group(0)

def _expand_ordinal(m, lang="en"):
    try:
        number = m.group(1).strip()
        if "million" in number.lower():
            number = number.replace("million", "").strip()
            return f"{num2words(int(number), ordinal=True, lang=lang if lang != 'cs' else 'cz')} million"
        elif "billion" in number.lower():
            number = number.replace("billion", "").strip()
            return f"{num2words(int(number), ordinal=True, lang=lang if lang != 'cs' else 'cz')} billion"
        elif "thousand" in number.lower():
            number = number.replace("thousand", "").strip()
            return f"{num2words(int(number), ordinal=True, lang=lang if lang != 'cs' else 'cz')} thousand"
        return num2words(int(number), ordinal=True, lang=lang if lang != "cs" else "cz")
    except (ValueError, IndexError):
        return m.group(0)

def _expand_number(m, lang="en"):
    try:
        number = m.group(1).strip()
        if "million" in number.lower():
            number = number.replace("million", "").strip()
            return f"{num2words(int(number), lang=lang if lang != 'cs' else 'cz')} million"
        elif "billion" in number.lower():
            number = number.replace("billion", "").strip()
            return f"{num2words(int(number), lang=lang if lang != 'cs' else 'cz')} billion"
        elif "thousand" in number.lower():
            number = number.replace("thousand", "").strip()
            return f"{num2words(int(number), lang=lang if lang != 'cs' else 'cz')} thousand"
        return num2words(int(number), lang=lang if lang != "cs" else "cz")
    except (ValueError, IndexError):
        return m.group(0)

def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    elif lang in ["ur", "sd"]:
        text = text
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
            text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
            text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
        except Exception as e:
            logger.warning(f"Number expansion failed for lang {lang}: {e}")
    return text

def expand_abbreviations_multilingual(text, lang="en"):
    try:
        for regex, replacement in _abbreviations.get(lang, []):
            text = re.sub(regex, replacement, text)
    except Exception as e:
        logger.warning(f"Abbreviation expansion failed for lang {lang}: {e}")
    return text

def expand_symbols_multilingual(text, lang="en"):
    try:
        for regex, replacement in _symbols_multilingual.get(lang, []):
            text = re.sub(regex, replacement, text)
            text = text.replace("  ", " ")
    except Exception as e:
        logger.warning(f"Symbol expansion failed for lang {lang}: {e}")
    return text.strip()

def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i").replace("Ö", "ö").replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang)
    text = collapse_whitespace(text)
    return text

def chinese_transliterate(text):
    try:
        import pypinyin
    except ImportError as e:
        raise ImportError("Chinese requires: pypinyin") from e
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )

def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text

def korean_transliterate(text):
    try:
        from hangul_romanize import Transliter
        from hangul_romanize.rule import academic
    except ImportError as e:
        raise ImportError("Korean requires: hangul_romanize") from e
    r = Transliter(academic)
    return r.translit(text)

DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json")

class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limits = {
            "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203, "pl": 224,
            "zh": 82, "ar": 166, "cs": 186, "ru": 182, "nl": 251, "tr": 226, "ja": 71,
            "hu": 224, "ko": 95, "hi": 150, "ur": 150, "sd": 400,
        }

    @cached_property
    def katsu(self):
        import cutlet
        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            logger.warning(
                "The text length exceeds the character limit of %d for language '%s', this might cause truncated audio.",
                limit, lang,
            )

    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko", "ur", "sd"}:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = korean_transliterate(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1


def test_expand_numbers_multilingual():
    test_cases = [
        # English
        ("In 12.5 seconds.", "In twelve point five seconds.", "en"),
        ("There were 50 soldiers.", "There were fifty soldiers.", "en"),
        ("This is a 1st test", "This is a first test", "en"),
        ("That will be $20 sir.", "That will be twenty dollars sir.", "en"),
        ("That will be 20€ sir.", "That will be twenty euro sir.", "en"),
        ("That will be 20.15€ sir.", "That will be twenty euro, fifteen cents sir.", "en"),
        ("That's 100,000.5.", "That's one hundred thousand point five.", "en"),
        # French
        ("En 12,5 secondes.", "En douze virgule cinq secondes.", "fr"),
        ("Il y avait 50 soldats.", "Il y avait cinquante soldats.", "fr"),
        ("Ceci est un 1er test", "Ceci est un premier test", "fr"),
        ("Cela vous fera $20 monsieur.", "Cela vous fera vingt dollars monsieur.", "fr"),
        ("Cela vous fera 20€ monsieur.", "Cela vous fera vingt euros monsieur.", "fr"),
        ("Cela vous fera 20,15€ monsieur.", "Cela vous fera vingt euros et quinze centimes monsieur.", "fr"),
        ("Ce sera 100.000,5.", "Ce sera cent mille virgule cinq.", "fr"),
        # German
        ("In 12,5 Sekunden.", "In zwölf Komma fünf Sekunden.", "de"),
        ("Es gab 50 Soldaten.", "Es gab fünfzig Soldaten.", "de"),
        ("Dies ist ein 1. Test", "Dies ist ein erste Test", "de"),  # Issue with gender
        ("Das macht $20 Herr.", "Das macht zwanzig Dollar Herr.", "de"),
        ("Das macht 20€ Herr.", "Das macht zwanzig Euro Herr.", "de"),
        ("Das macht 20,15€ Herr.", "Das macht zwanzig Euro und fünfzehn Cent Herr.", "de"),
        # Spanish
        ("En 12,5 segundos.", "En doce punto cinco segundos.", "es"),
        ("Había 50 soldados.", "Había cincuenta soldados.", "es"),
        ("Este es un 1er test", "Este es un primero test", "es"),
        ("Eso le costará $20 señor.", "Eso le costará veinte dólares señor.", "es"),
        ("Eso le costará 20€ señor.", "Eso le costará veinte euros señor.", "es"),
        ("Eso le costará 20,15€ señor.", "Eso le costará veinte euros con quince céntimos señor.", "es"),
        # Italian
        ("In 12,5 secondi.", "In dodici virgola cinque secondi.", "it"),
        ("C'erano 50 soldati.", "C'erano cinquanta soldati.", "it"),
        ("Questo è un 1° test", "Questo è un primo test", "it"),
        ("Ti costerà $20 signore.", "Ti costerà venti dollari signore.", "it"),
        ("Ti costerà 20€ signore.", "Ti costerà venti euro signore.", "it"),
        ("Ti costerà 20,15€ signore.", "Ti costerà venti euro e quindici centesimi signore.", "it"),
        # Portuguese
        ("Em 12,5 segundos.", "Em doze vírgula cinco segundos.", "pt"),
        ("Havia 50 soldados.", "Havia cinquenta soldados.", "pt"),
        ("Este é um 1º teste", "Este é um primeiro teste", "pt"),
        ("Isso custará $20 senhor.", "Isso custará vinte dólares senhor.", "pt"),
        ("Isso custará 20€ senhor.", "Isso custará vinte euros senhor.", "pt"),
        (
            "Isso custará 20,15€ senhor.",
            "Isso custará vinte euros e quinze cêntimos senhor.",
            "pt",
        ),  # "cêntimos" should be "centavos" num2words issue
        # Polish
        ("W 12,5 sekundy.", "W dwanaście przecinek pięć sekundy.", "pl"),
        ("Było 50 żołnierzy.", "Było pięćdziesiąt żołnierzy.", "pl"),
        ("To będzie kosztować 20€ panie.", "To będzie kosztować dwadzieścia euro panie.", "pl"),
        ("To będzie kosztować 20,15€ panie.", "To będzie kosztować dwadzieścia euro, piętnaście centów panie.", "pl"),
        # Arabic
        ("في الـ 12,5 ثانية.", "في الـ اثنا عشر  , خمسون ثانية.", "ar"),
        ("كان هناك 50 جنديًا.", "كان هناك خمسون جنديًا.", "ar"),
        # ("ستكون النتيجة $20 يا سيد.", 'ستكون النتيجة عشرون دولار يا سيد.', 'ar'), # $ and € are mising from num2words
        # ("ستكون النتيجة 20€ يا سيد.", 'ستكون النتيجة عشرون يورو يا سيد.', 'ar'),
        # Czech
        ("Za 12,5 vteřiny.", "Za dvanáct celá pět vteřiny.", "cs"),
        ("Bylo tam 50 vojáků.", "Bylo tam padesát vojáků.", "cs"),
        ("To bude stát 20€ pane.", "To bude stát dvacet euro pane.", "cs"),
        ("To bude 20.15€ pane.", "To bude dvacet euro, patnáct centů pane.", "cs"),
        # Russian
        ("Через 12.5 секунды.", "Через двенадцать запятая пять секунды.", "ru"),
        ("Там было 50 солдат.", "Там было пятьдесят солдат.", "ru"),
        ("Это будет 20.15€ сэр.", "Это будет двадцать евро, пятнадцать центов сэр.", "ru"),
        ("Это будет стоить 20€ господин.", "Это будет стоить двадцать евро господин.", "ru"),
        # Dutch
        ("In 12,5 seconden.", "In twaalf komma vijf seconden.", "nl"),
        ("Er waren 50 soldaten.", "Er waren vijftig soldaten.", "nl"),
        ("Dat wordt dan $20 meneer.", "Dat wordt dan twintig dollar meneer.", "nl"),
        ("Dat wordt dan 20€ meneer.", "Dat wordt dan twintig euro meneer.", "nl"),
        # Chinese (Simplified)
        ("在12.5秒内", "在十二点五秒内", "zh"),
        ("有50名士兵", "有五十名士兵", "zh"),
        # ("那将是$20先生", '那将是二十美元先生', 'zh'), currency doesn't work
        # ("那将是20€先生", '那将是二十欧元先生', 'zh'),
        # Turkish
        # ("12,5 saniye içinde.", 'On iki virgül beş saniye içinde.', 'tr'), # decimal doesn't work for TR
        ("50 asker vardı.", "elli asker vardı.", "tr"),
        ("Bu 1. test", "Bu birinci test", "tr"),
        # ("Bu 100.000,5.", 'Bu yüz bin virgül beş.', 'tr'),
        # Hungarian
        ("12,5 másodperc alatt.", "tizenkettő egész öt tized másodperc alatt.", "hu"),
        ("50 katona volt.", "ötven katona volt.", "hu"),
        ("Ez az 1. teszt", "Ez az első teszt", "hu"),
        # Korean
        ("12.5 초 안에.", "십이 점 다섯 초 안에.", "ko"),
        ("50 명의 병사가 있었다.", "오십 명의 병사가 있었다.", "ko"),
        ("이것은 1 번째 테스트입니다", "이것은 첫 번째 테스트입니다", "ko"),
        # Hindi
        ("12.5 सेकंड में।", "साढ़े बारह सेकंड में।", "hi"),
        ("50 सैनिक थे।", "पचास सैनिक थे।", "hi"),
    ]
    for a, b, lang in test_cases:
        out = expand_numbers_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"


def test_abbreviations_multilingual():
    test_cases = [
        # English
        ("Hello Mr. Smith.", "Hello mister Smith.", "en"),
        ("Dr. Jones is here.", "doctor Jones is here.", "en"),
        # Spanish
        ("Hola Sr. Garcia.", "Hola señor Garcia.", "es"),
        ("La Dra. Martinez es muy buena.", "La doctora Martinez es muy buena.", "es"),
        # French
        ("Bonjour Mr. Dupond.", "Bonjour monsieur Dupond.", "fr"),
        ("Mme. Moreau est absente aujourd'hui.", "madame Moreau est absente aujourd'hui.", "fr"),
        # German
        ("Frau Dr. Müller ist sehr klug.", "Frau doktor Müller ist sehr klug.", "de"),
        # Portuguese
        ("Olá Sr. Silva.", "Olá senhor Silva.", "pt"),
        ("Dra. Costa, você está disponível?", "doutora Costa, você está disponível?", "pt"),
        # Italian
        ("Buongiorno, Sig. Rossi.", "Buongiorno, signore Rossi.", "it"),
        # ("Sig.ra Bianchi, posso aiutarti?", 'signora Bianchi, posso aiutarti?', 'it'), # Issue with matching that pattern
        # Polish
        ("Dzień dobry, P. Kowalski.", "Dzień dobry, pani Kowalski.", "pl"),
        ("M. Nowak, czy mogę zadać pytanie?", "pan Nowak, czy mogę zadać pytanie?", "pl"),
        # Czech
        ("P. Novák", "pan Novák", "cs"),
        ("Dr. Vojtěch", "doktor Vojtěch", "cs"),
        # Dutch
        ("Dhr. Jansen", "de heer Jansen", "nl"),
        ("Mevr. de Vries", "mevrouw de Vries", "nl"),
        # Russian
        ("Здравствуйте Г-н Иванов.", "Здравствуйте господин Иванов.", "ru"),
        ("Д-р Смирнов здесь, чтобы увидеть вас.", "доктор Смирнов здесь, чтобы увидеть вас.", "ru"),
        # Turkish
        ("Merhaba B. Yılmaz.", "Merhaba bay Yılmaz.", "tr"),
        ("Dr. Ayşe burada.", "doktor Ayşe burada.", "tr"),
        # Hungarian
        ("Dr. Szabó itt van.", "doktor Szabó itt van.", "hu"),
    ]

    for a, b, lang in test_cases:
        out = expand_abbreviations_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"


def test_symbols_multilingual():
    test_cases = [
        ("I have 14% battery", "I have 14 percent battery", "en"),
        ("Te veo @ la fiesta", "Te veo arroba la fiesta", "es"),
        ("J'ai 14° de fièvre", "J'ai 14 degrés de fièvre", "fr"),
        ("Die Rechnung beträgt £ 20", "Die Rechnung beträgt pfund 20", "de"),
        ("O meu email é ana&joao@gmail.com", "O meu email é ana e joao arroba gmail.com", "pt"),
        ("linguaggio di programmazione C#", "linguaggio di programmazione C cancelletto", "it"),
        ("Moja temperatura to 36.6°", "Moja temperatura to 36.6 stopnie", "pl"),
        ("Mám 14% baterie", "Mám 14 procento baterie", "cs"),
        ("Těším se na tebe @ party", "Těším se na tebe na party", "cs"),
        ("У меня 14% заряда", "У меня 14 процентов заряда", "ru"),
        ("Я буду @ дома", "Я буду собака дома", "ru"),
        ("Ik heb 14% batterij", "Ik heb 14 procent batterij", "nl"),
        ("Ik zie je @ het feest", "Ik zie je bij het feest", "nl"),
        ("لدي 14% في البطارية", "لدي 14 في المئة في البطارية", "ar"),
        ("我的电量为 14%", "我的电量为 14 百分之", "zh"),
        ("Pilim %14 dolu.", "Pilim yüzde 14 dolu.", "tr"),
        ("Az akkumulátorom töltöttsége 14%", "Az akkumulátorom töltöttsége 14 százalék", "hu"),
        ("배터리 잔량이 14%입니다.", "배터리 잔량이 14 퍼센트입니다.", "ko"),
        ("मेरे पास 14% बैटरी है।", "मेरे पास चौदह प्रतिशत बैटरी है।", "hi"),
    ]

    for a, b, lang in test_cases:
        out = expand_symbols_multilingual(a, lang=lang)
        assert out == b, f"'{out}' vs '{b}'"


if __name__ == "__main__":
    test_expand_numbers_multilingual()
    test_abbreviations_multilingual()
    test_symbols_multilingual()