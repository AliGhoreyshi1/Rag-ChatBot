
import ast
import difflib
import json
import re

import pandas as pd

from Services.data_service import DOCTORS_FOR_RAG_CSV_PATH, DOCTORS_RAW_CANONICAL_CSV_PATH



RAG_CSV = DOCTORS_FOR_RAG_CSV_PATH            
INFO_CSV = DOCTORS_RAW_CANONICAL_CSV_PATH     

TOP_N = 3                                     

RATING_COL = "comment_rating"
COUNT_COL = "comment_count"


USE_BAYES = True
PRIOR_COUNT = 10

MIN_COMMENTS = 0            
NAME_MATCH_THRESHOLD = 0.85  

ASPECTS = {
    "waiting":     ("waiting_score", "waiting_count"),           
    "price":       ("price_score", "price_count"),               
    "behavior":    ("behavior_score", "behavior_count"),         
    "explanation": ("explanation_score", "explanation_count"),   
    "result":      ("result_score", "result_count"),             
    "staff":       ("staff_score", "staff_count"),               
    "quality":     ("quality_score", "quality_count"),           
}


ASPECT_ALIASES = {
    "waiting":     ["waiting", "انتظار", "معطل", "معطلی", "صف"],                       
    "price":       ["price", "هزینه", "قیمت", "ویزیت", "ارزان", "گران"],                
    "behavior":    ["behavior", "رفتار", "برخورد", "اخلاق"],                           
    "explanation": ["explanation", "توضیح"],                                           
    "result":      ["result", "نتیجه", "درمان", "بهبود", "جواب"],                       
    "staff":       ["staff", "کارکنان", "کارمند", "کادر", "منشی", "پرسنل", "پذیرش"],    
    "quality":     ["quality", "کیفیت"],                                              
}

GENERAL_QUALITY_WORDS = ["خوب", "خوبه", "خوبی", "عالی", "عالیه", "بد", "بده", "بدی",
                         "بهترین", "ضعیف", "راضی"]
ASPECT_WEIGHT = 0.7

ASPECT_PRIOR_COUNT = 3

NAME_STOPWORDS = {"دکتر", "دکترا", "خانم", "اقای", "استاد", "پروفسور", "جناب", "سرکار", "dr"}


SPECIALTY_GROUPS = [
    {   # مغز و اعصاب
        "aliases": ["مغز", "اعصاب", "نورولوژی", "نورولوژیست"],
        "keywords": ["مغز", "اعصاب"],
    },
    {   # ارتوپدی
        "aliases": ["ارتوپد", "اورتوپد", "استخوان", "مفصل", "مفاصل", "ستون فقرات", "زانو", "طب فیزیکی"],
        "keywords": ["ارتوپد", "اورتوپد", "استخوان", "مفصل", "مفاصل", "زانو", "شانه", "ارنج", "لگن",
                     "ارتروسکوپی", "جراحی دست", "پا و مچ پا", "اسکلتی", "ستون فقرات", "طب فیزیکی"],
    },
    {   # زنان و زایمان
        "aliases": ["زنان", "زایمان", "نازایی", "نازائی", "ناباروری", "مامایی"],
        "keywords": ["زنان", "زایمان", "نازایی", "نازائی", "پریناتولوژی", "جنین", "وی اف"],
    },
    {   # گوش و حلق و بینی
        "aliases": ["گوش", "حلق", "بینی", "حنجره", "سینوس"],
        "keywords": ["گوش", "حلق", "بینی", "حنجره", "سینوس"],
    },
    {   # پوست
        "aliases": ["پوست", "درماتولوژی"],
        "keywords": ["پوست", "درماتولوژی"],
    },
    {   # کلیه و مجاری ادراری
        "aliases": ["کلیه", "مجاری ادرار", "ارولوژی", "اورولوژی", "ارولوژیست", "اورولوژیست"],
        "keywords": ["رولوژی", "کلیه", "دستگاه ادراری", "مجاری ادرار"],
    },
]



_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize_text(text) -> str:
  
    text = str(text).translate(_DIGITS)
    text = (text.replace("ي", "ی").replace("ك", "ک")
                .replace("ۀ", "ه").replace("ة", "ه")
                .replace("آ", "ا"))
    text = text.replace("\u200c", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def name_tokens(text) -> list:
    return [t for t in normalize_text(text).split() if t and t not in NAME_STOPWORDS]


def parse_literal(value, default):
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def unique_phones(phones: list) -> list:
    seen, result = set(), []
    for p in phones:
        key = re.sub(r"\D", "", str(p).translate(_DIGITS))
        if key and key not in seen:
            seen.add(key)
            result.append(p)
    return result


def clean_number(value):
    return None if pd.isna(value) else float(value)




class DoctorRecommender:

    def __init__(self):
        rag = pd.read_csv(RAG_CSV)
        info = pd.read_csv(INFO_CSV)


        rag["doctor_id"] = rag["doctor_id"].astype(str)
        info["doctor_id"] = info["doctor_id"].astype(str)
        info = info.drop_duplicates("doctor_id")


        info = info[["doctor_id", "rating_score_5", "offices", "phones"]]
        df = rag.merge(info, on="doctor_id", how="left")

        missing = df["offices"].isna().sum()
        if missing:
            print(f"هشدار: {missing} دکتر در فایل اطلاعات پیدا نشد (آدرس و تلفن ندارند).")


        df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce").fillna(0)
        df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")


        self.global_mean = float(df[RATING_COL].mean())


        rating = df[RATING_COL].fillna(self.global_mean)
        if USE_BAYES:
            v = df[COUNT_COL]
            df["rank_score"] = (v / (v + PRIOR_COUNT)) * rating + (PRIOR_COUNT / (v + PRIOR_COUNT)) * self.global_mean
        else:
            df["rank_score"] = rating

        self.aspects_available = []
        for key, (score_col, count_col) in ASPECTS.items():
            if score_col not in df.columns or count_col not in df.columns:
                print(f"هشدار: ستون‌های {score_col} / {count_col} در فایل نیست؛ بخش «{key}» کنار گذاشته شد.")
                continue
            score = pd.to_numeric(df[score_col], errors="coerce")
            n = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
            mean = float(score.mean())
            if pd.isna(mean):
                continue
            df[f"aspect_{key}"] = ((n / (n + ASPECT_PRIOR_COUNT)) * score.fillna(mean)
                                   + (ASPECT_PRIOR_COUNT / (n + ASPECT_PRIOR_COUNT)) * mean)
            self.aspects_available.append(key)

        
        df["_spec_norm"] = df["specialty"].fillna("").map(normalize_text)
        df["_name_tokens"] = df["name"].fillna("").map(name_tokens)

        self.df = df
        print(f"{len(df)} دکتر بارگذاری شد.")

    
    @staticmethod
    def _specialty_keywords(specialty: str) -> list:
        q = normalize_text(specialty)
        keywords = []
        for group in SPECIALTY_GROUPS:
            if any(normalize_text(a) in q for a in group["aliases"]):
                keywords += group["keywords"]
        return [normalize_text(k) for k in keywords] if keywords else [q]

    
    def _search_by_name(self, doctor_name: str) -> pd.DataFrame:
        q_tokens = name_tokens(doctor_name)
        if not q_tokens:
            return self.df.iloc[0:0]

        scores = []
        for idx, tokens in self.df["_name_tokens"].items():
            if not tokens:
                continue
            per_token = [
                max(difflib.SequenceMatcher(None, qt, nt).ratio() for nt in tokens)
                for qt in q_tokens
            ]
            if min(per_token) >= NAME_MATCH_THRESHOLD:          
                scores.append((idx, sum(per_token) / len(per_token)))

        if not scores:
            return self.df.iloc[0:0]
        order = pd.DataFrame(scores, columns=["idx", "name_match"]).set_index("idx")
        found = self.df.loc[order.index].copy()
        found["name_match"] = order["name_match"]
        return found.sort_values(["name_match", "rank_score"], ascending=False).head(TOP_N)

    
    @staticmethod
    def _format_doctor(row) -> dict:
        offices_raw = parse_literal(row["offices"], [])
        return {
            "name": row["name"],                       
            "specialty": row["specialty"],             
            "rating": clean_number(row[RATING_COL]),   
            "offices": [                               
                o.get("address")
                for o in offices_raw
                if isinstance(o, dict) and o.get("address")
            ],
        }

    def _resolve_aspects(self, aspects) -> list:
        if not aspects:
            return []
        if isinstance(aspects, str):
            aspects = [aspects]

        resolved = []
        for text in aspects:
            t = normalize_text(text)
            for key, aliases in ASPECT_ALIASES.items():
                if (key in self.aspects_available and key not in resolved
                        and any(normalize_text(a) in t for a in aliases)):
                    resolved.append(key)

        if not resolved and "quality" in self.aspects_available:
            tokens = set()
            for text in aspects:
                tokens.update(normalize_text(text).split())
            if tokens & {normalize_text(w) for w in GENERAL_QUALITY_WORDS}:
                resolved.append("quality")

        return resolved

    
    def recommend(self, specialty: str = None, doctor_name: str = None, aspects=None) -> dict:
        if doctor_name and doctor_name.strip():
            found = self._search_by_name(doctor_name)
            if len(found):
                return {"query_type": "name",
                        "doctors": [self._format_doctor(r) for _, r in found.iterrows()]}

        
        if specialty and specialty.strip():
            keywords = self._specialty_keywords(specialty)
            mask = self.df["_spec_norm"].map(lambda s: any(k in s for k in keywords))
            pool = self.df[mask]
            pool = pool[pool[COUNT_COL] >= MIN_COMMENTS]

            if len(pool):
                
                aspect_keys = self._resolve_aspects(aspects)
                if aspect_keys:
                    aspect_mean = pool[[f"aspect_{k}" for k in aspect_keys]].mean(axis=1)
                    final = ASPECT_WEIGHT * aspect_mean + (1 - ASPECT_WEIGHT) * pool["rank_score"]
                else:
                    final = pool["rank_score"]

                pool = pool.assign(final_score=final)
                top = pool.sort_values(["final_score", COUNT_COL], ascending=False).head(TOP_N)
                return {"query_type": "specialty",
                        "aspects_used": aspect_keys,
                        "doctors": [self._format_doctor(r) for _, r in top.iterrows()]}

        return {"query_type": "none", "doctors": [],
                "note": "no doctor with a matching specialty in the database"}


    
    def recommend_from_text(self, text: str) -> dict:
        
        text = (text or "").strip()
        if not text:
            return {"query_type": "none", "doctors": [], "note": "empty input"}

        q = normalize_text(text)
        has_specialty = any(normalize_text(a) in q
                            for g in SPECIALTY_GROUPS for a in g["aliases"])

        if has_specialty:
            
            return self.recommend(specialty=text, aspects=[text])

        return self.recommend(doctor_name=text)


# API عمومی — این دو تابع همان چیزی‌اند که دیگران استفاده می‌کنند
# ======================================================================
__all__ = ["get_recommendations", "get_recommendations_json", "DoctorRecommender"]

_bot = None


def _get_bot() -> "DoctorRecommender":
    global _bot
    if _bot is None:
        _bot = DoctorRecommender()
    return _bot


def get_recommendations(text: str) -> dict:
 
    return _get_bot().recommend_from_text(text)


def get_recommendations_json(text: str) -> str:
    
    return json.dumps(get_recommendations(text), ensure_ascii=False)


# راهنمای استفاده اسان از کد 
# from doctor_rag_v2 import get_recommendations, get_recommendations_json
# result = get_recommendations("ارتوپدی معطل نکنه") -> # خروجی به‌صورت دیکشنری
# text = get_recommendations_json("ارتوپدی معطل نکنه") -> # خروجی به‌صورت رشتهٔ JSON (مناسب برگرداندن به LLM)