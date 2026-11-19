"""نمودارهای TF-IDF. فقط رسم می‌کند؛ محاسبه در tfidf_service است."""
import matplotlib

matplotlib.use("Agg")
import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd 
from bidi.algorithm import get_display

FONT_FAMILY = ["Tahoma", "Vazirmatn", "Arial", "DejaVu Sans"]


def setup_fonts() -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False


def fa(text: str) -> str:
    """شکل‌دهی متن فارسی برای matplotlib."""
    return get_display(arabic_reshaper.reshape(str(text))[::-1])


def plot_terms(result: pd.DataFrame, global_mean: float, out_path: str,
               top_n: int = 25, min_count: int = 50) -> None:
    shown = result[result["comment_count"] >= min_count].copy()
    shown["deviation"] = shown["smoothed_score"] - global_mean
    plot_df = pd.concat([shown.head(top_n), shown.tail(top_n)]).sort_values("deviation")

    labels = [fa(t) + f"  ({n})" for t, n in zip(plot_df["term"], plot_df["comment_count"])]
    colors = ["#d9534f" if d < 0 else "#5cb85c" for d in plot_df["deviation"]]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(labels, plot_df["deviation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(fa(f"اختلاف با میانگین کل ({global_mean:.2f}) - داخل پرانتز: تعداد کامنت"))
    ax.set_title(fa("کلماتی که امتیاز کامنت را بالا یا پایین می‌برند"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(result: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(result["comment_count"], result["smoothed_score"], s=15, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(fa("تعداد کامنت‌های دارای کلمه (لگاریتمی)"))
    ax.set_ylabel(fa("امتیاز میانگین‌گیری‌شده"))
    ax.set_title(fa("هر نقطه یک کلمه است"))
    ax.axhline(result["smoothed_score"].mean(), color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)