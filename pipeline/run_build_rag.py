"""مرحله‌ی نهایی: ساخت doctors_for_rag.csv."""
from Services.data_service import DOCTORS_FOR_RAG_CSV_PATH
from Services.dataset_service import load_doctors, load_normalized_comments
from Services.keywords import ASPECT_KEYWORDS
from Services.rag_service import build_doctor_frame, prepare_comments, save_doctor_frame


def main() -> None:
    doctors = load_doctors()
    comments = prepare_comments(load_normalized_comments())

    df = build_doctor_frame(doctors, comments)
    save_doctor_frame(df, DOCTORS_FOR_RAG_CSV_PATH)

    print(f"Rows: {len(df)}")
    print(f"program_rating NaN: {df['program_rating'].isna().sum()}")
    print(f"comment_rating NaN: {df['comment_rating'].isna().sum()}")
    print(f"comment_count zero: {(df['comment_count'] == 0).sum()}")
    print(f"Aspects: {list(ASPECT_KEYWORDS)}")


if __name__ == "__main__":
    main()