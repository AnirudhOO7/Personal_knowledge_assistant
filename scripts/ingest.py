from app.ingestion.loader import load_file

text = load_file("data/raw/test.pdf")

print("=== OUTPUT ===")
print(text)
print("=== LENGTH ===")
print(len(text))