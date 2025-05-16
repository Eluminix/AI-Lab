from utils import extract_data_from_raw

# Beispielpfade – anpassen an eure Ordnerstruktur
data_link_file = "data/train_links.csv"
raw_data_dir = "jsons"
manual_file = "manual_crawl.json"  # kann leer sein
output_csv = "data/train.csv"

extract_data_from_raw(data_link_file, raw_data_dir, manual_file, output_csv)
