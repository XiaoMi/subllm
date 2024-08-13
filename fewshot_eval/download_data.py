import os
import zipfile
from tqdm import tqdm
from huggingface_hub import snapshot_download

evaluation_dir = "./"

snapshot_download(
    repo_id="yuzhaouoe/eval_data",
    repo_type="model",
    allow_patterns="eval_data.zip",
    local_dir=evaluation_dir,
)
compressed_eval_data_path = os.path.join(evaluation_dir, "eval_data.zip")

eval_data_dir = os.path.join(evaluation_dir, "eval_data")
if not os.path.exists(eval_data_dir):
    os.mkdir(eval_data_dir)

compressed_eval_data = zipfile.ZipFile(compressed_eval_data_path, 'r')
for file in tqdm(compressed_eval_data.namelist()):
    compressed_eval_data.extract(file, eval_data_dir)
compressed_eval_data.close()
os.remove(compressed_eval_data_path)
