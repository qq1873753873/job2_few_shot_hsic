import kagglehub

# Download latest version
path = kagglehub.dataset_download("mingliu123/chikusei-imdb-128")

print("Path to dataset files:", path)