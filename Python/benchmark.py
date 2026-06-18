from ultralytics.utils.benchmarks import benchmark

# Benchmark the model with specified parameters
benchmark(
    model='runs/detect/train-7/weights/best.pt',  # Path to the trained model
    data='RecipeData/data.yaml',  # Path to the dataset YAML file
    device='0',  # Use GPU 0 for benchmarking
    imgsz=640  # Image size for benchmarking
)