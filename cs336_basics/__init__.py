import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# nohup ./train_cloud.sh >/tmp/train.log 2>&1 &