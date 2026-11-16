"""Sections:
    backbones.py   turns image into an embedding vector
    data.py        finds images on disk and preps them for the network
    classifier.py  compares embeddings to the labelled examples and names the object
    profiler.py    measures latency, throughput and memory while the above runs
    report.py      writes those measurements to json/csv/md files
"""

__version__ = "1.0.0"
