#!/usr/bin/env python3
"""
Data Integration Module
========================
Monitors and integrates new files into the HOI4 vector database.
"""

import os
import json
import logging
from hoi4_vector_db import HOI4VectorUploader, HOI4VectorDB

class DataDirectoryIntegrator:
    def __init__(self, vector_db, data_dir="data"):
        self.vector_db = vector_db
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def check_and_update(self):
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt") or f.endswith(".json")]
        uploader = HOI4VectorUploader(self.vector_db)
        for file in files:
            uploader.ingest_from_json(os.path.join(self.data_dir, file))
