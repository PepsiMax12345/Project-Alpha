# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 20:21:18 2025

@author: ethan
"""

from EPR_data_pipeline import run_condensed_pipeline

if __name__ == "__main__":
    print("Running full analysis pipeline...")

    
    history_path = "data/history.csv"
    details_path = "data/details.csv"

    results = run_condensed_pipeline(
        
        history_path=history_path,
        details_path=details_path
    )

    print("Running")
