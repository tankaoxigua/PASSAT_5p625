import csv
import os


class ExperimentLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path

        self.fieldnames = [
            "epoch",
            "space_method",
            "lmax",
            "integrator",
            "dt",
            "step",
            "rmse",
            "std_rmse",
            "acc",
            "alloc_mem_MB",
            "peak_mem_MB",
            "time_per_epoch",
        ]

        file_exists = os.path.exists(csv_path)

        self.file = open(csv_path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists:
            self.writer.writeheader()

    def log(self, log_dict):
        # 自动补全字段（防止缺字段报错）
        for key in self.fieldnames:
            if key not in log_dict:
                log_dict[key] = None

        self.writer.writerow(log_dict)
        self.file.flush()

    def close(self):
        self.file.close()
