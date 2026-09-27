from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    data_root: Path
    output_dir: Path = Path("output")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    experiment_dir: Path = Path("experiments")
    random_state: int = 42
    device_type: str = "cuda"
    gpu_device_id: int = 0
    validation_fraction: float = 0.20
    max_candidates_per_s1: int = 80
    tfidf_top_k: int = 30
    exact_candidate_limit: int = 80
    token_candidate_limit: int = 20
    prefix_candidate_limit: int = 10
    blocking_prefix_length: int = 4
    max_name_block_tokens: int = 5
    max_address_block_tokens: int = 6
    exact_family_fraction: float = 0.20
    name_family_fraction: float = 0.25
    address_family_fraction: float = 0.25
    number_family_fraction: float = 0.20
    train_negative_multiplier: int = 6
    min_candidate_score: float = 0.0
    threshold_low: float = 0.50
    threshold_high: float = 0.995
    threshold_step: float = 0.01

    def prepare(self):
        for p in (self.output_dir, self.model_dir, self.report_dir, self.experiment_dir):
            p.mkdir(parents=True, exist_ok=True)
