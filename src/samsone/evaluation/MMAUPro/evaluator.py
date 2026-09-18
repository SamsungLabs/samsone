import pandas as pd

from samsone.evaluation.base import ALMEvaluator
from samsone.evaluation.MMAUPro.evaluate_mmau_pro_comprehensive import (
    main as mmaupro_evaluate,
)


class MMAUProEvaluator(ALMEvaluator):
    model_output_column_name = "model_output"

    def __init__(
        self,
        meta_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "MMAUPro",
        use_reduced_models: bool = True,
    ):
        super().__init__(test_dataset_name=test_dataset_name)
        self.meta_file_path = meta_file_path
        self.df = pd.read_parquet(self.meta_file_path)
        self.id_column_name = id_column_name
        self.use_reduced_models = use_reduced_models

        self.results_filename = f"{self.test_dataset_name}_results.json"

    def evaluate(self, generated_answers: dict):
        self.df[MMAUProEvaluator.model_output_column_name] = self.df[
            self.id_column_name
        ].map(generated_answers)

        self.results_summary = mmaupro_evaluate(
            self.df,
            self.model_output_column_name,
            self.meta_file_path,
            use_reduced_models=self.use_reduced_models,
        )


class ReasonAQAEvaluator(ALMEvaluator):
    def __init__(self, test_dataset_name: str = "ReasonAQA"):
        super().__init__(test_dataset_name=test_dataset_name)

    def evaluate(self, generated_answers: dict):
        pass

    def finalize(self, results_dir: str):
        pass
