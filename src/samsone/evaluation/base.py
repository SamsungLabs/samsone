import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
from aac_metrics.functional import fense
from clearml import Logger
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from samsone.data.constants import META_RESPONSE_KEY


class ALMEvaluator(ABC):
    def __init__(self, test_dataset_name: str | None = None, **kwargs):
        if test_dataset_name is None:
            raise TypeError(
                f"Class {self.__name__} must pass a 'test_dataset_name' to a super class init."
            )
        self.test_dataset_name = test_dataset_name

    @abstractmethod
    def evaluate(self, generated_answers: dict):
        pass

    def finalize(self, results_dir: str):
        table = pl.from_dict(self.results_summary)
        Logger.current_logger().report_table(
            title="Results",
            series=self.test_dataset_name,
            iteration=None,
            table_plot=table.to_pandas(),
        )

        output_filename = Path(results_dir) / self.results_filename

        with open(output_filename, "w") as f:
            json.dump(self.results_summary, f, indent=2, default=float)


class MCQEvaluator(ALMEvaluator):
    def __init__(
        self,
        meta_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "MCQDataset",
    ):
        super().__init__(test_dataset_name=test_dataset_name)
        self.meta_file_path = meta_file_path
        self.df = pl.read_parquet(self.meta_file_path)
        self.id_column_name = id_column_name

        self.results_filename = f"{self.test_dataset_name}_results.json"

    def evaluate(self, generated_answers: dict):
        output_df = self.df.with_columns(
            output=pl.col(self.id_column_name).replace_strict(generated_answers)
        ).with_columns(
            y_true=pl.col(
                META_RESPONSE_KEY
            )  # split answers like "b) A Man" by the ')' symbol to extract the chosen letter
            .str.split(")")
            .list.get(0)
            .str.to_lowercase(),
            y_pred=pl.col("output").str.split(")").list.get(0).str.to_lowercase(),
        )

        acc = accuracy_score(
            output_df.get_column("y_true"), output_df.get_column("y_pred")
        )
        precision = precision_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
            average="weighted",
        )
        recall = recall_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
            average="weighted",
        )
        f1 = f1_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
            average="weighted",
        )

        self.results_summary = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
        print(f"***** {self.test_dataset_name} *****")
        print(self.results_summary)


class BinaryAQAEvaluator(ALMEvaluator):
    def __init__(
        self,
        meta_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "BinaryAQADataset",
    ):
        super().__init__(test_dataset_name=test_dataset_name)
        self.meta_file_path = meta_file_path
        self.df = pl.read_parquet(self.meta_file_path)
        self.id_column_name = id_column_name

        self.results_filename = f"{self.test_dataset_name}_results.json"

    def evaluate(self, generated_answers: dict):
        output_df = self.df.with_columns(
            output=pl.col(self.id_column_name).replace_strict(generated_answers)
        ).with_columns(
            y_true=pl.when(
                pl.col(META_RESPONSE_KEY) == "yes"
            )  # change "yes/no" answers to 1/0 for binary classification metrics
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
            y_pred=pl.when(pl.col("output") == "yes")
            .then(pl.lit(1))
            .otherwise(pl.lit(0)),
        )

        acc = accuracy_score(
            output_df.get_column("y_true"), output_df.get_column("y_pred")
        )
        precision = precision_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
        )
        recall = recall_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
        )
        f1 = f1_score(
            output_df.get_column("y_true"),
            output_df.get_column("y_pred"),
        )

        self.results_summary = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
        print(f"***** {self.test_dataset_name} *****")
        print(self.results_summary)


class CaptioningEvaluator(ALMEvaluator):
    def __init__(
        self,
        meta_file_path: str,
        id_column_name: str = "id",
        test_dataset_name: str = "CaptioningDataset",
        group_by_column_name: str | None = None,
        strip_prefix: str = "",
    ):
        super().__init__(test_dataset_name=test_dataset_name)
        self.meta_file_path = meta_file_path
        self.df = pl.read_parquet(self.meta_file_path)
        self.id_column_name = id_column_name
        self.group_by_column_name = group_by_column_name
        self.strip_prefix = strip_prefix

        self.results_filename = f"{self.test_dataset_name}_results.json"

    def evaluate(self, generated_answers: dict):
        output_df = self.df.with_columns(
            pl.col(self.id_column_name)
            .replace_strict(generated_answers)
            .str.strip_prefix(self.strip_prefix)
            .alias("output"),
            pl.col(META_RESPONSE_KEY)
            .str.strip_prefix(self.strip_prefix)
            .alias(META_RESPONSE_KEY),
        )

        if self.group_by_column_name:
            # group by first element of filepaths list (1 element list in case of clotho)
            output_df = output_df.group_by(
                pl.col(self.group_by_column_name).list.first().alias("group_key")
            ).agg(pl.all())
        else:
            # ensure type compatibility
            output_df = (
                output_df.with_row_index(name="idx")
                .group_by(pl.col("idx"))
                .agg(pl.all())
            )

        self.results_summary = {}

        gts = {}
        res = {}

        ground_truths = output_df.get_column(META_RESPONSE_KEY).to_list()
        generated = output_df.get_column("output").to_list()
        ids = output_df.get_column(self.id_column_name).to_list()

        for i, (gt, gen, sample_id) in enumerate(zip(ground_truths, generated, ids)):
            assert len(gt) == len(gen) == len(sample_id), (
                "There should be an equal number of examples per filepath!"
            )
            gts[i] = [{"caption": gt[n]} for n in range(len(gt))]
            res[i] = [{"caption": gen[0]}]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        scorers_dict = {
            "BLEU": Bleu,
            "METEOR": Meteor,
            "ROUGE_L": Rouge,
            "CIDEr": Cider,
            "SPICE": Spice,
        }

        for metric, scorer in scorers_dict.items():
            if metric == "BLEU":
                scorer = Bleu(4)
                score, _ = scorer.compute_score(gts, res)
                for i in range(1, 5):
                    self.results_summary[f"BLEU_{i}"] = score[i - 1]
            else:
                scorer = scorer()
                self.results_summary[metric] = scorer.compute_score(gts, res)[0]

        fense_scores = fense(
            candidates=[gen[0] for gen in generated],
            mult_references=ground_truths,
            return_all_scores=False,
        )
        self.results_summary["FENSE"] = fense_scores.item()

        print(f"***** {self.test_dataset_name} *****")
        print(self.results_summary)
