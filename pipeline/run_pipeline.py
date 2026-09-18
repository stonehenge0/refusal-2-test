import torch
import random
import json
import os
import argparse
import datetime
import logging as python_logging

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.translate import translate

from pipeline.evaluator.evalharness import LMEvalHarness

import mmengine
from pipeline.utils.hook_utils import add_hooks
import jsonpickle
import sys
import pandas as pd
from transformers import logging
from datetime import datetime

start = datetime.now()

os.environ["TQDM_DISABLE"] = "1"

python_logging.getLogger("transformers").setLevel(python_logging.ERROR)
python_logging.getLogger("transformers.generation").setLevel(python_logging.ERROR)
python_logging.getLogger("accelerate").setLevel(python_logging.ERROR)


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config"
    )
    parser.add_argument(
        "--model_path", type=str, required=False, default=None, help="Path to the model"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=None, help="Batch size"
    )
    return parser.parse_args()


logging.set_verbosity_error()


class Logger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_file, "a", buffering=1)  # Line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(cfg.random_seed)
    data1_train = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_1, split="train", instructions_only=True
        ),
        cfg.n_train,
    )
    data2_train = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_2, split="train", instructions_only=True
        ),
        cfg.n_train,
    )
    data3_train = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_3, split="train", instructions_only=True
        ),
        cfg.n_train,
    )

    data1_val = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_1, split="val", instructions_only=True
        ),
        cfg.n_val,
    )
    data2_val = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_2, split="val", instructions_only=True
        ),
        cfg.n_val,
    )
    data3_val = random.sample(
        load_dataset_split(
            harmtype=cfg.harmtype_3, split="val", instructions_only=True
        ),
        cfg.n_val,
    )
    return data1_train, data2_train, data3_train, data1_val, data2_val, data3_val


def filter_data(
    cfg,
    model_base,
    harmful_train,
    harmless_train,
    harmful_val,
    harmless_val,
    or_train,
    or_val,
    system_harm,
    system_or,
    system_harmless,
):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [
            inst
            for inst, score in zip(dataset, scores.tolist())
            if comparison(score, threshold)
        ]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(
            model_base.model,
            harmful_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_harm,
            batch_size=cfg.batch_size,
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model,
            harmless_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_harmless,
            batch_size=cfg.batch_size,
        )
        harmful_train = filter_examples(
            harmful_train, harmful_train_scores, 0, lambda x, y: x > y
        )
        harmless_train = filter_examples(
            harmless_train, harmless_train_scores, 0, lambda x, y: x < y
        )

    if cfg.filter_or:
        or_train_scores = get_refusal_scores(
            model_base.model,
            or_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_or,
            batch_size=cfg.batch_size,
        )
        or_train = filter_examples(or_train, or_train_scores, 0, lambda x, y: x > y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(
            model_base.model,
            harmful_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_harm,
            batch_size=cfg.batch_size,
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model,
            harmless_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_harmless,
            batch_size=cfg.batch_size,
        )
        or_val_scores = get_refusal_scores(
            model_base.model,
            or_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            system=system_or,
            batch_size=cfg.batch_size,
        )
        harmful_val = filter_examples(
            harmful_val, harmful_val_scores, 0, lambda x, y: x > y
        )
        harmless_val = filter_examples(
            harmless_val, harmless_val_scores, 0, lambda x, y: x < y
        )
        or_val = filter_examples(or_val, or_val_scores, 0, lambda x, y: x > y)

    return harmful_train, harmless_train, harmful_val, harmless_val, or_train, or_val


def generate_and_save_candidate_directions(
    cfg, model_base, harmful_train, harmless_train, system
):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path, "generate_directions")):
        os.makedirs(os.path.join(cfg.artifact_path, "generate_directions"))

    mean_diffs = generate_directions(
        system,
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path, "generate_directions"),
        batch_size=cfg.batch_size,
    )

    torch.save(
        mean_diffs, os.path.join(cfg.artifact_path, "generate_directions/mean_diffs.pt")
    )

    return mean_diffs


def select_and_save_direction(
    cfg,
    model_base,
    harmful_val,
    harmless_val,
    candidate_directions,
    pair_name,
    mode,
    kl_threshold,
    top_n,
):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path, "select_direction")):
        os.makedirs(os.path.join(cfg.artifact_path, "select_direction"))

    pos, layer, direction = select_direction(
        cfg,
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        pair_name,
        kl_threshold=kl_threshold,
        artifact_dir=os.path.join(cfg.artifact_path, "select_direction"),
        mode=mode,
        top_n=top_n,
        batch_size=cfg.batch_size,
    )

    with open(f"{cfg.artifact_path}/direction_metadata_{mode}.json", "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f"{cfg.artifact_path}/direction_{mode}.pt")

    return pos, layer, direction


def generate_and_save_completions_for_dataset(
    cfg,
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    intervention_label,
    dataset_name,
    dataset=None,
    system=None,
):
    """Generate and save completions for a dataset."""

    print(
        f"-----Generating completions for {dataset_name}, {intervention_label} (generate and save completions for dataset)"
    )
    if not os.path.exists(os.path.join(cfg.artifact_path, "completions")):
        os.makedirs(os.path.join(cfg.artifact_path, "completions"))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        system=system,
    )

    with open(
        f"{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_completions.json",
        "w",
    ) as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)


def translate_and_save_translated_responses_for_dataset(
    cfg, dataset_name, intervention_label
):
    """Translates the output from Bengali, Japanes etc. to English for evaluation."""
    print(
        f"-----Calling translate and save translated responses for {dataset_name}, {intervention_label}"
    )
    completions_df = pd.read_json(
        os.path.join(
            cfg.artifact_path,
            f"completions/{dataset_name}_{intervention_label}_completions.json",
        )
    )

    translated_df = translate(cfg, completions_df)
    translated_completions = translated_df.to_dict(orient="records")

    with open(
        f"{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_completions.json",
        "w",
    ) as f:
        json.dump(translated_completions, f, indent=4, ensure_ascii=False)


def evaluate_completions_and_save_results_for_dataset(
    cfg, intervention_label, dataset_name, eval_methodologies
):
    """Evaluate completions and save results for a dataset."""
    print(
        f"-----Evaluating completions for {dataset_name}, {intervention_label} using ({eval_methodologies}(evaluate completions and save results for dataset defined in pipeline.))"
    )

    with open(
        os.path.join(
            cfg.artifact_path,
            f"completions/{dataset_name}_{intervention_label}_completions.json",
        ),
        "r",
    ) as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(
            cfg.artifact_path,
            "completions",
            f"{dataset_name}_{intervention_label}_evaluations.json",
        ),
    )

    with open(
        f"{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_evaluations.json",
        "w",
    ) as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)


def eval_harness(cfg, model_base, identifier):

    eval_harness_evaluator_mmlu = LMEvalHarness(cfg.eval_harness_mmlu)
    lm_eval_results_mmlu = eval_harness_evaluator_mmlu.evaluate(model=model_base)
    if not os.path.exists(f"{cfg.artifact_path}/lm_eval_results"):
        os.makedirs(f"{cfg.artifact_path}/lm_eval_results")
    # save to file
    with open(f"{cfg.artifact_path}/lm_eval_results/{identifier}_mmlu.json", "w") as f:
        f.write(jsonpickle.encode(lm_eval_results_mmlu, indent=4))

    if "eval_harness" in cfg:
        eval_harness_evaluator = LMEvalHarness(cfg.eval_harness)
        lm_eval_results = eval_harness_evaluator.evaluate(model=model_base)
        if not os.path.exists(f"{cfg.artifact_path}/lm_eval_results"):
            os.makedirs(f"{cfg.artifact_path}/lm_eval_results")
        # save to file
        with open(f"{cfg.artifact_path}/lm_eval_results/{identifier}.json", "w") as f:
            f.write(jsonpickle.encode(lm_eval_results, indent=4))

    return lm_eval_results_mmlu


def ortho_refusal_directions(
    cfg, candidate_directions_harm_contrast, candidate_directions_or_contrast
):

    epsilon = 1e-8  # To prevent division by zero

    # Step 1: Compute dot product between candidate_directions_harm_contrast and candidate_directions_or_contrast
    dp = torch.sum(
        candidate_directions_harm_contrast * candidate_directions_or_contrast, dim=-1
    )  # Shape: [6, 32]

    # Step 2: Compute norm squared of candidate_directions_or_contrast
    FUD_norm_sq = torch.sum(
        candidate_directions_or_contrast * candidate_directions_or_contrast, dim=-1
    )  # Shape: [6, 32]

    # Step 3: Compute scaling factor
    s = dp / (FUD_norm_sq + epsilon)  # Shape: [6, 32]

    # Step 4: Compute projection of candidate_directions_harm_contrast onto candidate_directions_or_contrast
    s_expanded = s.unsqueeze(-1)  # Shape: [6, 32, 1]
    proj = s_expanded * candidate_directions_or_contrast  # Shape: [6, 32, 4096]

    lambda_value = cfg.ortho_lambda
    # Step 5: Compute orthogonalized candidate_directions_harm_contrast
    candidate_directions_harm_contrast_orth = (
        candidate_directions_harm_contrast - lambda_value * proj
    )  # Shape: [6, 32, 4096]

    # Optional Step 6: Normalize the orthogonalized candidate_directions_harm_contrast
    harm_contrast_orth_norm = torch.sqrt(
        torch.sum(
            candidate_directions_harm_contrast_orth
            * candidate_directions_harm_contrast_orth,
            dim=-1,
        )
    )  # Shape: [6, 32]
    candidate_directions_harm_contrast_orth_normalized = (
        candidate_directions_harm_contrast_orth
        / (harm_contrast_orth_norm.unsqueeze(-1) + epsilon)
    )  # Shape: [6, 32, 4096]

    return candidate_directions_harm_contrast_orth_normalized


def run_pipeline(config_path, model_path, batch_size):
    """Run the full pipeline."""

    # cfg = Config(model_alias=model_alias, model_path=model_path)
    cfg = mmengine.Config.fromfile(config_path)
    if model_path is not None:
        cfg.model_path = model_path
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias

    if batch_size is not None:
        cfg.batch_size = batch_size

    if "artifact_path" not in cfg:
        cfg.artifact_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "runs",
            cfg.model_alias,
            f"mode_{cfg.mode}",
            f"ortho_l{cfg.ortho_lambda}_addact_c{cfg.addact_coeff}",
        )

    sys.stdout = Logger(f"{cfg.artifact_path}/output.log")
    sys.stderr = Logger(f"{cfg.artifact_path}/error.log")

    # Redirect stdout and stderr to the log file

    cfg.dump(f"{cfg.artifact_path}/config_run.yaml")

    print("---- Starting to construct model base")
    print(datetime.now() - start)
    model_base = construct_model_base(cfg.model_path)

    # lm_eval_results_harm_ablation = eval_harness(cfg, model_base, 'or_ablation_harm_actadd' )

    # Load and sample datasets
    or_train, harmless_train, harmful_train, or_val, harmless_val, harmful_val = (
        load_and_sample_datasets(cfg)
    )
    # Filter datasets based on refusal scores
    # cfg.system = None
    harmful_train, harmless_train, harmful_val, harmless_val, or_train, or_val = (
        filter_data(
            cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
            or_train,
            or_val,
            system_harm=None,
            system_or=cfg.system,
            system_harmless=None,
        )
    )

    or_bench_hard_data = random.sample(
        load_dataset_split(harmtype="or_bench_hard", split="test"), cfg.n_test
    )
    harmful_data = random.sample(
        load_dataset_split(harmtype="harmful", split="test"), cfg.n_test
    )
    # Initialize hook lists
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    # Generate candidate directions for harmful contrast

    # candidate_directions_or_harm_contrast = generate_and_save_candidate_directions(cfg, model_base, harmful_train, or_train)[:,

    if not cfg.baseline:
        print("-----Went into the if not Baseline part")
        print(datetime.now() - start)
        candidate_directions_harm_contrast = generate_and_save_candidate_directions(
            cfg, model_base, harmful_train, harmless_train, system=None
        )[
            :,
            cfg.start_layer :,
        ]
        pair_name = ["harmful", "harmless"]
        # # Select the most effective steering direction (top n)
        positions_harm, layers_harm, directions_harm = select_and_save_direction(
            cfg,
            model_base,
            harmful_val,
            harmless_val,
            candidate_directions_harm_contrast,
            pair_name,
            "addition",
            cfg.steer_kl_threshold,
            1,
        )

        # # Generate candidate directions for false refusal contrast
        candidate_directions_or_contrast = generate_and_save_candidate_directions(
            cfg, model_base, or_train, harmless_train, system=cfg.system
        )[
            :,
            cfg.start_layer :,
        ]
        candidate_directions_or_contrast_orth_normalized = ortho_refusal_directions(
            cfg, candidate_directions_or_contrast, candidate_directions_harm_contrast
        )
        # # 2. Select the most effective refusal direction
        pair_name = ["over_refusal", "harmless"]
        # # pos_or, layer_or, direction_or = select_and_save_direction(cfg, model_base, or_val, harmless_val, candidate_directions_or_contrast, pair_name, 'ablation' ,cfg.ablate_kl_threshold)
        positions_or, layers_or, directions_or = select_and_save_direction(
            cfg,
            model_base,
            or_val,
            harmless_val,
            candidate_directions_or_contrast_orth_normalized,
            pair_name,
            "ablation",
            cfg.ablate_kl_threshold,
            cfg.top_n,
        )

        harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks = [], []
        harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks = [], []
        or_actadd_fwd_pre_hooks, or_actadd_fwd_hooks = [], []
        or_ablation_fwd_pre_hooks, or_ablation_fwd_hooks = [], []
        or_ablation_harm_actadd_fwd_pre_hooks, or_ablation_harm_actadd_fwd_hooks = (
            [],
            [],
        )

        # Process hooks for all selected harmful contrasts
        for pos_harm, layer_harm, direction_harm in zip(
            positions_harm, layers_harm, directions_harm
        ):
            # Activation addition hooks for harmful contrast
            harm_actadd_fwd_pre_hooks.append(
                (
                    model_base.model_block_modules[layer_harm],
                    get_activation_addition_input_pre_hook(
                        vector=direction_harm, coeff=+cfg.addact_coeff
                    ),
                )
            )

            # Ablation hooks for harmful contrast
            pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(
                model_base, direction_harm, cfg.start_layer, cfg.ablation_coeff
            )
            harm_ablation_fwd_pre_hooks.extend(pre_hooks)
            harm_ablation_fwd_hooks.extend(fwd_hooks)

        # Process hooks for all selected refusal contrasts
        for pos_or, layer_or, direction_or in zip(
            positions_or, layers_or, directions_or
        ):
            # Activation addition hooks for refusal contrast
            # or_actadd_fwd_pre_hooks.append(
            #     (model_base.model_block_modules[layer_or], get_activation_addition_input_pre_hook(vector=direction_or, coeff=+1.0))
            # )

            # Ablation hooks for refusal contrast
            pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(
                model_base, direction_or, cfg.start_layer
            )
            or_ablation_fwd_pre_hooks.extend(pre_hooks)
            or_ablation_fwd_hooks.extend(fwd_hooks)

        # # Combine refusal ablation and harmful addition hooks
        or_ablation_harm_actadd_fwd_pre_hooks = (
            or_ablation_fwd_pre_hooks + harm_actadd_fwd_pre_hooks
        )
        or_ablation_harm_actadd_fwd_hooks = (
            or_ablation_fwd_hooks + harm_actadd_fwd_hooks
        )
        print("-----Finished derivation of steers. Starting generation.")
        print(datetime.now() - start)

        if (
            cfg.mode == "or_ablation_harm_actadd"
        ):  # Emma: I took a bunch of benchmarks out here for speed, but only in this block
            print("------ or_ablation_harm condition triggered")
            print(datetime.now() - start)
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                or_ablation_harm_actadd_fwd_pre_hooks,
                or_ablation_harm_actadd_fwd_hooks,
                "or_ablation_harm_actadd",
                "harmful",
                dataset=harmful_data,
                system=cfg.system,
            )
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                or_ablation_harm_actadd_fwd_pre_hooks,
                or_ablation_harm_actadd_fwd_hooks,
                "or_ablation_harm_actadd",
                "or_bench_hard",
                dataset=or_bench_hard_data,
                system=cfg.system,
            )

            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                or_ablation_harm_actadd_fwd_pre_hooks,
                or_ablation_harm_actadd_fwd_hooks,
                "or_ablation_harm_actadd",
                "or_bench_multilingual",
                system=cfg.system,
            )
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                or_ablation_harm_actadd_fwd_pre_hooks,
                or_ablation_harm_actadd_fwd_hooks,
                "or_ablation_harm_actadd",
                "multijail",
                system=cfg.system,
            )
            print("----- Finished all generations, starting eval harness")
            print(datetime.now() - start)

            with add_hooks(
                module_forward_pre_hooks=or_ablation_harm_actadd_fwd_pre_hooks,
                module_forward_hooks=or_ablation_harm_actadd_fwd_hooks,
            ):
                lm_eval_results_harm_ablation = eval_harness(
                    cfg, model_base, "or_ablation_harm_actadd"
                )

            print("--------Finished LM eval harness evaluation. Starting translations.")
            print(datetime.now() - start)

            # delete model base to free up some memory before we go into translation.
            del model_base

            # Translate multijail and or bench multilingual answers before passing to the evaluation function
            translate_and_save_translated_responses_for_dataset(
                cfg, "multijail", "or_ablation_harm_actadd"
            )
            translate_and_save_translated_responses_for_dataset(
                cfg, "or_bench_multilingual", "or_ablation_harm_actadd"
            )

            print("--- Finished translations, starting evaluations.")
            print(datetime.now() - start)

            for dataset_name in cfg.jailbreak_evaluation_datasets:
                evaluate_completions_and_save_results_for_dataset(
                    cfg,
                    "or_ablation_harm_actadd",
                    dataset_name,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                )
            for dataset_name in cfg.over_refusal_evaluation_datasets:
                evaluate_completions_and_save_results_for_dataset(
                    cfg,
                    "or_ablation_harm_actadd",
                    dataset_name,
                    eval_methodologies=cfg.refusal_eval_methodologies,
                )

        elif cfg.mode == "harm_ablation":

            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                harm_ablation_fwd_pre_hooks,
                harm_ablation_fwd_hooks,
                "harm_ablation",
                "or_bench_multilingual",
                system=cfg.system,
            )

            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                harm_ablation_fwd_pre_hooks,
                harm_ablation_fwd_hooks,
                "harm_ablation",
                "multijail",
                system=cfg.system,
            )

            del model_base  # Free up some memory before going into translation

            # Translate multijail and or bench multilingual answers before passing to the evaluation function
            translate_and_save_translated_responses_for_dataset(
                cfg, "multijail", "harm_ablation"
            )
            translate_and_save_translated_responses_for_dataset(
                cfg, "or_bench_multilingual", "harm_ablation"
            )

            for dataset_name in cfg.jailbreak_evaluation_datasets:
                evaluate_completions_and_save_results_for_dataset(
                    cfg,
                    "harm_ablation",
                    dataset_name,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                )
            for dataset_name in cfg.over_refusal_evaluation_datasets:
                evaluate_completions_and_save_results_for_dataset(
                    cfg,
                    "harm_ablation",
                    dataset_name,
                    eval_methodologies=cfg.refusal_eval_methodologies,
                )
            with add_hooks(
                module_forward_pre_hooks=harm_ablation_fwd_pre_hooks,
                module_forward_hooks=harm_ablation_fwd_hooks,
            ):
                lm_eval_results_harm_ablation = eval_harness(
                    cfg, model_base, "harm_ablation"
                )

    else:
        print("---- Starting generations for baseline.")
        print(datetime.now() - start)
        with add_hooks(
            module_forward_pre_hooks=baseline_fwd_pre_hooks,
            module_forward_hooks=baseline_fwd_hooks,
        ):
            lm_eval_results_harm_ablation = eval_harness(cfg, model_base, "baseline")
        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            baseline_fwd_pre_hooks,
            baseline_fwd_hooks,
            "baseline",
            "or_bench_hard",
            dataset=or_bench_hard_data,
            system=cfg.system,
        )

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            baseline_fwd_pre_hooks,
            baseline_fwd_hooks,
            "baseline",
            "or_bench_multilingual",
            system=cfg.system,
        )

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            baseline_fwd_pre_hooks,
            baseline_fwd_hooks,
            "baseline",
            "multijail",
            system=cfg.system,
        )
        print("----Finished generations for baseline. Starting translations. ")
        print(datetime.now() - start)

        del model_base

        # Translate multijail and or bench multilingual answers before passing to the evaluation function
        translate_and_save_translated_responses_for_dataset(
            cfg, "multijail", "baseline"
        )
        translate_and_save_translated_responses_for_dataset(
            cfg, "or_bench_multilingual", "baseline"
        )
        print("-----Finished translations. Starting evaluations.")
        print(datetime.now() - start)

        for dataset_name in cfg.jailbreak_evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(
                cfg,
                "baseline",
                dataset_name,
                eval_methodologies=cfg.jailbreak_eval_methodologies,
            )
        for dataset_name in cfg.over_refusal_evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(
                cfg,
                "baseline",
                dataset_name,
                eval_methodologies=cfg.refusal_eval_methodologies,
            )
        print("---- Finished evaluations for baselinen.")
        print(datetime.now() - start)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        config_path=args.config_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
