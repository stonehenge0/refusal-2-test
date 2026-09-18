"""Translate output prompts to English."""

import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers import logging
from huggingface_hub import scan_cache_dir

sep = "-"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

logging.set_verbosity_error()


GROUP2LANG = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}

ISO_TO_NAME = {
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
}

LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}


def clean_hf_cache():
    """Clean the latest used model from Huggingface cache to avoid cache becoming too large over time."""
    try:
        cache_info = scan_cache_dir()
        model_repos = [repo for repo in cache_info.repos if repo.repo_type == "model"]

        if not model_repos:
            return

        most_recent = max(model_repos, key=lambda x: getattr(x, "last_accessed", 0))

        if not hasattr(most_recent, "revisions") or not most_recent.revisions:
            return

        revision_hashes = [
            rev.commit_hash
            for rev in most_recent.revisions
            if hasattr(rev, "commit_hash")
        ]

        if not revision_hashes:
            return

        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        delete_strategy.execute()
    except Exception:
        pass


def detect_english(texts, batch_size=32):
    """Detect if texts are in English using XLM-RoBERTa (language detection).

    Parameters
    ----------
    texts : list of str
        Input texts to check for English language.
    batch_size : int, optional
        Number of texts to process per batch.

    Returns
    -------
    list of bool
        True if text is detected as English, False otherwise.
    """
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    lang_detector = pipeline("text-classification", model=model_ckpt)

    response_is_en = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = lang_detector(batch, top_k=1, truncation=True)

        batch_response_is_en = [result[0]["label"] == "en" for result in results]
        response_is_en.extend(batch_response_is_en)

    print(f"Detected {sum(response_is_en)}/{len(response_is_en)} texts as English")
    return response_is_en


def prepare_dataframe(df):
    """Prepare dataframe for translation by adding English prompts and detecting response language.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with multilingual prompts and responses.

    Returns
    -------
    pd.DataFrame
        Prepared dataframe with prompt_english, response_is_en, and response_translated columns.
    """

    english_prompts = df[df["language"] == "English"][["prompt_id", "prompt"]].rename(
        columns={"prompt": "prompt_english"}
    )
    df = df.merge(english_prompts, on="prompt_id", how="left")

    # Models often respond in English for non-English prompts. Here we want to catch those so we don't translate double.
    df["response_is_en"] = detect_english(df["response"].tolist())

    df["response_translated"] = None

    df.loc[df["response_is_en"], "response_translated"] = df.loc[
        df["response_is_en"], "response"
    ]

    print(f"Found {df['response_is_en'].sum()} responses already in English")
    print(f"Need to translate {(~df['response_is_en']).sum()} responses\n")

    return df


def translate_batch(
    texts,
    model,
    tokenizer,
    source_lang,
    target_lang,
    batch_size=8,
    use_chat_template=False,
):
    """Translate a list of texts in batches.

    Parameters
    ----------
    texts : list of str
        Input texts to be translated.
    model :
        A causal language model used for text generation.
    tokenizer :
        Tokenizer corresponding to the model.
    source_lang : str
        Source language name (e.g., "English").
    target_lang : str
        Target language name (e.g., "Bengali").
    batch_size : int, optional
        Number of texts to translate per batch.
    use_chat_template : bool, optional
        Whether to use chat template formatting.

    Returns
    -------
    list of str
        Translated texts in the same order as the input list."""

    outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Gemma and X-Alma require different prompt formatting with regards to chat templates.
        if use_chat_template:
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f"Translate this from {source_lang} to {target_lang}.\n{source_lang}: {t}\n{target_lang}:",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in batch
            ]
        else:
            prompts = [
                f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {t}\n{target_lang}:"
                for t in batch
            ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=1200)

        input_lengths = inputs["input_ids"].shape[1]
        generated_only = generated[:, input_lengths:]

        decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        outputs.extend(decoded)

    return outputs


def translate_gemmax2(cfg, df):
    "Translate OR Bench to Bengali."

    model_id_gemma = "ModelSpace/GemmaX2-28-9B-v0.1"
    tokenizer_gemma = AutoTokenizer.from_pretrained(model_id_gemma)
    tokenizer_gemma.padding_side = "left"
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_id_gemma,
        device_map="auto",
        quantization_config=quantization_config,
    )

    bengali_mask = (df["language"] == "Bengali") & ~df["response_is_en"]
    bengali_texts = df.loc[bengali_mask, "response"].tolist()

    bengali_translations = translate_batch(
        texts=bengali_texts,
        model=model_gemma,
        tokenizer=tokenizer_gemma,
        batch_size=cfg.batch_size,
        source_lang="Bengali",
        target_lang="English",
    )

    df.loc[bengali_mask, "response_translated"] = bengali_translations

    del model_gemma
    del tokenizer_gemma
    torch.cuda.empty_cache()

    return df


def translate_xalma(cfg, df):
    "Translate to English from Chinese, Italian, Vietnamese, Arabic, Korean, Thai with x-Alma."

    """
    Translation that loads each model (per language family) only once and processes all prompts
    for that language before moving to the next language.

    Args:
        sampled_df (pd.DataFrame): Input DataFrame with prompts.
        target_langs (List[str]): List of ISO codes to translate into.

    Returns:
        pd.DataFrame: DataFrame with translated samples, ids, and language codes.
    """

    source_lang_ids = ["it", "th", "vi", "zh", "ja", "ko", "ar"]

    non_english_mask = ~df["response_is_en"]

    for lang_id in source_lang_ids:
        print(f"Starting X-Alma translation for {lang_id}")
        print()

        language_name_full = ISO_TO_NAME[lang_id]

        lang_mask = (df["language"] == language_name_full) & non_english_mask

        lang_texts = df.loc[lang_mask, "response"].tolist()

        group_id = LANG2GROUP[lang_id]
        model_xalma = AutoModelForCausalLM.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group{group_id}",
            device_map="auto",
            quantization_config=quantization_config,
        )

        tokenizer_xalma = AutoTokenizer.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side="left"
        )
        tokenizer_xalma.padding_side = "left"

        translations_xalma = translate_batch(
            lang_texts,
            model=model_xalma,
            tokenizer=tokenizer_xalma,
            source_lang=language_name_full,
            batch_size=cfg.batch_size,
            target_lang="English",
            use_chat_template=True,
        )

        df.loc[lang_mask, "response_translated"] = translations_xalma

        torch.cuda.empty_cache()

    return df


def translate(cfg, df):
    """Translate multilingual responses to English.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with multilingual prompts and responses.
        Must have columns: prompt_id, language, prompt, response

    Returns
    -------
    pd.DataFrame
        Dataframe with added columns: prompt_english, response_is_en, response_translated
    """
    # 1. Prepare dataframe (add English prompts, detect language, initialize columns)
    df = prepare_dataframe(df)

    # 2. Translate from Bengali (XGemma)
    df = translate_gemmax2(cfg, df)

    # 3. Translate from all other languages (X-Alma)
    df = translate_xalma(cfg, df)

    # change column names so the right prompts get passed to the evaluation function
    df = df.rename(
        columns={
            "response": "response_original",
            "prompt": "prompt_original",
            "prompt_english": "prompt",
            "category": "real_category",
            "language": "category",
            "response_translated": "response",
        }
    )

    return df
