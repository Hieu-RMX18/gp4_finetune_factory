PYTHON ?= python3
GP4_WS ?= /home/hieu2/gp4_ws
SEED ?= data/seed/*.jsonl
VALIDATED ?= data/validated/seed_validated.jsonl
GENERATED ?= data/generated/pilot_50.jsonl
PILOT_VALIDATED ?= data/validated/pilot_100_validated.jsonl
HELDOUT_MIN ?= 11
ADAPTER_DIR ?= models/qwen25_gp4_lora_pilot
BUNDLE ?= artifact_downloads/gp4_finetune_factory_retrain_bundle.zip

.PHONY: contract test validate-seed validate-generated validate-pilot dedupe-seed dedupe-pilot splits export-unsloth review review-pilot train-dry-run train infer-dry-run infer eval gates retrain-bundle pilot-data

contract:
	$(PYTHON) scripts/extract_repo_contract.py --repo $(GP4_WS) --output reports/repo_contract.json

test:
	$(PYTHON) -m pytest -q

validate-seed:
	$(PYTHON) scripts/validate_dataset.py --input $(SEED) --strict --contract-repo $(GP4_WS) --report reports/validation_seed.json

validate-generated:
	$(PYTHON) scripts/validate_dataset.py --input $(GENERATED) --strict --contract-repo $(GP4_WS) --report reports/validation_generated_pilot_50.json

validate-pilot:
	$(PYTHON) scripts/validate_dataset.py --input $(PILOT_VALIDATED) --strict --contract-repo $(GP4_WS) --report reports/validation_pilot_100.json

dedupe-seed:
	mkdir -p data/validated
	$(PYTHON) scripts/dedupe_dataset.py --input data/seed/gp4_seed_starter.jsonl --output $(VALIDATED)

dedupe-pilot:
	mkdir -p data/validated
	$(PYTHON) scripts/dedupe_dataset.py --input data/seed/gp4_seed_starter.jsonl $(GENERATED) --output $(PILOT_VALIDATED)

splits: dedupe-pilot
	$(PYTHON) scripts/build_splits.py --input $(PILOT_VALIDATED) --output-dir data/splits

export-unsloth: splits
	$(PYTHON) scripts/export_unsloth.py --input data/splits/train.jsonl --output data/splits/train_unsloth.jsonl
	$(PYTHON) scripts/export_unsloth.py --input data/splits/val.jsonl --output data/splits/val_unsloth.jsonl
	$(PYTHON) scripts/export_unsloth.py --input data/splits/test.jsonl --output data/splits/test_unsloth.jsonl

review:
	$(PYTHON) scripts/render_review_html.py --input data/seed/gp4_seed_starter.jsonl --output reports/review.html

review-pilot:
	$(PYTHON) scripts/render_review_html.py --input $(PILOT_VALIDATED) --output reports/review_pilot_100.html

train-dry-run:
	$(PYTHON) scripts/train_unsloth_qlora.py --train data/splits/train.jsonl --val data/splits/val.jsonl --output-dir $(ADAPTER_DIR) --dry-run

train:
	$(PYTHON) scripts/train_unsloth_qlora.py --train data/splits/train.jsonl --val data/splits/val.jsonl --output-dir $(ADAPTER_DIR)

infer-dry-run:
	$(PYTHON) scripts/run_adapter_inference.py --input data/splits/test.jsonl --adapter-dir $(ADAPTER_DIR) --dry-run

infer:
	$(PYTHON) scripts/run_adapter_inference.py --input data/splits/test.jsonl --adapter-dir $(ADAPTER_DIR) --output outputs/model_outputs.jsonl

eval:
	$(PYTHON) scripts/eval_model_outputs.py --input outputs/model_outputs.jsonl --contract-repo $(GP4_WS)

gates:
	$(PYTHON) scripts/check_acceptance_gates.py --eval-report reports/eval_report.json --min-rows $(HELDOUT_MIN)

retrain-bundle:
	$(PYTHON) scripts/build_retrain_bundle.py --output $(BUNDLE)

pilot-data: contract test validate-seed validate-generated dedupe-pilot validate-pilot splits export-unsloth review-pilot
