PYTHON ?= python3
GP4_WS ?=
GP4_WS_EXPECTED_COMMIT ?=
CLOUD_ROOT ?=
RUN_ID ?=
SEED ?= data/seed/*.jsonl
REPORT_DIR ?= $(CLOUD_ROOT)/reports
VALIDATED ?= $(CLOUD_ROOT)/data/validated/seed_validated.jsonl
GENERATED ?= $(CLOUD_ROOT)/data/generated/pilot_50.jsonl
PILOT_VALIDATED ?= $(CLOUD_ROOT)/data/validated/pilot_100_validated.jsonl
SPLIT_DIR ?= $(CLOUD_ROOT)/data/splits
OUTPUT_DIR ?= $(CLOUD_ROOT)/outputs
HELDOUT_MIN ?= 11
ADAPTER_DIR ?= $(CLOUD_ROOT)/models/qwen25_gp4_lora_pilot
BUNDLE ?= $(CLOUD_ROOT)/bundles/gp4_finetune_factory_source_bundle.zip
LOCAL_INSTALL_MANIFEST ?= $(CLOUD_ROOT)/reports/local_install_manifest_$(RUN_ID).json
LOCAL_TARGET_REPO ?= $(GP4_WS)

.PHONY: require-cloud-root require-gp4-ws require-run-id contract test validate-seed validate-generated validate-pilot dedupe-seed dedupe-pilot splits export-unsloth review review-pilot train-dry-run train infer-dry-run infer eval gates audit-cloud-run local-install-manifest open-colab-brave cloud-dry-run cloud-v2-dry-run retrain-bundle pilot-data

require-cloud-root:
	@test -n "$(CLOUD_ROOT)" || (echo "CLOUD_ROOT is required for artifact-producing targets" && exit 1)

require-gp4-ws:
	@test -n "$(GP4_WS)" || (echo "GP4_WS is required; set GP4_WS=/path/to/gp4_ws" && exit 1)

require-run-id:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 1)

contract: require-cloud-root require-gp4-ws
	$(PYTHON) scripts/extract_repo_contract.py --repo $(GP4_WS) --output "$(REPORT_DIR)/repo_contract.json" --cloud-root "$(CLOUD_ROOT)"

test:
	$(PYTHON) -m pytest -q

validate-seed: require-cloud-root require-gp4-ws
	$(PYTHON) scripts/validate_dataset.py --input $(SEED) --strict --contract-repo $(GP4_WS) --report "$(REPORT_DIR)/validation_seed.json" --cloud-root "$(CLOUD_ROOT)"

validate-generated: require-cloud-root require-gp4-ws
	$(PYTHON) scripts/validate_dataset.py --input "$(GENERATED)" --strict --contract-repo $(GP4_WS) --report "$(REPORT_DIR)/validation_generated_pilot_50.json" --cloud-root "$(CLOUD_ROOT)"

validate-pilot: require-cloud-root require-gp4-ws
	$(PYTHON) scripts/validate_dataset.py --input "$(PILOT_VALIDATED)" --strict --contract-repo $(GP4_WS) --report "$(REPORT_DIR)/validation_pilot_100.json" --cloud-root "$(CLOUD_ROOT)"

dedupe-seed: require-cloud-root
	mkdir -p "$(CLOUD_ROOT)/data/validated"
	$(PYTHON) scripts/dedupe_dataset.py --input data/seed/gp4_seed_starter.jsonl --output "$(VALIDATED)" --report "$(REPORT_DIR)/dedupe_seed.json" --cloud-root "$(CLOUD_ROOT)"

dedupe-pilot: require-cloud-root
	mkdir -p "$(CLOUD_ROOT)/data/validated"
	$(PYTHON) scripts/dedupe_dataset.py --input data/seed/gp4_seed_starter.jsonl "$(GENERATED)" --output "$(PILOT_VALIDATED)" --report "$(REPORT_DIR)/dedupe_pilot.json" --cloud-root "$(CLOUD_ROOT)"

splits: dedupe-pilot
	$(PYTHON) scripts/build_splits.py --input "$(PILOT_VALIDATED)" --output-dir "$(SPLIT_DIR)" --report "$(REPORT_DIR)/split_report.json" --cloud-root "$(CLOUD_ROOT)"

export-unsloth: splits
	$(PYTHON) scripts/export_unsloth.py --input "$(SPLIT_DIR)/train.jsonl" --output "$(SPLIT_DIR)/train_unsloth.jsonl" --cloud-root "$(CLOUD_ROOT)"
	$(PYTHON) scripts/export_unsloth.py --input "$(SPLIT_DIR)/val.jsonl" --output "$(SPLIT_DIR)/val_unsloth.jsonl" --cloud-root "$(CLOUD_ROOT)"
	$(PYTHON) scripts/export_unsloth.py --input "$(SPLIT_DIR)/test.jsonl" --output "$(SPLIT_DIR)/test_unsloth.jsonl" --cloud-root "$(CLOUD_ROOT)"

review: require-cloud-root
	$(PYTHON) scripts/render_review_html.py --input data/seed/gp4_seed_starter.jsonl --output "$(REPORT_DIR)/review.html" --cloud-root "$(CLOUD_ROOT)"

review-pilot: require-cloud-root
	$(PYTHON) scripts/render_review_html.py --input "$(PILOT_VALIDATED)" --output "$(REPORT_DIR)/review_pilot_100.html" --cloud-root "$(CLOUD_ROOT)"

train-dry-run: require-cloud-root
	$(PYTHON) scripts/train_unsloth_qlora.py --train "$(SPLIT_DIR)/train.jsonl" --val "$(SPLIT_DIR)/val.jsonl" --output-dir "$(ADAPTER_DIR)" --report "$(REPORT_DIR)/training_report.json" --cloud-root "$(CLOUD_ROOT)" --dry-run

train: require-cloud-root
	$(PYTHON) scripts/train_unsloth_qlora.py --train "$(SPLIT_DIR)/train.jsonl" --val "$(SPLIT_DIR)/val.jsonl" --output-dir "$(ADAPTER_DIR)" --report "$(REPORT_DIR)/training_report.json" --cloud-root "$(CLOUD_ROOT)"

infer-dry-run: require-cloud-root
	$(PYTHON) scripts/run_adapter_inference.py --input "$(SPLIT_DIR)/test.jsonl" --adapter-dir "$(ADAPTER_DIR)" --output "$(OUTPUT_DIR)/model_outputs.jsonl" --report "$(REPORT_DIR)/inference_report.json" --cloud-root "$(CLOUD_ROOT)" --dry-run

infer: require-cloud-root
	$(PYTHON) scripts/run_adapter_inference.py --input "$(SPLIT_DIR)/test.jsonl" --adapter-dir "$(ADAPTER_DIR)" --output "$(OUTPUT_DIR)/model_outputs.jsonl" --report "$(REPORT_DIR)/inference_report.json" --cloud-root "$(CLOUD_ROOT)"

eval: require-cloud-root require-gp4-ws
	$(PYTHON) scripts/eval_model_outputs.py --input "$(OUTPUT_DIR)/model_outputs.jsonl" --contract-repo $(GP4_WS) --report "$(REPORT_DIR)/eval_report.json" --cloud-root "$(CLOUD_ROOT)"

gates: require-cloud-root
	$(PYTHON) scripts/check_acceptance_gates.py --eval-report "$(REPORT_DIR)/eval_report.json" --min-rows $(HELDOUT_MIN) --report "$(REPORT_DIR)/acceptance_gate_report.json" --cloud-root "$(CLOUD_ROOT)"

audit-cloud-run: require-cloud-root require-run-id
	$(PYTHON) scripts/audit_cloud_completion.py --cloud-root "$(CLOUD_ROOT)" --run-id "$(RUN_ID)" --report "$(REPORT_DIR)/completion_audit_$(RUN_ID).json"

local-install-manifest: require-cloud-root require-run-id require-gp4-ws
	$(PYTHON) scripts/build_local_install_manifest.py --adapter-dir "$(ADAPTER_DIR)" --acceptance-report "$(REPORT_DIR)/acceptance_gate_report_$(RUN_ID).json" --target-repo "$(LOCAL_TARGET_REPO)" --expected-commit "$(GP4_WS_EXPECTED_COMMIT)" --cloud-root "$(CLOUD_ROOT)" --output "$(LOCAL_INSTALL_MANIFEST)"

open-colab-brave:
	$(PYTHON) scripts/launch_cloud_runtime.py

cloud-dry-run:
	$(PYTHON) scripts/cloud_orchestrator.py --run-id dryrun --cloud-root /tmp/gp4_finetune_factory_cloud_test --seed data/seed/gp4_seed_starter.jsonl --phases cloud-setup,provider-probe,contract,seed-check,generate-smoke,generate-1k,quality-gate-1k,generate-30k,quality-gate-30k,generate-50k,quality-gate-50k,dedupe,split,train,infer,eval,package --dry-run --allow-tmp

cloud-v2-dry-run:
	$(PYTHON) scripts/cloud_orchestrator.py --run-id dryrun-v2 --cloud-root /tmp/gp4_finetune_factory_cloud_test_v2 --seed data/seed/gp4_seed_starter.jsonl --phases ignored --preset v2-300k --source-plan docs/superpowers/plans/2026-05-20-gp4-v2-300k-readiness.md --dry-run --allow-tmp

validate-react-ir:
	$(PYTHON) scripts/validate_react_ir_dataset.py --input data/seed/*.jsonl --strict --report /tmp/gp4_finetune_factory_validation_report.json --allow-tmp

retrain-bundle: require-cloud-root
	$(PYTHON) scripts/build_retrain_bundle.py --cloud-root "$(CLOUD_ROOT)" --output "$(BUNDLE)"

pilot-data: contract test validate-seed validate-generated dedupe-pilot validate-pilot splits export-unsloth review-pilot
