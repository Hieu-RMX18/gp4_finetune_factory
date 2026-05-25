#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from cloud_runtime import CloudPathError, validate_cloud_output_path


@dataclass(frozen=True)
class ProviderProbeResult:
    provider: str
    available: bool
    cloud_storage_ready: bool
    free_tier: bool
    paid_risk: bool
    blocked_reason: str
    account_creation_automation: bool = False
    quota_bypass_attempt: bool = False
    idle_bypass_attempt: bool = False
    gpu_required: bool = False
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_probe_error: str = ""

    @property
    def is_usable(self) -> bool:
        return (
            self.available
            and self.cloud_storage_ready
            and self.free_tier
            and not self.paid_risk
            and not self.account_creation_automation
            and not self.quota_bypass_attempt
            and not self.idle_bypass_attempt
            and (not self.gpu_required or self.gpu_available)
            and not self.blocked_reason
        )


def choose_provider(results: list[ProviderProbeResult]) -> ProviderProbeResult:
    for result in results:
        if result.is_usable:
            return result
    if not results:
        return ProviderProbeResult(
            provider="unknown",
            available=False,
            cloud_storage_ready=False,
            free_tier=False,
            paid_risk=False,
            blocked_reason="no providers probed",
        )
    return results[0]


def probe_from_environment(
    env: Mapping[str, str],
    cloud_root: str,
    *,
    provider: str | None = None,
    require_gpu: bool = False,
) -> ProviderProbeResult:
    detected_provider = provider or _detect_provider(env)
    if not cloud_root:
        return ProviderProbeResult(
            detected_provider,
            available=False,
            cloud_storage_ready=False,
            free_tier=True,
            paid_risk=False,
            blocked_reason="cloud storage root is not configured",
        )

    root = Path(cloud_root).expanduser()
    paid_risk = env.get("PAID_PROVIDER_CONFIRMED") == "1"
    account_creation_automation = env.get("ACCOUNT_CREATION_AUTOMATION") == "1"
    quota_bypass_attempt = env.get("QUOTA_BYPASS_ATTEMPT") == "1"
    idle_bypass_attempt = env.get("IDLE_BYPASS_ATTEMPT") == "1"
    cloud_ready = root.exists()
    gpu_available, gpu_name, gpu_probe_error = _probe_nvidia_gpu()
    blocked_reason = ""
    if paid_risk:
        blocked_reason = "paid provider configuration is forbidden"
    elif not cloud_ready:
        blocked_reason = f"cloud storage root is not mounted: {cloud_root}"
    elif account_creation_automation:
        blocked_reason = "account creation automation is forbidden"
    elif quota_bypass_attempt:
        blocked_reason = "quota bypass attempts are forbidden"
    elif idle_bypass_attempt:
        blocked_reason = "idle bypass attempts are forbidden"
    elif require_gpu and not gpu_available:
        blocked_reason = (
            "CUDA GPU is required for Qwen2.5-7B QLoRA training and inference"
        )

    return ProviderProbeResult(
        provider=detected_provider,
        available=detected_provider != "local",
        cloud_storage_ready=cloud_ready,
        free_tier=env.get("FREE_TIER_CONFIRMED", "1") != "0",
        paid_risk=paid_risk,
        blocked_reason=blocked_reason,
        account_creation_automation=account_creation_automation,
        quota_bypass_attempt=quota_bypass_attempt,
        idle_bypass_attempt=idle_bypass_attempt,
        gpu_required=require_gpu,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_probe_error=gpu_probe_error,
    )


def write_platform_status(path: Path, result: ProviderProbeResult) -> dict:
    payload = {
        **asdict(result),
        "is_usable": result.is_usable,
        "passed": result.is_usable,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _detect_provider(env: Mapping[str, str]) -> str:
    if env.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if env.get("LIGHTNING_CLOUD_URL"):
        return "lightning"
    if env.get("COLAB_RELEASE_TAG") or env.get("HOME") == "/root":
        return "colab"
    if str(Path.cwd()).startswith("/content"):
        return "colab"
    return "local"

def _probe_nvidia_gpu() -> tuple[bool, str, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "", "nvidia-smi is not available"
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False, "", (result.stderr or result.stdout).strip()[-500:]
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpu_names:
        return False, "", "nvidia-smi returned no GPU names"
    return True, gpu_names[0], ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe free/trial cloud provider readiness.")
    parser.add_argument("--cloud-root", default="")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()

    try:
        validate_cloud_output_path(
            args.report,
            args.cloud_root,
            allow_tmp=args.allow_tmp,
        )
    except CloudPathError as exc:
        print(str(exc))
        return 1

    result = probe_from_environment(
        os.environ,
        args.cloud_root,
        provider=args.provider,
        require_gpu=args.require_gpu,
    )
    payload = write_platform_status(args.report, result)
    print(
        f"provider={payload['provider']} passed={payload['passed']} "
        f"blocked_reason={payload['blocked_reason']} report={args.report}"
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
