#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ProviderProbeResult:
    provider: str
    available: bool
    cloud_storage_ready: bool
    free_tier: bool
    paid_risk: bool
    blocked_reason: str

    @property
    def is_usable(self) -> bool:
        return (
            self.available
            and self.cloud_storage_ready
            and self.free_tier
            and not self.paid_risk
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
    cloud_ready = root.exists()
    blocked_reason = ""
    if paid_risk:
        blocked_reason = "paid provider configuration is forbidden"
    elif not cloud_ready:
        blocked_reason = f"cloud storage root is not mounted: {cloud_root}"

    return ProviderProbeResult(
        provider=detected_provider,
        available=detected_provider != "local",
        cloud_storage_ready=cloud_ready,
        free_tier=env.get("FREE_TIER_CONFIRMED", "1") != "0",
        paid_risk=paid_risk,
        blocked_reason=blocked_reason,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe free/trial cloud provider readiness.")
    parser.add_argument("--cloud-root", default="")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = probe_from_environment(
        os.environ,
        args.cloud_root,
        provider=args.provider,
    )
    payload = write_platform_status(args.report, result)
    print(
        f"provider={payload['provider']} passed={payload['passed']} "
        f"blocked_reason={payload['blocked_reason']} report={args.report}"
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
