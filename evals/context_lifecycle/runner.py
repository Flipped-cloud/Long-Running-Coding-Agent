from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from longrun_agent.context_probes.runner import run_probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--probe", default="position")
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-provider-script", type=Path, default=None)
    args = parser.parse_args()
    result = run_probe(
        config_path=args.config,
        probe=args.probe,
        lengths=[int(item) for item in args.lengths.split(",") if item],
        samples=args.samples,
        seed=args.seed,
        modes=[item for item in args.modes.split(",") if item] if args.modes else None,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        fake_provider_script=args.fake_provider_script,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
