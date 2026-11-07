"""Run the standard-library test suite from any working directory."""

import argparse
import os
import sys
import unittest

from _bootstrap import ROOT as ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true", help="Include real model parity checks")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RUN_MODEL_TESTS"] = "1" if args.models else "0"
    sys.path.insert(0, str(ROOT / "scripts"))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    return not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(main())
