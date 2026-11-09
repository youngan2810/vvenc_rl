"""Verify the executed source, not only tensor architecture compatibility."""
import subprocess


def verify_source(repo, experiment):
    commit = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if commit != experiment["source_commit"]:
        raise ValueError("Collector/Validation source differs from checkpoint source_commit")
    dirty = subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True).strip()
    if dirty:
        raise ValueError("Collector/Validation requires a clean committed source checkout")

