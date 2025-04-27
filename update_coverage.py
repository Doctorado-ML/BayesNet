# ***************************************************************
# SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT
# ***************************************************************

import subprocess
import sys

readme_file = "README.md"
print("Updating coverage...")
# Generate badge line
output = subprocess.check_output(
    "lcov --summary " + sys.argv[1] + "/coverage.info",
    shell=True,
)
value = output.decode("utf-8").strip()
percentage = 0
for line in value.splitlines():
    if "lines" in line:
        percentage = float(line.split(":")[1].split("%")[0])
        break
print(f"Coverage: {percentage}%")
if percentage < 90:
    print("⛔Coverage is less than 90%. I won't update the badge.")
    sys.exit(1)
percentage_label = str(percentage).replace('.', ',')
coverage_line = f"[![Coverage Badge](https://img.shields.io/badge/Coverage-{percentage_label}%25-green)](https://gitea.rmontanana.es/rmontanana/BayesNet)"
# Update README.md
with open(readme_file, "r") as f:
    lines = f.readlines()
with open(readme_file, "w") as f:
    for line in lines:
        if "img.shields.io/badge/Coverage" in line:
            f.write(coverage_line + "\n")
        else:
            f.write(line)
print(f"✅Coverage updated with value: {percentage}")
