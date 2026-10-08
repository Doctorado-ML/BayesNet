# ***************************************************************
# SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: MIT
# ***************************************************************

import sys

readme_file = "README.md"
print("Updating coverage...")
# Generate badge line
coverage_file = sys.argv[1] + "/coverage.info"
lines_found = lines_hit = 0
with open(coverage_file, "r") as coverage:
    for line in coverage:
        if line.startswith("LF:"):
            lines_found += int(line[3:])
        elif line.startswith("LH:"):
            lines_hit += int(line[3:])
percentage = round(100 * lines_hit / lines_found, 1) if lines_found else 0
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
