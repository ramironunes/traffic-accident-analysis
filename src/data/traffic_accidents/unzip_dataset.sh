#!/bin/bash
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-07-08 07:52:06
# @Info:   A brief description of the file
# ============================================================================

for dir in */; do
    (cd "$dir" && for file in *.zip; do unzip "$file"; done)
done
