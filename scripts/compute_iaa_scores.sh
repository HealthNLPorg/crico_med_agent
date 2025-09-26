#!/bin/sh
printf "IAA batch2: Dave as predicted, YY as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave/IAA_batch2 \
   -r iaa_sanitized/yang/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_predicted_yy_reference.txt
printf "IAA batch2: Dave as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave/IAA_batch2 \
   -p iaa_sanitized/yang/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_reference_yy_predicted.txt

printf "IAA batch2: Dave adjudication as predicted, YY as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave_adjudication/IAA_batch2 \
   -r iaa_sanitized/yang/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_adjudication_predicted_yy_reference.txt
printf "IAA batch2: Dave adjudication as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave_adjudication/IAA_batch2 \
   -p iaa_sanitized/yang/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_adjudication_reference_yy_predicted.txt

printf "IAA batch2: Dave adjudication as predicted, Dave as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave_adjudication/IAA_batch2 \
   -r iaa_sanitized/dave/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_adjudication_predicted_dave_reference.txt
printf "IAA batch2: Dave adjudication as reference, Dave as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave_adjudication/IAA_batch2 \
   -p iaa_sanitized/dave/IAA_batch2 \
   --overlap > iaa_results/iaa_batch2/full_dave_adjudication_reference_dave_predicted.txt

printf "IAA batch3: Dave as predicted, YY as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave/IAA_batch3 \
   -r iaa_sanitized/yang/IAA_batch3 \
   --overlap > iaa_results/iaa_batch3/full_dave_predicted_yy_reference.txt

printf "IAA batch3: Dave as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave/IAA_batch3 \
   -p iaa_sanitized/yang/IAA_batch3 \
   --overlap > iaa_results/iaa_batch3/full_dave_reference_yy_predicted.txt

printf "batch2_deduplicated: Dave as predicted, YY as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave/batch2_deduplicated \
   -r iaa_sanitized/yang/batch2_deduplicated \
   --overlap > iaa_results/batch2_deduplicated/full_dave_predicted_yy_reference.txt
printf "batch2_deduplicated: Dave as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave/batch2_deduplicated \
   -p iaa_sanitized/yang/batch2_deduplicated \
   --overlap > iaa_results/batch2_deduplicated/full_dave_reference_yy_predicted.txt

printf "batch2_deduplicated_reformat: Dave as predicted, YY as reference\n"
uv run python -m anafora.evaluate \
   -p iaa_sanitized/dave/batch2_deduplicated_reformat \
   -r iaa_sanitized/yang/batch2_deduplicated_reformat \
   --overlap > iaa_results/batch2_deduplicated_reformat/full_dave_predicted_yy_reference.txt
printf "batch2_deduplicated_reformat: Dave as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave/batch2_deduplicated_reformat \
   -p iaa_sanitized/yang/batch2_deduplicated_reformat \
   --overlap > iaa_results/batch2_deduplicated_reformat/full_dave_reference_yy_predicted.txt

printf "batch3: Dave as reference, YY as predicted\n"
uv run python -m anafora.evaluate \
   -r iaa_sanitized/dave/batch3 \
   -p iaa_sanitized/yang/batch3 \
   --overlap > iaa_results/batch3/full_dave_reference_yy_predicted.txt
