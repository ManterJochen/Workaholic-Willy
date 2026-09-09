# Which objects a corpus is made of, from the command line.
#
# The Python twin of this file is scripts/examples/api/06_datagen/bring_your_own_parts.py and it
# reads the same library and runs the same licence gate. `--fetch` with `--from` imports a directory
# you already have; the collections are downloaded by naming them with `--source` instead.

# 1. What this machine holds, per source, and whether it is enough to render from.
python -m datagen.assets --check

# 2. The attribution text a CC-BY collection obliges you to ship with any dataset built from it.
python -m datagen.assets --attribution

# 3. The licence gate on the manifest the shipped config draws. It runs before a render, not after.
python -m datagen audit

# 4. Your own parts. `--license` is required for `custom` and refused rather than defaulted.
#    python -m datagen.assets --fetch --from ./my_cad --source custom `
#        --license own --attribution-text "ACME GmbH"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
