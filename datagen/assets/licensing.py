"""What may enter a dataset, and what may never: the rules, owned by the code that consumes them.

A rendered image is a derivative work of every mesh in it. A dataset is therefore not licence-clean
because the generator is; it is licence-clean because every asset that reached a pixel was. So the
licence travels with the asset row, through the render, into the dataset's attribution file.

Three rules, all fail-closed:

* Never non-commercial. Not for code, not for meshes, not for anything that ends up in a pixel.
* Never a forbidden project: research-only grasp datasets whose licences do not survive commercial
  use. `FORBIDDEN` is that list, and the repo-wide guard imports its vocabulary from here so the two
  can never disagree.
* CC0, CC-BY, or ours. Anything else is refused by name rather than waved through, because "probably
  fine" is how a dataset acquires a licence nobody can reconstruct two years later.

CC-BY is allowed and obligating: it requires attribution, so any asset carrying it must also carry
an ``attribution`` string, and :func:`audit_asset_rows` refuses the row without one. A CC-BY mesh
with no author recorded is a licence violation waiting for the first person who shares the dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, LICENSING_LOG_FILE

__all__ = [
    "ALLOWED_ASSET_LICENSES",
    "ATTRIBUTION_REQUIRED_PREFIXES",
    "FORBIDDEN",
    "audit_asset_rows",
    "forbidden_hits",
    "is_noncommercial", "is_noderivatives", "is_sharealike",
    "normalise_license",
]

#: The gate keeps its own file. A licence refusal is the one event here with a legal reason to be
#: findable months later, and it must not be buried under a night of render progress.
logger = create_logger("datagen.licensing", LICENSING_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Projects whose code or datasets are non-commercial, research-only, or otherwise not a boundary
#: this project can rely on. Case-insensitive substring match, so "GraspNet-1Billion", "graspnetAPI"
#: and "contact_graspnet" all hit. Adding a name here is a licensing decision: record why beside it.
#:
#: Owned by production code, because production code is what has to obey it. The repo-wide scanner
#: imports this tuple, so a name can never be forbidden in one place and allowed in the other.
FORBIDDEN: tuple[str, ...] = (
    # --- mirrored from the Aletheia data engine, deliberately identical ---
    "graspnet",          # GraspNet-1Billion: CC BY-NC-SA. The root licence most of this family inherits.
    "gsnet",             # the GraspNet-1Billion network
    "graspness",         # trained on GraspNet-1Billion
    "anygrasp",          # licence-key gated, non-commercial research licence
    "contact-graspnet",  # NVIDIA, non-commercial research licence
    "shapenet",          # research-only, per-model terms
    "acronym",           # NVIDIA ACRONYM: CC BY-NC-SA
    # --- added for this project ---
    "suctionnet",        # built on GraspNet-1Billion
    "dex-net",           # research licence, non-commercial terms
    "dexnet",
)

#: Asset licences that may enter a dataset. ``own`` means procedurally generated here.
ALLOWED_ASSET_LICENSES: tuple[str, ...] = ("cc0", "cc-by", "own")

#: Licences that legally require naming the author wherever the work, or a derivative such as any
#: render, appears. A row carrying one of these without an ``attribution`` string is refused.
ATTRIBUTION_REQUIRED_PREFIXES: tuple[str, ...] = ("cc-by",)


def forbidden_hits(text: str) -> list[str]:
    """Which forbidden project names appear in ``text`` (case-insensitive substring)."""
    lowered = text.lower()
    return [name for name in FORBIDDEN if name in lowered]


def normalise_license(license_id: str) -> str:
    """Lowercase, with every run of non-alphanumerics collapsed to one hyphen.

    Without this the punctuation does the gating: `startswith(("cc0", "cc-by", "own"))` reads
    `"CC BY-ND 4.0"` as forbidden and `"cc-by-nd-4.0"` as permitted, which are the same licence
    spelled two ways. Normalising first makes the answer a property of the licence rather than of
    whoever typed it, which is the precondition for the two rules below meaning anything.
    """
    out = "".join(c if c.isalnum() else "-" for c in license_id.strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def is_sharealike(license_id: str) -> bool:
    """True for a copyleft ShareAlike clause. A rendered corpus is a derivative work.

    A prefix test is not enough: `cc-by-sa-4.0` starts with `cc-by`, so the clause has to be read
    out of the normalised identifier rather than inferred from its opening.
    """
    text = normalise_license(license_id)
    return "sa" in text.split("-") or "sharealike" in text.replace("-", "")


def is_noderivatives(license_id: str) -> bool:
    """True for a NoDerivatives clause, which no amount of attribution can satisfy.

    ND forbids distributing a derivative at all, and every rendered scene is one. `cc-by-nd-4.0`
    starts with `cc-by`, so a prefix test admits it; the space-separated spelling `CC BY-ND 4.0`
    happens to be refused by such a test, which is an accident of string formatting rather than the
    rule working.
    """
    text = normalise_license(license_id)
    return "nd" in text.split("-") or "noderiv" in text.replace("-", "")


def is_noncommercial(license_id: str) -> bool:
    """True for a licence identifier carrying a non-commercial clause (``CC BY-NC-SA``, ``...-NC-...``)."""
    normalised = license_id.lower().replace(" ", "-").replace("_", "-")
    return (
        "nc" in normalised.split("-")
        or "noncommercial" in normalised
        or "non-commercial" in license_id.lower()
    )


def audit_asset_rows(rows: Iterable[Mapping[str, str]]) -> list[str]:
    """Every problem with these asset rows, as human-readable strings. An empty list means clean.

    Returns every problem rather than raising on the first: a manifest is audited before a long
    generation run, and one bad row at a time turns a five-minute fix into five runs.
    """
    problems: list[str] = []
    counted = 0
    for row in rows:
        counted += 1
        asset_id = row.get("id", "?")
        license_id = str(row.get("license", "")).strip()
        source = str(row.get("source", ""))
        for hit in forbidden_hits(source) + forbidden_hits(str(asset_id)):
            problems.append(f"asset {asset_id!r}: forbidden source (matches {hit!r})")
        if not license_id:
            problems.append(f"asset {asset_id!r}: missing license")
            continue
        if is_noncommercial(license_id):
            problems.append(f"asset {asset_id!r}: non-commercial license {license_id!r}")
            continue
        if is_sharealike(license_id):
            problems.append(
                f"asset {asset_id!r}: share-alike license {license_id!r} is copyleft, and a rendered "
                f"corpus is a derivative work that would have to ship under the same terms"
            )
            continue
        if is_noderivatives(license_id):
            problems.append(
                f"asset {asset_id!r}: no-derivatives license {license_id!r} forbids distributing a "
                f"derivative, and every rendered image of this mesh is one"
            )
            continue
        if not normalise_license(license_id).startswith(ALLOWED_ASSET_LICENSES):
            problems.append(f"asset {asset_id!r}: license {license_id!r} is not CC0 / CC-BY / own")
            continue
        if license_id.lower().startswith(ATTRIBUTION_REQUIRED_PREFIXES) and not row.get("attribution"):
            problems.append(
                f"asset {asset_id!r}: license {license_id!r} requires attribution, and none is recorded; "
                f"every rendered image is a derivative work of this mesh"
            )
    # Once per audit, never per row: the count is the result, and the reasons are the evidence for
    # it. Logged at error rather than raised, because returning the refusal and letting the caller
    # decide is what this function does; without this line a refused audit leaves no trace here.
    if problems:
        logger.error("licence audit REFUSED %d row(s): %s", len(problems), "; ".join(problems[:10]))
    else:
        logger.info("licence audit clean: %d row(s), all CC0 / CC-BY / own", counted)
    return problems
