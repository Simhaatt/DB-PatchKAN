# Zenodo publication checklist

Software authors/order and MIT license are confirmed. Affiliations are transcribed from the supplied manuscript title image. No ORCIDs or contact emails were supplied. `.zenodo.json` and `CITATION.cff` contain consistent full display names; `.zenodo.json` is authoritative for Zenodo if both are present.

1. Sign into production Zenodo and link GitHub.
2. Open profile → GitHub → Sync now → enable `Simhaatt/DB-PatchKAN`.
3. Review this snapshot, outstanding limitations and actual CI results. Do not create a final manuscript release merely because files have been pushed.
4. Choose the verified commit and final tag; publish the GitHub release after integration is enabled. No release/tag has been created by this preparation.
5. Wait for Zenodo ingestion; check the author order, title, MIT license and software version before citing its DOI.
6. Deposit the approved large artifact archive separately as a new record. Include checkpoints, immutable input/split manifests, runtime versions, full predictions and checksums, with permission-appropriate access/license. Do not label raw third-party data as MIT.
7. Cross-link code/artifact records using real DOI identifiers. Keep separate BMS/capacity/charging definitions and scope notes in the artifact README.
8. Add the verified DOI(s) and version-specific manuscript citation. Never use the paper DOI for a different software/dataset object.

Official guides: https://help.zenodo.org/docs/github/enable-repository/ and https://help.zenodo.org/docs/github/archive-software/github-upload/ and https://help.zenodo.org/docs/deposit/create-new-upload/ .

Source/raw-data/checkpoint archives remain in the research workspace and are not automatically included in a GitHub-derived source snapshot. The repository is usable for code/figure inspection and the public-data fleet route, but it is not yet a deposited full artifact package.
