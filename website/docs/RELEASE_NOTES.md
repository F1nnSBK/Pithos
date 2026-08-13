---
id: release-notes
title: Release Notes
sidebar_label: Release Notes
---

# Release Notes

## v1.0.1 (Patch)

**Author**: Finn

### Bug Fixes

**Gate 2 QEG Planetary Grid Filter**
- **The Issue**: A flaw in the 3-way gate logic of the Planetary Grid (`executeVotingRange`). A hardcoded filter was unintentionally discarding 50% of the entire vector space during the search phase because it unconditionally checked if the MSB of the dataset record was `0`, without correlating it with the query vector. 
- **The Fix**: Removed the unconditional MSB discard logic in both 1-bit and 2-bit quantization modes.
- **Impact**: Recall in `queryPlanetaryGrid` searches has increased dramatically (candidates evaluated roughly doubled in uniform random benchmarks).

> [!TIP]
> **Backward Compatibility**: 100%
> This fix only modifies the runtime query evaluation (`FlatIndex.java`). The binary format of the index on disk is completely untouched. You can seamlessly query any index compiled with Pithos `v1.0` using this new version. No re-indexing is required.
