<!-- markdownlint-disable MD033 -->
# <img src="https://github.com/user-attachments/assets/ba54b620-961e-4d51-9720-f3dcb9b4014f" alt="DecentralizedAI Logo" width="85" height="80"> Dolphi dML - A Decentralized Machine Learning PoC
<!-- markdownlint-enable MD033 -->

![Super-Linter](https://github.com/EASS-HIT-PART-B/dML/actions/workflows/super-linter.yml/badge.svg)
![Build Status](https://github.com/EASS-HIT-PART-B/dML/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/EASS-HIT-PART-B/dML/badge.svg?branch=main)](https://coveralls.io/github/EASS-HIT-PART-B/dML?branch=main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/EASS-HIT-PART-B/dML/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/EASS-HIT-PART-B/dML.svg)](https://github.com/EASS-HIT-PART-B/dML/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/EASS-HIT-PART-B/dML.svg)](https://github.com/EASS-HIT-PART-B/dML/commits/main)


[Project Website](https://eass-hit-part-b.github.io/dML-paper/)

## Project Tasks

- [ ] Classical Mode (IMDB, MNIST) (Itay & Yair)
    - [ ] DP, TP, MP with Transformer, CNN+FC
- [ ] dML Mode (Dolphi + Hardhat) (Yair & Itay)
    - [ ] DP, TP, MP with Smart Contracts (IMDB, MNIST)
    - [ ] Models: Transformer, CNN+FC
    - [ ] zk-SNARK (for ZKP) (Yair)
- [ ] Testnet Connection (Optional) (Yair)
    - [ ] Demonstrate connection to a testnet (PoC) without exposing keys/secrets
- [ ] dML Mode (Dolphi + Substrate) (Marco)
    - [ ] DP, TP, MP with Substrate (Rust)
    - [ ] Models: Transformer (optional), CNN+FC
    - [ ] Integration with Torch & Burn (optional)
- [ ] Profiling & Comparison (Yair & Marco)
    - [ ] Compare performance across modes
    - [ ] Discuss outcomes
- [ ] Preprint & Website (Yossi et al.)
    - [ ] Publish preprint
    - [ ] Create a [website for dissemination](https://eass-hit-part-b.github.io/dML-paper/)

## Code Structure (see the project this is a bit outdated)

```plaintext
Dolphi-dML/
├── README.md
├── dml_hardhat/
│   ├── DP/
│   │   ├── contracts/
│   │   ├── models/
│   │   └── scripts/
│   ├── MP/
│   │   ├── contracts/
│   │   ├── models/
│   │   └── scripts/
│   ├── TP/
│   │   ├── contracts/
│   │   ├── models/
│   │   └── scripts/
│   └── zk-snark/
├── dml_substrate/
│   ├── DP/
│   │   ├── substrate/
│   │   ├── models/
│   │   └── integration/
│   ├── MP/
│   │   ├── substrate/
│   │   ├── models/
│   │   └── integration/
│   ├── TP/
│   │   ├── substrate/
│   │   ├── models/
│   │   └── integration/
├── testnet_connection/
│   ├── scripts/
│   └── configs/
├── profiling_comparison/
│   ├── dml_hardhat/
│   └── dml_substrate/
├── evaluation/
│   └── analysis/
└── scripts/
```
