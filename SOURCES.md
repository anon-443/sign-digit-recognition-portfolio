# Data provenance and citation

## Sign Language Digits Dataset

- Upstream repository: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- Reproducible revision pinned in `gesturevision.digit_data`: `4b09433ef9d5c60602e8805bb54b8aa2428527f8` (2021-05-08).
- Upstream states that the images are 100×100 RGB, contain ten classes (0–9), and were prepared by Turkey Ankara Ayrancı Anadolu High School students; its README describes 218 students and ten samples per student.
- This revision contains **2,062 image files**. The observed counts in the pinned archive are: 0: 205, 1: 206, 2: 206, 3: 206, 4: 207, 5: 207, 6: 207, 7: 206, 8: 208, 9: 204. All 2,062 image-file SHA-256 hashes are unique in the acquired snapshot.
- The repository is licensed **Apache-2.0**; the complete upstream license text is preserved in [`licenses/DATASET_APACHE_2.0.txt`](licenses/DATASET_APACHE_2.0.txt). The raw images remain under `data/` and are not included in source archives.
- Upstream citation: Mavi, A. (2020). “A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits.” arXiv:2011.08927 [cs.CV]. https://arxiv.org/abs/2011.08927

## Evaluation caveat

The upstream repository does not provide participant IDs or an official participant-disjoint train/test manifest. This project creates deterministic nested `StratifiedGroupKFold` train/validation/test partitions grouped by exact image SHA-256 to prevent byte-identical images crossing partitions. That prevents exact-file leakage only; it **does not** establish person-independent generalization. Treat the held-out result as a small, repository-derived benchmark, not evidence of general ASL recognition or deployment performance.
