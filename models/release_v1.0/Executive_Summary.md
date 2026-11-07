# ScamDefy Executive Summary Report
**Date**: 2026-02-19
**Version**: 1.0.0
**Model**: distilroberta-base
---
## 1. Executive Summary
ScamDefy v1.0 meets its clean-data F1 and false-positive targets on SMS spam data. The adversarial ladder shows it misses unobfuscated account-compromise messages (L0/L1) while catching obfuscated ones, so robustness results should be read with that in mind.

## 2. Key Performance Indicators (KPIs)
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| Clean F1 (Balanced) | 0.9462 | > 0.92 | ✅ |
| Production F1 (99:1) | 0.9231 | > 0.90 | ✅ |
| False Positive Rate | 0.1475%| < 0.5% | ✅ |

## 3. Adversarial Robustness Audit
Performance against escalating attack severity:
| Severity | Attack Type | Detection Probability | Outcome |
| :--- | :--- | :--- | :--- |
| L0 (Clean) | Your account has been compromi... | 0.0005 | ❌ BYPASSED |
| L1 (Homoglyph) | Yоur aсcount has been compromi... | 0.0003 | ❌ BYPASSED |
| L2 (Leetspeak) | Your account h@$ been c0mpr0m1... | 0.9985 | 🛡️ BLOCKED |
| L3 (Combined) | Your аcсoun+ ha$ b3en cоmрromi... | 0.9943 | 🛡️ BLOCKED |
| **L5 (Extreme)** | YoUr аСC0uNt ha$ beEN c0mproMi... | 0.5358 | 🛡️ BLOCKED |

**Robustness goal (v1.1)**: catch L0/L1 and raise L5 probability above 0.8. Note these are single-example probes, not detection rates.

## 4. Zero-Day & False Positive Stress Test
- **Zero-Day Evasion (Bank Variant)**: 0.9989 (PyTorch) / 0.9974 (ONNX)
- **False Positive Trap (Legit Urgent)**: 0.0003 (CORRECT)

## 5. Deployment & Cost Analysis
- **Latency**: 25.53 ms per message (ONNX, CPU, batch 1, one thread)
- **Throughput**: about 39 msgs/sec per thread
- **Cost**: about $1.69 per 1 million messages (AWS t3.xlarge, 70% utilisation, one thread)

*Corrected November 2026: the original report stated ~39,000 msgs/sec and $0.80 per million, which were hardcoded rather than measured.*

## 6. Recommendations
1. **Do not deploy v1.0 as-is**: it misses plain-text account-compromise scams (L0/L1) and was trained on SMS spam only.
2. **Future Hardening**: v1.1 training collapsed; curriculum learning is the next thing to try for 'L5 Extreme' (probe probability 0.54).
3. **Monitor**: Watch for 'homoglyph' and 'L5' style attacks in logs.