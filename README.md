# VISIONFORGE

**Vision-to-3D Production AI Agent** — Turn reference photos into packed, QA-verified 3D assets.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Blender](https://img.shields.io/badge/blender-3.6%20%7C%204.x-orange)

## 📖 Overview

VisionForge is an agentic pipeline that converts one or more reference images (plus an optional text brief) into a production-ready `.blend` file. The output contains a QuadriFlow-remeshed, UV-unwrapped, 4K-PBR-baked asset that is automatically validated against quality metrics.

The system is designed to be **resilient** and **resource-aware**. Heavy lifting (segmentation, 3D reconstruction, Blender baking) runs locally as subprocesses. An LLM agent orchestrates the stages, making intelligent decisions about topology, baking settings, and quality control based on episodic memory from previous runs.

## ✨ Key Features

*   **Agentic Orchestration:** Two-tier routing system (System 1 for fast, deterministic decisions; System 2 for deep reasoning on ambiguous inputs).
*   **Provider Failover:** Automatic fallback between multiple LLM providers. Rate limits and quota errors are intercepted and cooled down transparently.
*   **Episodic Memory:** Uses SQLite to remember successful past runs and recommend optimal remesh/bake settings for new objects.
*   **Headless Blender Automation:** A single script that works across Blender 3.6 LTS and 4.x, handling API changes (e.g., Principled BSDF socket aliases).
*   **Automated QA:** Delivers assets with a clear label (`DRAFT`, `STATIC_READY`, `RIG_READY_CANDIDATE`, `ANIMATION_READY`) based on geometric and texture validation.

## 🏗️ Architecture

```text
Images + Prompt
      │
      ▼
┌─────────────────┐
│  System 1       │  Fast, deterministic rules (0 tokens)
│  Fast Router    │  + 1-token classifier for gray zones
└────────┬────────┘
         │ (if ambiguous)
         ▼
┌─────────────────┐
│  System 2       │  Deep reasoning with episodic few-shot examples
│  Reasoning      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LiteLLM        │  Provider Cascade (Primary → Fallbacks)
│  Cascade        │  Handles rate limits & auth errors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Local Pipeline │  Preprocess → Reconstruct → Blender Stage
│  (Subprocesses) │  (QuadriFlow · UV · 4K Bake · Pack)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QA & Memory    │  Turntable render · SSIM/LPIPS/IoU
│  Verification   │  Record run to SQLite
└─────────────────┘
