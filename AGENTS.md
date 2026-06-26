# AGENTS.md

# HirePulse AI - AI Coding Agent Instructions

## Project Overview

HirePulse is an AI-powered candidate discovery and ranking platform built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.

The objective is NOT keyword matching.

The objective is to understand a Job Description like an experienced recruiter, retrieve the most relevant candidates from a dataset of 100,000 candidates, rank them using multiple intelligent signals, and generate explainable recommendations.

This project must be production-quality, modular, explainable, and easy to maintain.

---

# High-Level Architecture

Job Description
        │
        ▼
JD Understanding
        │
        ▼
Structured Job Object
        │
────────────────────────────────────────────
        │
        ▼
Candidate Processing
        │
        ▼
Candidate Features
        │
        ▼
Hybrid Retrieval
        │
        ▼
Hybrid Ranking
        │
        ▼
Explanation Generator
        │
        ▼
Top 100 Candidates

---

# Project Structure

backend/

core/
Shared configuration, constants, helper utilities.

jd/
Job Description parsing and understanding.

candidate/
Candidate parsing and feature engineering.

models/
Pydantic domain models.

retrieval/
BM25, Dense Retrieval, Hybrid Retrieval.

ranking/
Ranking algorithms and scoring.

explainability/
Candidate explanation generation.

validation/
Profile validation and anomaly detection.

pipelines/
End-to-end orchestration pipelines.

frontend/
User Interface.

data/
Raw and processed datasets.

tests/
Unit and integration tests.

configs/
Configuration files.

docs/
Architecture and documentation.

---

# Development Philosophy

Always build the system from the inside out.

Order of implementation:

1. Domain Models
2. Parsers
3. Feature Engineering
4. Retrieval
5. Ranking
6. Explainability
7. UI

Never build UI before backend functionality exists.

---

# Domain Driven Design

Everything revolves around three primary models.

JobDescription

Candidate

RankingResult

Additional feature models may exist, but these remain the core business entities.

---

# Coding Standards

Use Python 3.12+

Use Pydantic v2 for models.

Use type hints everywhere.

Write descriptive docstrings.

Prefer composition over inheritance.

Keep modules small.

Avoid global variables.

Avoid giant utility files.

Functions should have one responsibility.

Code should be readable before being clever.

---

# AI Agent Rules

When generating code:

Only modify files explicitly requested.

Never rewrite unrelated files.

Never rename project folders.

Never introduce new frameworks unless requested.

Never add unnecessary dependencies.

Never create placeholder TODO comments.

Never generate mock implementations unless requested.

Never silently change interfaces.

If assumptions are required, explain them.

---

# Retrieval Philosophy

Retrieval is NOT ranking.

Retrieval narrows candidates.

Ranking decides ordering.

Keep these systems independent.

---

# Ranking Philosophy

Ranking should consider multiple independent signals.

Examples include:

Technical Match

Semantic Similarity

Experience Match

Behavioral Signals

Recruitability

Location Compatibility

Notice Period

Validation Score

No single score should dominate unless explicitly designed.

---

# Explainability

Every ranking decision should be explainable.

The system should always be able to answer:

Why was this candidate ranked here?

Avoid hallucinated explanations.

Use only factual information from candidate data.

---

# Validation

Validate candidate data before ranking.

Detect impossible timelines.

Detect inconsistent experience.

Detect suspicious profiles.

Never ignore validation.

---

# Performance Goals

Design for:

100,000 candidates

CPU-only execution

Fast retrieval

Low memory usage

Modular pipeline

No network dependency during ranking.

---

# Git Workflow

One feature = one commit.

Keep commits small.

Every commit should represent one meaningful capability.

Preferred commit examples:

feat(models): add job description model

feat(candidate): implement candidate parser

feat(retrieval): add BM25 retrieval

feat(ranking): implement weighted ranking engine

Avoid huge AI-generated commits.

---

# Testing

Write tests whenever practical.

Never merge code that obviously fails.

Prefer deterministic behavior.

---

# Documentation

Public functions should be documented.

Complex algorithms should include concise explanations.

Avoid excessive comments explaining obvious code.

---

# General Principle

Build software like an engineering team, not like an AI demo.

Correctness over cleverness.

Maintainability over shortcuts.

Readability over complexity.

Every module should have a single clear responsibility.

