# SCH: SALT–SFLT Models for Stochastic Camassa–Holm Dynamics

## 👋 Overview
This repository contains numerical implementations of stochastic Camassa–Holm (SCH) models using **structure-preserving stochastic transport frameworks**:
- **SALT** (Stochastic Advection by Lie Transport)
- **SFLT** (Stochastic Forcing by Lie Transport)

These models introduce stochasticity in a geometrically consistent way, preserving key physical structures of the underlying PDEs.

---

## 🔬 Motivation
Stochastic parameterisations such as SALT and SFLT provide a principled way to model unresolved scales in fluid dynamics and geophysical systems. They decompose deterministic dynamics into drift and stochastic components while maintaining conservation properties. :contentReference[oaicite:0]{index=0}

This repository focuses on:
- Numerical simulation of SCH systems
- Comparison between SALT and SFLT formulations
- Data generation for uncertainty quantification and data assimilation

---

## ⚙️ Key Features
- Structure-preserving stochastic discretisations
- Particle-based simulations of SCH dynamics
- Support for ensemble generation
- Output for post-processing and visualization (e.g., `.pvd`, field data)

---

## 📁 Repository Structure
