“For the next 21 days, I will work on the Hallucination via Manifold Distance project for 30 minutes each day. I will define a core prototype that can compute LLM embedding distances to learned clusters/manifolds, evaluate them on 3 example queries, and publish a working notebook + README.”


# Hallucination via Manifold Distance

## Goal
Quantify the hallucination likelihood of LLM responses by measuring their distance to learned low-dimensional manifolds in embedding space.

## Motivation
Large Language Models are known to hallucinate, especially when faced with out-of-distribution prompts. This project explores whether the embedding space of LLMs forms disconnected low-dimensional manifolds, and if the distance of a response/query to these manifolds can signal hallucination or unreliability.

## Commitment

**Start Date:** Sept 27, 2025  
**End Date:** Oct 17, 2025  
**Daily Effort:** ~30 minutes  
**Deliverable:** A working Python prototype that:
- Embeds LLM queries/responses (OpenAI or HuggingFace)
- Learns manifold approximations (e.g., via clustering or GMM)
- Computes distance of new inputs to the nearest manifold
- Produces a “hallucination score” or visualization
- Is tested on at least 3 example tasks (e.g., TruthfulQA, fabricated prompts)

## Milestones

| Week | Goal |
|------|------|
| Week 1 | Setup, data collection, basic embedding + clustering |
| Week 2 | Distance metric experimentation + visualization |
| Week 3 | Evaluation on real prompts, polish, and write-up |



