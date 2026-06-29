---
layout: page
title: Home
---

<section class="landing-intro">
  <h1>Juan Pablo Oberhauser</h1>
  <p class="lede">
    Computer vision scientist developing behavioral phenotyping systems for laboratory mice in preclinical research,
    spanning detection, tracking, pose estimation, and behavioral classification.
  </p>
  <p class="lede-interests">
    <strong>Interested in:</strong> self-supervised pretraining, multi-task learning, contrastive learning,
    masked pretraining, vision-language models, few-shot learning, temporal sequence modeling,
    knowledge distillation, transformer architectures, active learning.
  </p>
</section>

<section class="landing-section">
  <h2>Publications</h2>
  <ul class="pub-list">
    <li>
      <div class="pub-meta">December 2025 · <em>PNAS</em></div>
      <div class="pub-title"><a href="https://www.biorxiv.org/content/10.1101/2025.10.29.685181v1">Multi-week digital home cage monitoring reduces noise and enhances reproducibility</a></div>
    </li>
    <li>
      <div class="pub-meta">June 2025 · <em>CVPR 2025, CV4Animals</em> (poster + oral)</div>
      <div class="pub-title"><a href="https://arxiv.org/pdf/2507.07929">Towards Continuous Home Cage Monitoring: An Evaluation of Tracking and Identification Strategies for Laboratory Mice</a></div>
    </li>
    <li>
      <div class="pub-meta">December 2024 · <em>bioRxiv</em></div>
      <div class="pub-title"><a href="https://www.biorxiv.org/content/10.1101/2024.12.18.629281v2">An integrated and scalable rodent cage system enabling continuous computer vision-based behavioral analysis and AI-enhanced digital biomarker development</a></div>
    </li>
  </ul>
</section>

<section class="landing-section">
  <h2>Open courses I am creating</h2>
  <p class="section-lede">I built these because I couldn't find ground-up resources on these topics. Each one is a full skeleton with exercise notebooks and worked solutions.</p>
  <div class="card-grid">
    <a class="card" href="https://github.com/jpoberhauser/vision-transformers-and-ssl">
      <div class="card-title">Vision Transformers &amp; Self-Supervised Learning</div>
      <div class="card-desc">Hands-on course covering ViT architectures and modern SSL methods.</div>
      <div class="card-link">github.com/jpoberhauser/vision-transformers-and-ssl</div>
    </a>
    <a class="card" href="https://github.com/jpoberhauser/vision_language_course">
      <div class="card-title">Vision-Language Course</div>
      <div class="card-desc">Self-paced material on multimodal vision-language models.</div>
      <div class="card-link">github.com/jpoberhauser/vision_language_course</div>
    </a>
  </div>
</section>

<section class="landing-section">
  <h2>Projects</h2>
  <div class="card-grid">
    <a class="card" href="https://github.com/jpoberhauser/pybaseball">
      <div class="card-title">pybaseball</div>
      <div class="card-desc">Fork of pybaseball supercharged with LLMs for easy natural language question answering! Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraphs).</div>
      <div class="card-link">github.com/jpoberhauser/pybaseball</div>
    </a>
    <a class="card" href="https://github.com/jpoberhauser/multi-object-trackers-collection">
      <div class="card-title">Multi-Object Trackers Collection</div>
      <div class="card-desc">Keeping up with multi-object trackers. A collection, organized by type, of the latest trackers, ReID, and surveys — plus all the building blocks of trackers.</div>
      <div class="card-link">github.com/jpoberhauser/multi-object-trackers-collection</div>
    </a>
    <a class="card" href="https://github.com/jpoberhauser/baseballCompanion">
      <div class="card-title">baseballCompanion</div>
      <div class="card-desc">Ask natural language questions about the current state of baseball, grounded in recent YouTube analysis videos. Uses Whisper for transcription, SentenceTransformers for embeddings, FAISS as the vector store, and llama.cpp for local LLM inference with RAG (currently Mistral 7B Instruct, swappable backends). No API keys required.</div>
      <div class="card-link">github.com/jpoberhauser/baseballCompanion</div>
    </a>
  </div>
</section>

<section class="landing-section">
  <h2>Recent notes</h2>
  <ul class="post-list-compact">
    {% for post in site.posts limit:5 %}
      <li>
        <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
        <a href="{{ post.url | relative_url }}">{{ post.title | default: post.name }}</a>
      </li>
    {% endfor %}
  </ul>
  <p><a href="{{ '/blog/' | relative_url }}">See all posts →</a></p>
</section>
