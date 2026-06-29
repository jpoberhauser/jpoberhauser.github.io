---
layout: page
title: Home
---

<section class="landing-intro">
  <h1>Pablo Oberhauser</h1>
  <p class="lede">
    Data Scientist with 10+ years of experience in computer vision, deep learning, and tabular ML.
    I build models for continuous behavioral monitoring, design self-paced learning material,
    and write notes on the papers and tools I use along the way.
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
  <h2>Self-paced courses</h2>
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
      <div class="card-desc">Baseball data analysis and modeling in Python.</div>
      <div class="card-link">github.com/jpoberhauser/pybaseball</div>
    </a>
    <a class="card" href="https://github.com/jpoberhauser/multi-object-trackers-collection">
      <div class="card-title">Multi-Object Trackers Collection</div>
      <div class="card-desc">A reference collection of multi-object tracking implementations.</div>
      <div class="card-link">github.com/jpoberhauser/multi-object-trackers-collection</div>
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
