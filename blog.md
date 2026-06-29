---
layout: page
title: Blog
permalink: /blog/
---

<ul class="post-list-compact">
  {% for post in site.posts %}
    <li>
      <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
      <a href="{{ post.url | relative_url }}">{{ post.title | default: post.name }}</a>
    </li>
  {% endfor %}
</ul>
