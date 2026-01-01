---
layout: default
title: "Blog"
---

# Blog

Welcome to my blog.  
I write about AI, coding, research, and Philosophy of AI.

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> <span style="color:#aaa;">({{ post.date | date: "%b %d, %Y" }})</span>
    </li>
  {% endfor %}
</ul>
