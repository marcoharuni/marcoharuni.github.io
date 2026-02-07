#!/usr/bin/env python3
"""
Simple Markdown to HTML converter for Marco Haruni's blog.
Just write .md files in blog/ folder and run: python3 convert_blog.py

First time setup:
    sudo apt install python3-markdown python3-pygments
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import markdown
    from markdown.extensions import fenced_code, tables, toc, codehilite
except ImportError:
    print("❌ Error: Required packages not installed.\n")
    print("📦 Please install them with ONE of these methods:\n")
    print("Option 1 (Recommended - System packages):")
    print("    sudo apt install python3-markdown python3-pygments\n")
    print("Option 2 (Virtual environment):")
    print("    python3 -m venv venv")
    print("    source venv/bin/activate")
    print("    pip install markdown pygments")
    sys.exit(1)

# HTML template for blog posts
BLOG_POST_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Marco Haruni</title>
    <meta name="description" content="{description}">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>">
    <link rel="stylesheet" href="../css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <!-- Code highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">

    <!-- Math support -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <div class="nav-wrapper">
                <a href="../index.html" class="logo">Marco Haruni</a>
                <button class="mobile-toggle"><span></span><span></span><span></span></button>
                <ul class="nav-menu">
                    <li><a href="../index.html">Home</a></li>
                    <li><a href="../about.html">About</a></li>
                    <li><a href="../projects.html">Projects</a></li>
                    <li><a href="../blog.html" class="active">Blog</a></li>
                    <li><a href="../publications.html">Publications</a></li>
                    <li><a href="../books.html">Books</a></li>
                    <li><button class="theme-toggle"><svg class="sun-icon" width="18" height="18" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/></svg><svg class="moon-icon" width="18" height="18" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/></svg></button></li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="blog-post">
        <a href="../blog.html" style="display: inline-flex; align-items: center; gap: 0.5rem; margin-bottom: 2rem; color: var(--text-secondary);">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
            </svg>
            Back to Blog
        </a>

        <h1>{title}</h1>
        <div class="blog-post-meta">{date} • {read_time} min read</div>

        <div class="blog-content">
            {content}
        </div>
    </div>

    <footer class="footer-minimal">
        <div class="container">
            <p>&copy; 2026 Marco Haruni. All rights reserved.</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="../js/main.js"></script>

    <script>
    // Render math
    document.addEventListener("DOMContentLoaded", function() {{
        renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}},
                {{left: '\\\\[', right: '\\\\]', display: true}},
                {{left: '\\\\(', right: '\\\\)', display: false}}
            ],
            throwOnError: false
        }});
    }});
    </script>
</body>
</html>
"""

def parse_frontmatter(content):
    """Extract YAML frontmatter from markdown file."""
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    markdown_content = match.group(2)

    # Parse simple YAML
    frontmatter = {}
    for line in frontmatter_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    return frontmatter, markdown_content

def estimate_read_time(content):
    """Estimate reading time (200 words per minute)."""
    words = len(content.split())
    minutes = max(1, round(words / 200))
    return minutes

def convert_markdown_to_html(md_file):
    """Convert a single markdown file to HTML."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse frontmatter
    frontmatter, markdown_content = parse_frontmatter(content)

    # Get metadata
    title = frontmatter.get('title', 'Untitled')
    description = frontmatter.get('description', '')
    date = frontmatter.get('date', datetime.now().strftime('%B %d, %Y'))
    tags = frontmatter.get('tags', '').split(',')
    tags = [tag.strip() for tag in tags if tag.strip()]

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'fenced_code',
        'tables',
        'toc',
        'codehilite',
        'nl2br',
        'sane_lists'
    ])
    html_content = md.convert(markdown_content)

    # Estimate read time
    read_time = estimate_read_time(markdown_content)

    # Generate HTML
    html = BLOG_POST_TEMPLATE.format(
        title=title,
        description=description,
        date=date,
        read_time=read_time,
        content=html_content
    )

    # Output file name
    base_name = Path(md_file).stem
    html_file = f"blog/{base_name}.html"

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Converted: {md_file} -> {html_file}")

    return {
        'filename': f"{base_name}.html",
        'title': title,
        'description': description,
        'date': date,
        'tags': tags,
        'read_time': read_time
    }

def update_blog_index(posts):
    """Update blog.html with list of all posts."""

    # Sort posts by date (most recent first)
    posts.sort(key=lambda x: x['date'], reverse=True)

    # Generate post cards HTML
    cards_html = ""
    for post in posts:
        tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in post['tags']])

        card = f"""                <div class="content-card">
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">{post['date']}</p>
                    <h3><a href="blog/{post['filename']}">{post['title']}</a></h3>
                    <p>{post['description']}</p>
                    <div class="content-meta">
                        {tags_html}
                    </div>
                </div>
"""
        cards_html += card

    # Read current blog.html
    with open('blog.html', 'r', encoding='utf-8') as f:
        blog_html = f.read()

    # Replace content-grid section
    pattern = r'(<div class="content-grid">).*?(</div>\s*</div>\s*</section>)'
    replacement = f'\\1\n{cards_html}\n            \\2'

    new_blog_html = re.sub(pattern, replacement, blog_html, flags=re.DOTALL)

    with open('blog.html', 'w', encoding='utf-8') as f:
        f.write(new_blog_html)

    print(f"\n✓ Updated blog.html with {len(posts)} posts")

def main():
    """Main conversion function."""
    print("🚀 Converting Markdown blog posts to HTML...\n")

    # Find all .md files in blog/
    md_files = list(Path('blog').glob('*.md'))

    if not md_files:
        print("No .md files found in blog/ folder.")
        print("Create a file like: blog/2026-02-08-my-first-post.md")
        return

    # Convert all markdown files
    posts = []
    for md_file in md_files:
        post_data = convert_markdown_to_html(md_file)
        posts.append(post_data)

    # Update blog index
    update_blog_index(posts)

    print("\n✅ All done! Your blog is ready.")
    print("📝 To add new posts: Create .md files in blog/ and run this script again.")

if __name__ == "__main__":
    main()
