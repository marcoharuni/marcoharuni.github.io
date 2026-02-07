# 🚀 Your Website - Complete Daily Usage Guide

## 🎯 Quick Start

Your website: `/home/miss_merry/Documents/marcoharuni.github.io/`

**View locally**: Double-click `index.html`

---

## ✍️ Writing Blog Posts (Daily)

### Option 1: Simple Text Blog Post

1. **Create new file** in `blog/` folder:
```bash
cd ~/Documents/marcoharuni.github.io/blog
nano 2026-02-08-my-thoughts.html
```

2. **Copy this template**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Post Title | Marco Haruni</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>">
    <link rel="stylesheet" href="../css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
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

        <h1>Your Post Title Here</h1>
        <div class="blog-post-meta">February 8, 2026 • 5 min read</div>

        <div class="blog-content">
            <p>Your first paragraph here...</p>

            <h2>Your Subheading</h2>
            <p>More content...</p>

            <h3>Smaller Heading</h3>
            <p>More text...</p>

            <ul>
                <li>Point 1</li>
                <li>Point 2</li>
            </ul>

            <p>More paragraphs...</p>
        </div>
    </div>

    <footer class="footer-minimal">
        <div class="container">
            <p>&copy; 2026 Marco Haruni. All rights reserved.</p>
        </div>
    </footer>

    <script src="../js/main.js"></script>
</body>
</html>
```

3. **Add your content** between the `<div class="blog-content">` tags

4. **Add to blog list** - Edit `blog.html` and add:
```html
<div class="content-card">
    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">Feb 8, 2026</p>
    <h3><a href="blog/2026-02-08-my-thoughts.html">Your Post Title</a></h3>
    <p>Brief description...</p>
    <div class="content-meta">
        <span class="tag">AI</span>
        <span class="tag">Research</span>
    </div>
</div>
```

---

### Option 2: Blog Post with CODE

Add code blocks like this:

```html
<pre><code>def hello_world():
    print("Hello, AGI!")
    return "Building the future"

hello_world()
</code></pre>
```

**For syntax highlighting** (optional), add before `</head>`:
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
```

---

### Option 3: Blog Post with MATH

Add math equations:

```html
<!-- Add before </head> -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ]
    });
});
</script>

<!-- Then use in your content -->
<p>Inline math: $E = mc^2$</p>

<p>Display math:</p>
$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$
```

---

## 📚 Adding Books

Edit `books.html` and add:

```html
<div class="book-card">
    <div class="book-info">
        <h3>Your Book Title</h3>
        <p style="color: var(--text-secondary);">Description of your book and what readers will learn...</p>
    </div>
    <div class="book-actions">
        <a href="books/your-book.pdf" class="btn" download>Download PDF</a>
        <a href="https://amazon.com/your-book" target="_blank" class="btn btn-primary">Buy on Amazon</a>
    </div>
</div>
```

**To add downloadable PDF**:
1. Put PDF in `books/` folder
2. Link to it: `href="books/your-book.pdf"`

---

## 🔬 Adding Projects

Edit `projects.html` and add:

```html
<div class="content-card">
    <h3>Project Name</h3>
    <p>What this project does and why it matters...</p>
    <div class="content-meta">
        <span class="tag">Python</span>
        <span class="tag">PyTorch</span>
        <span class="tag">AGI</span>
    </div>
    <a href="https://github.com/marcoharuni/project" target="_blank">View on GitHub →</a>
</div>
```

---

## 📄 Adding Publications

Edit `publications.html` and add:

```html
<div class="content-card">
    <h3>Paper Title: Building Better AI Systems</h3>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 0.5rem;">Marco Haruni, Co-Author Name</p>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;"><em>NeurIPS</em> • 2026</p>
    <div style="display: flex; gap: 1rem;">
        <a href="https://arxiv.org/..." target="_blank">Paper →</a>
        <a href="publications/paper.pdf" target="_blank">PDF →</a>
        <a href="https://github.com/..." target="_blank">Code →</a>
    </div>
</div>
```

---

## 🌐 Deploying Updates

### Method 1: GitHub Website (Easiest)
1. Go to your repo on GitHub
2. Click on file you changed
3. Click "Edit" (pencil icon)
4. Make changes
5. "Commit changes"
6. Wait 1 minute → Live!

### Method 2: Git Command Line
```bash
cd ~/Documents/marcoharuni.github.io
git add .
git commit -m "Added new blog post"
git push
```

---

## 🎨 Customization Tips

### Change Theme Colors
Edit `css/style.css` (line 7-8):
```css
--accent: #1d9bf0;  /* Change to your preferred color */
```

### Change Fonts
Edit in HTML `<head>`:
```html
<link href="https://fonts.googleapis.com/css2?family=YOUR_FONT&display=swap" rel="stylesheet">
```

Then in CSS:
```css
--font-main: 'YOUR_FONT', sans-serif;
```

---

## 📊 Adding Images

1. **Put image in `images/` folder**
```bash
cp ~/Downloads/diagram.png ~/Documents/marcoharuni.github.io/images/
```

2. **Use in blog post**:
```html
<img src="../images/diagram.png" alt="Description" style="border-radius: 12px; margin: 2rem 0;">
```

---

## 🔥 Pro Tips

### Quick Blog Template Script
```bash
#!/bin/bash
# Save as: create-post.sh

DATE=$(date +%Y-%m-%d)
TITLE=$1
FILE="blog/${DATE}-${TITLE}.html"

# Copy template
cp blog/welcome.html "$FILE"

# Edit
nano "$FILE"

echo "Created: $FILE"
```

Use: `./create-post.sh my-new-post`

### Add Social Share Buttons
```html
<div style="display: flex; gap: 1rem; margin: 2rem 0;">
    <a href="https://twitter.com/intent/tweet?url=YOUR_URL&text=YOUR_TITLE" target="_blank" class="btn">Share on X</a>
    <a href="https://www.linkedin.com/sharing/share-offsite/?url=YOUR_URL" target="_blank" class="btn">Share on LinkedIn</a>
</div>
```

### Add Comments (Giscus)
Add before `</body>`:
```html
<script src="https://giscus.app/client.js"
        data-repo="marcoharuni/marcoharuni.github.io"
        data-repo-id="YOUR_REPO_ID"
        data-category="Comments"
        data-category-id="YOUR_CATEGORY_ID"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>
```

---

## ✅ Daily Workflow Example

```bash
# Morning routine
cd ~/Documents/marcoharuni.github.io

# Write today's blog
nano blog/2026-02-08-building-agi.html
# (Write your content)

# Update blog list
nano blog.html
# (Add link to new post)

# Test locally
xdg-open index.html

# Deploy
git add .
git commit -m "Daily post: Building AGI"
git push

# Done! Live in 1 minute 🎉
```

---

## 🚀 Your Journey Starts Now!

**Your website has**:
- ✅ Beautiful modern design
- ✅ X-style dim/light mode
- ✅ Animated social links
- ✅ Code & math support
- ✅ Mobile responsive
- ✅ Fast & professional
- ✅ Easy to update

**Write everyday. Build everyday. Share everyday.**

The world needs your work. Go build AGI! 🤖

---

**Need help?** Just ask or check README.md

**Your mission**: Serve humanity through AGI. Your website is ready. 🌟
