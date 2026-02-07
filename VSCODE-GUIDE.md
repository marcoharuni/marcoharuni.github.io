# 🚀 Super Simple Guide - Using VS Code Only (No Terminal!)

## ✅ One-Time Setup

### 1. Install VS Code
Download from: https://code.visualstudio.com

### 2. Install Git
```bash
sudo apt-get install git
```

### 3. Open Your Website in VS Code
1. Open VS Code
2. File → Open Folder
3. Select `Documents/marcoharuni.github.io`
4. Done! You're ready to go!

---

## 📝 Writing a Daily Blog Post (3 Steps!)

### Step 1: Copy the Template
1. In VS Code, open `blog/TEMPLATE.html`
2. Right-click on the file → Copy
3. Right-click on `blog` folder → Paste
4. Rename the copy to today's date: `2026-02-07-my-thoughts.html`

### Step 2: Edit Your Content
1. Open your new file `2026-02-07-my-thoughts.html`
2. Change **3 things at the top**:
   - Line 6: `<title>Your Post Title | Marco Haruni</title>` → Change "Your Post Title"
   - Line 44: `<h1>Your Post Title Here</h1>` → Change title
   - Line 45: `<div class="blog-post-meta">February 7, 2026 • 5 min read</div>` → Change date
3. **Write your content** between `<div class="blog-content">` and `</div>`
   - Use `<p>` for paragraphs
   - Use `<h2>` for headings
   - Use `<ul><li>` for lists
4. Save (Ctrl+S)

### Step 3: Add to Blog List
1. Open `blog.html`
2. Find the section that says `<!-- Add your blog posts here -->`
3. Add this **above** it:
```html
<div class="content-card">
    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">Feb 7, 2026</p>
    <h3><a href="blog/2026-02-07-my-thoughts.html">Your Post Title</a></h3>
    <p>Brief description of your post...</p>
    <div class="content-meta">
        <span class="tag">AI</span>
        <span class="tag">LLM</span>
    </div>
</div>
```
4. Change the date, filename, title, and description
5. Save (Ctrl+S)

---

## 🌐 Publishing Your Changes (VS Code Git)

### Easy Way (VS Code Built-in Git):

1. **See Your Changes**
   - Click the **Source Control** icon on left sidebar (3rd icon, looks like branches)
   - You'll see all files you changed

2. **Commit Changes**
   - In the message box at top, type: "Added new blog post"
   - Click the ✓ **Commit** button

3. **Push to GitHub**
   - Click the **...** (three dots) menu
   - Click **Push**
   - Done! Wait 1-2 minutes, your site is live!

### Even Easier Way (GitHub Desktop):

1. Download **GitHub Desktop**: https://desktop.github.com
2. Install and sign in with your GitHub account
3. File → Clone Repository → Select `marcoharuni.github.io`
4. Make your changes in VS Code (as normal)
5. Go to GitHub Desktop:
   - It shows all your changes automatically
   - Write commit message: "Added new blog post"
   - Click **Commit to main**
   - Click **Push origin**
6. Done! Live in 1-2 minutes!

---

## 📚 Adding Your Projects (Same Easy Way!)

1. Open `projects.html` in VS Code
2. Find where it says `<!-- Add more projects here -->`
3. Copy the existing project card above it
4. Paste and edit:
```html
<div class="content-card">
    <h3>Your Project Name</h3>
    <p>Description of what it does...</p>
    <div class="content-meta">
        <span class="tag">Python</span>
        <span class="tag">AI</span>
    </div>
    <a href="https://github.com/marcoharuni/your-repo" target="_blank">View on GitHub →</a>
</div>
```
5. Save, commit, push (same as above)

---

## 📄 Adding Publications

1. Open `publications.html`
2. Add:
```html
<div class="content-card">
    <h3>Paper Title: Your Amazing Research</h3>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 0.5rem;">Marco Haruni, Co-Author</p>
    <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;"><em>Conference Name</em> • 2026</p>
    <div style="display: flex; gap: 1rem;">
        <a href="https://arxiv.org/..." target="_blank">Paper →</a>
        <a href="publications/paper.pdf" target="_blank">PDF →</a>
        <a href="https://github.com/..." target="_blank">Code →</a>
    </div>
</div>
```

---

## 📚 Adding Books (with Download!)

1. Put your PDF in the `books/` folder (just drag and drop in VS Code)
2. Open `books.html`
3. Add:
```html
<div class="book-card">
    <div class="book-info">
        <h3>Your Book Title</h3>
        <p style="color: var(--text-secondary);">Description of what readers will learn...</p>
    </div>
    <div class="book-actions">
        <a href="books/your-book.pdf" class="btn" download>Download PDF</a>
        <a href="https://amazon.com/your-book" target="_blank" class="btn btn-primary">Buy on Amazon</a>
    </div>
</div>
```

---

## 🎨 Writing Tips

### Simple Text
Just write between `<p>` tags:
```html
<p>This is a paragraph. Easy!</p>
```

### Headings
```html
<h2>Big Heading</h2>
<h3>Smaller Heading</h3>
```

### Lists
```html
<ul>
    <li>First point</li>
    <li>Second point</li>
    <li>Third point</li>
</ul>
```

### Bold and Italic
```html
<strong>Bold text</strong>
<em>Italic text</em>
```

### Links
```html
<a href="https://example.com">Click here</a>
```

### Code Blocks
```html
<pre><code>def my_function():
    return "Hello, AGI!"
</code></pre>
```

### Images (put image in `images/` folder first)
```html
<img src="../images/diagram.png" alt="Description" style="border-radius: 12px; margin: 2rem 0;">
```

---

## ⚡ Quick Daily Workflow Example

**Morning routine:**
1. Open VS Code → Open `marcoharuni.github.io` folder
2. Copy `blog/TEMPLATE.html` → Rename to today's date
3. Write your thoughts (15-30 minutes)
4. Add to `blog.html` list
5. Source Control → Commit → Push
6. Done! Live in 2 minutes! ☕

---

## 🔥 Pro Tips

### Preview Locally Before Publishing
1. In VS Code, right-click `index.html`
2. Select "Open with Live Server" (install Live Server extension first)
3. Your site opens in browser
4. Make changes, see them instantly!

### Install Live Server Extension
1. Click Extensions icon (left sidebar, looks like squares)
2. Search "Live Server"
3. Click Install
4. Now you can preview your site!

---

## ❓ Common Questions

**Q: Do I need to use the terminal/bash?**
A: **NO!** Just use VS Code + GitHub Desktop. Never touch terminal!

**Q: How do I add code with syntax highlighting?**
A: Add this before `</head>` in your blog post:
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
```

**Q: How do I add math equations?**
A: See the comment in `blog/TEMPLATE.html` - just copy that code!

**Q: Can I write in Markdown instead of HTML?**
A: Not by default, but HTML is actually easier for your setup! Just copy-paste and edit.

**Q: What if I make a mistake?**
A: No worries! Just edit the file again and push. Git keeps history, so you can always go back.

---

## ✅ You're All Set!

Your workflow is now super simple:
1. ✅ Open VS Code
2. ✅ Edit files (copy templates, paste, modify)
3. ✅ Commit in VS Code (Source Control)
4. ✅ Push
5. ✅ Live in 2 minutes!

**No terminal. No complex commands. Just edit and push!** 🚀

---

**Your mission**: Write everyday. Build everyday. Share everyday.

The world needs your work. Go build AGI! 🤖
