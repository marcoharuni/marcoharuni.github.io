# 🚀 Write Blog Posts in Markdown (Like Karpathy!)

## 📦 One-Time Setup (Do This Once!)

Run this command once to install required packages:

```bash
cd ~/Documents/marcoharuni.github.io
bash setup.sh
```

Or manually:
```bash
sudo apt install python3-markdown python3-pygments
```

That's it! You're ready to write in Markdown!

---

## ✨ Super Simple Workflow

1. **Write** your post in Markdown (`.md` file)
2. **Run** one command: `python3 convert_blog.py`
3. **Push** to GitHub
4. **Done!** Your blog is live in 2 minutes

No HTML editing needed! Just write naturally.

---

## 📝 Quick Start

### 1. Create a New Blog Post

In VS Code:
1. Open `blog/` folder
2. Create new file: `2026-02-08-my-post.md` (use today's date)
3. Copy the template from `blog/TEMPLATE.md`
4. Start writing!

### 2. Write Your Content

Your post starts with metadata (frontmatter):

```markdown
---
title: Building Large Language Models
description: A deep dive into LLM architecture and training
date: February 8, 2026
tags: AI, LLM, Deep Learning
---

Your content starts here...
```

Then write naturally! Just like writing a document.

### 3. Convert to HTML

Open terminal (or VS Code terminal) and run:

```bash
cd ~/Documents/marcoharuni.github.io
python convert_blog.py
```

That's it! The script:
- ✅ Converts all `.md` files to beautiful HTML
- ✅ Adds syntax highlighting for code
- ✅ Renders math equations
- ✅ Updates your blog index automatically

### 4. Publish

In VS Code:
1. Source Control icon (left sidebar)
2. Type commit message: "New blog post"
3. Click ✓ Commit
4. Click ... → Push
5. Live in 2 minutes!

---

## 📖 Writing Guide

### Basic Text

```markdown
This is a paragraph. Just write normally!

This is another paragraph. Leave blank lines between paragraphs.

You can make text **bold** or *italic* or ***both***.
```

### Headings

```markdown
## Main Section Heading

### Subsection Heading

#### Smaller Heading
```

### Code Blocks

Just use triple backticks with the language name:

````markdown
```python
def hello_world():
    print("Hello, AGI!")
    return 42
```
````

Supports: Python, JavaScript, Bash, C++, Java, Go, Rust, and many more!

### Math Equations

**Inline math** (in the middle of text):
```markdown
The formula $E = mc^2$ is Einstein's famous equation.
```

**Display math** (centered, larger):
```markdown
$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(w_t | w_{<t}; \theta)
$$
```

Complex equations:
```markdown
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

### Lists

**Bullet points:**
```markdown
- First point
- Second point
  - Nested point
  - Another nested point
- Third point
```

**Numbered lists:**
```markdown
1. First step
2. Second step
3. Third step
```

### Images

1. Put image in `images/` folder (drag & drop in VS Code)
2. Reference it:

```markdown
![Alt text](../images/my-diagram.png)
```

With custom styling:
```markdown
<img src="../images/plot.png" alt="Results" style="border-radius: 12px; max-width: 100%;">
```

### Tables

```markdown
| Model | Parameters | Performance |
|-------|-----------|-------------|
| GPT-2 | 1.5B | 85.2% |
| GPT-3 | 175B | 92.4% |
| GPT-4 | 1.8T | 96.3% |
```

### Links

```markdown
[Link text](https://example.com)
[Another post](2026-02-07-previous-post.html)
```

### Block Quotes

```markdown
> "The best way to predict the future is to invent it."
> — Alan Kay
```

### Inline Code

```markdown
Use the `torch.nn.Transformer` class for this.
```

---

## 🎨 Advanced Features

### Footnotes

```markdown
This is some text[^1] with a footnote.

[^1]: This is the footnote content.
```

### Horizontal Lines

```markdown
---
```

### HTML (when needed)

You can use HTML when you need precise control:

```markdown
<div style="background: var(--surface); padding: 1rem; border-radius: 8px;">
Custom styled content here
</div>
```

---

## 💡 Examples Like Karpathy/Raschka

### Example 1: Explaining an Algorithm

```markdown
## The Backpropagation Algorithm

Let's walk through backprop step by step.

### Forward Pass

Given input $x$, compute:

$$
h = \sigma(Wx + b)
$$

```python
def forward(x, W, b):
    z = x @ W + b
    h = torch.relu(z)
    return h, z
```

### Backward Pass

Compute gradients using chain rule...
```

### Example 2: Architecture Explanation

```markdown
## Transformer Architecture

The transformer uses multi-head attention:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

where each head computes:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # ... implementation
```
```

### Example 3: Training Results

```markdown
## Results

Here are the training curves:

![Training Loss](../images/training-loss.png)

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 2.43 | 67.2% |
| 10 | 0.87 | 89.5% |
| 100 | 0.12 | 98.1% |

The model converged after ~50 epochs.
```

---

## ⚡ Daily Workflow

### Morning Routine

```bash
# 1. Open VS Code
code ~/Documents/marcoharuni.github.io

# 2. Create new .md file in blog/
# File → New File → blog/2026-02-08-my-thoughts.md

# 3. Write your content (15-30 min)

# 4. Convert to HTML
python convert_blog.py

# 5. Check locally (optional)
# Open blog/2026-02-08-my-thoughts.html in browser

# 6. Commit and push
git add .
git commit -m "New post: My thoughts on AI"
git push

# Done! ☕
```

### VS Code Only (No Terminal)

1. **Write:** Create `.md` file in `blog/` folder
2. **Convert:** In VS Code, press `Ctrl+`` to open terminal, type `python convert_blog.py`
3. **Commit:** Source Control → Commit → Push
4. **Live!** 🎉

---

## 🔥 Pro Tips

### 1. Live Preview in VS Code

Install "Markdown Preview Enhanced" extension:
- Press `Ctrl+Shift+V` to see preview while writing
- See exactly how it will look!

### 2. Write First, Edit Later

Don't worry about perfection. Just write your thoughts:
1. Brain dump everything
2. Run converter to see HTML
3. Edit and improve
4. Push when ready

### 3. Use TEMPLATE.md

Always start by copying `blog/TEMPLATE.md`:
- Has all examples
- Shows all features
- Just delete what you don't need

### 4. Test Math Locally

Before pushing, make sure your math renders correctly:
1. Run `python convert_blog.py`
2. Open the HTML file in browser
3. Check if equations look right
4. Fix if needed, convert again

### 5. Organize Images

Keep images organized:
```
images/
├── 2026-02-08-post1/
│   ├── diagram.png
│   └── results.png
├── 2026-02-09-post2/
│   └── architecture.png
```

---

## 📚 Full Example Post

See `blog/2026-02-08-building-llms.md` for a complete example with:
- ✅ Complex math equations
- ✅ Python code with syntax highlighting
- ✅ Tables
- ✅ Proper structure
- ✅ References

It's ready to use as a reference!

---

## ✅ Checklist for Each Post

Before publishing:
- [ ] Frontmatter filled out (title, description, date, tags)
- [ ] Content written and proofread
- [ ] Code blocks have language specified (```python not just ```)
- [ ] Math equations use $$ for display, $ for inline
- [ ] Images referenced correctly (../images/...)
- [ ] Ran `python convert_blog.py` successfully
- [ ] Checked HTML in browser
- [ ] Committed and pushed

---

## 🚀 You're Ready!

Now you can write **complex, beautiful technical blog posts** like Karpathy and Raschka:
- ✅ Pure Markdown (no HTML tags!)
- ✅ Beautiful math rendering
- ✅ Syntax-highlighted code
- ✅ Images, tables, everything
- ✅ One command converts everything
- ✅ Professional output

**Just write naturally and let the converter handle the rest!**

---

## ❓ Troubleshooting

**Q: "ModuleNotFoundError: No module named 'markdown'"**
```bash
pip install markdown pygments
```

**Q: Math not rendering?**
Make sure you're using $$ for display math, $ for inline.

**Q: Code not highlighted?**
Add language name after triple backticks: ```python not just ```

**Q: Images not showing?**
Check path is correct: `../images/file.png` (note the ../)

**Q: Want to delete a post?**
1. Delete the .md file
2. Run `python convert_blog.py` again
3. It will update blog.html automatically

---

**Happy writing! Build AGI through daily documentation! 🤖**
