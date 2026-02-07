# How to Add Your Photo

## Step 1: Prepare Your Photo

1. Choose a nice professional photo of yourself
2. Rename it to `profile.jpg` or `profile.png`
3. Recommended size: 500x500 pixels (square)
4. Make sure it's not too large (under 500KB is good)

## Step 2: Add Photo to Website

### Option A: Using File Manager (Easiest)

1. Open your file manager
2. Navigate to `/home/miss_merry/Documents/marcoharuni.github.io/images/`
3. Copy your photo (`profile.jpg`) into this folder

### Option B: Using Terminal

```bash
# Copy your photo from wherever it is to the images folder
cp /path/to/your/photo.jpg ~/Documents/marcoharuni.github.io/images/profile.jpg
```

For example, if your photo is in Downloads:
```bash
cp ~/Downloads/my-photo.jpg ~/Documents/marcoharuni.github.io/images/profile.jpg
```

## Step 3: Update the HTML

Open `about.html` and find this line (around line 88):

```html
<div class="profile-placeholder">MH</div>
```

Replace it with:

```html
<img src="images/profile.jpg" alt="Marco Haruni" class="profile-image">
```

## Step 4: Save and Test

1. Save the file
2. Open `about.html` in your web browser to see your photo
3. If it looks good, you're done!

## Adding More Images

You can add images anywhere on your site:

```html
<img src="images/project-screenshot.png" alt="Description">
```

Just put all images in the `images/` folder and reference them like above.

## Image Tips

- **Profile photo**: 500x500px, square, professional
- **Project screenshots**: 1200x800px or similar
- **Blog images**: 1200x630px (good for social media sharing)
- **File formats**: JPG for photos, PNG for screenshots/graphics
- **File size**: Keep under 1MB for fast loading

---

# Quick Deployment to GitHub

## Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click "+" → "New repository"
3. Name it: `marcoharuni.github.io` (exactly this!)
4. Make it Public
5. DON'T add README or .gitignore
6. Click "Create repository"

## Step 2: Upload Your Website

### Option A: Using GitHub Website (Easiest!)

1. On your new repository page, click "uploading an existing file"
2. Drag and drop ALL files from `/home/miss_merry/Documents/marcoharuni.github.io/`
3. Write commit message: "Initial website"
4. Click "Commit changes"

### Option B: Using Git (Terminal)

```bash
cd ~/Documents/marcoharuni.github.io

# Initialize git
git init
git add .
git commit -m "Initial website"

# Add GitHub as remote
git branch -M main
git remote add origin https://github.com/marcoharuni/marcoharuni.github.io.git
git push -u origin main
```

## Step 3: Enable GitHub Pages

1. Go to your repository
2. Click "Settings"
3. Click "Pages" (left sidebar)
4. Under "Source", select "main" branch
5. Click "Save"
6. Wait 1-2 minutes

## Step 4: Visit Your Website!

Go to: **https://marcoharuni.github.io**

Your website is now live! 🎉

---

# Updating Your Website

After making changes:

### Using GitHub Website:
1. Go to your repository
2. Navigate to the file you want to edit
3. Click the pencil icon to edit
4. Make your changes
5. Click "Commit changes"
6. Wait 1 minute, refresh your site!

### Using Git:
```bash
cd ~/Documents/marcoharuni.github.io
# Make your changes to files
git add .
git commit -m "Updated about page"
git push
```

Wait 1 minute and your changes are live!

---

# Adding a Blog Post

## Create a new HTML file in the `blog/` folder:

`blog/my-first-post.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My First Post | Marco Haruni</title>
    <link rel="stylesheet" href="../css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <nav class="navbar">
        <!-- Copy navigation from index.html -->
    </nav>

    <article style="max-width: 800px; margin: 6rem auto; padding: 0 2rem;">
        <a href="../blog.html" style="display: inline-block; margin-bottom: 2rem;">&larr; Back to Blog</a>

        <h1 style="font-size: 3rem; font-family: var(--font-serif); font-style: italic; margin-bottom: 1rem;">My First Post</h1>

        <p style="color: var(--text-secondary); margin-bottom: 2rem;">February 7, 2026</p>

        <div style="line-height: 1.8; font-size: 1.1rem;">
            <p>Your blog post content here...</p>

            <p>Add more paragraphs, images, code, etc.</p>
        </div>
    </article>

    <footer class="footer">
        <!-- Copy footer from index.html -->
    </footer>

    <script src="../js/main.js"></script>
</body>
</html>
```

Then update `blog.html` to link to your new post!

---

**Need help?** Just ask! 😊
