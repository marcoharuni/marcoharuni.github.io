# Marco Haruni - Personal Website

Beautiful, modern personal website built with pure HTML, CSS, and JavaScript.

🌐 **Live Site**: https://marcoharuni.github.io (after deployment)

## ✨ Features

- ✅ **Pure HTML/CSS/JS** - No build process needed!
- 🎨 **Beautiful Design** - Modern, elegant, professional
- 🌓 **Dark/Light Mode** - Smooth theme switching
- 📱 **Fully Responsive** - Works on all devices
- ⚡ **Fast Loading** - Optimized performance
- 🎯 **SEO Friendly** - Proper meta tags

## 📂 Structure

```
marcoharuni.github.io/
├── index.html          # Home page
├── about.html          # About page with photo section
├── projects.html       # Projects showcase
├── blog.html           # Blog index
├── publications.html   # Research publications
├── books.html          # Reading list
├── contact.html        # Contact information
├── css/
│   └── style.css       # All styles
├── js/
│   └── main.js         # Theme toggle, mobile menu
├── images/             # Put your photos here!
└── blog/
    └── welcome.html    # Sample blog post
```

## 🚀 Quick Start - View Locally

### Option 1: Double-click (Easiest!)
1. Go to `/home/miss_merry/Documents/marcoharuni.github.io/`
2. Double-click `index.html`
3. It opens in your browser!

### Option 2: Use Python Server
```bash
cd ~/Documents/marcoharuni.github.io
python3 -m http.server 8000
# Visit: http://localhost:8000
```

## 📸 Add Your Photo

**Step 1:** Put your photo in the `images/` folder
```bash
cp ~/path/to/your-photo.jpg ~/Documents/marcoharuni.github.io/images/profile.jpg
```

**Step 2:** Edit `about.html` (line ~88):

Find:
```html
<div class="profile-placeholder">MH</div>
```

Replace with:
```html
<img src="images/profile.jpg" alt="Marco Haruni" class="profile-image">
```

Done! 🎉

## 🌍 Deploy to GitHub Pages

### Method 1: Upload via GitHub Website (Easiest!)

1. **Create Repository**
   - Go to https://github.com
   - Click "+" → "New repository"
   - Name: `marcoharuni.github.io` (exactly this!)
   - Make it Public
   - Click "Create repository"

2. **Upload Files**
   - Click "uploading an existing file"
   - Drag ALL files from your website folder
   - Commit changes

3. **Enable Pages**
   - Go to Settings → Pages
   - Source: "main" branch
   - Save

4. **Visit Your Site** (wait 1-2 minutes)
   - **https://marcoharuni.github.io** 🎉

### Method 2: Use Git

```bash
cd ~/Documents/marcoharuni.github.io

# Initialize git
git init
git add .
git commit -m "Initial website"

# Connect to GitHub
git branch -M main
git remote add origin https://github.com/marcoharuni/marcoharuni.github.io.git
git push -u origin main
```

## ✏️ Edit Your Website

### Change Colors
Edit `css/style.css`, line 3:
```css
--primary: #ff6b4a;  /* Change to your color! */
```

### Update Content
Just edit the HTML files directly:
- **About bio**: `about.html`
- **Contact info**: `contact.html`
- **Home page**: `index.html`

### Add a Blog Post

1. Create new file: `blog/my-post.html`
2. Copy structure from `blog/welcome.html`
3. Update content
4. Add link in `blog.html`

## 📝 Adding Projects

Edit `projects.html`, duplicate the example project card:

```html
<div class="feature-card">
    <h3>Your Project Name</h3>
    <p>Description...</p>
    <div>
        <span class="tag">Python</span>
        <span class="tag">AI</span>
    </div>
    <a href="https://github.com/..." target="_blank">View Project →</a>
</div>
```

## 📚 Adding Publications

Edit `publications.html`:

```html
<article class="feature-card">
    <h3>Paper Title</h3>
    <p>Author Names</p>
    <p><em>Conference/Journal</em> • 2026</p>
    <a href="https://..." target="_blank">Paper →</a>
</article>
```

## 🔄 Update Live Site

After making changes:

**Via GitHub Website:**
1. Go to your repo
2. Click on the file
3. Click edit (pencil icon)
4. Make changes
5. Commit
6. Wait 1 minute → changes are live!

**Via Git:**
```bash
git add .
git commit -m "Updated content"
git push
```

## 💡 Tips

- Keep images under 1MB
- Test locally before pushing
- Mobile menu works automatically
- Theme persists across visits
- Works offline!

## 🎨 Customization Ideas

- Change fonts in CSS (Google Fonts)
- Adjust spacing and sizes
- Add more sections
- Add image galleries
- Add animations

## 📧 Your Info

Already configured:
- Email: marcoharuni95@gmail.com
- GitHub: @marcoharuni
- X/Twitter: @marcoharuni
- YouTube: @marcoharuni
- HuggingFace: @marcoharuni95
- Location: San Francisco, CA

## 🐛 Troubleshooting

**Theme not working?**
- Make sure JavaScript is enabled
- Clear browser cache

**Mobile menu not opening?**
- Check `js/main.js` is loaded
- Try hard refresh (Ctrl+Shift+R)

**Images not showing?**
- Check file path is correct
- Ensure image is in `images/` folder
- Check file extension (jpg/png)

## 📖 Need Help?

Read **HOW-TO-ADD-PHOTO.md** for detailed photo and deployment instructions!

---

**Built with ❤️ using pure HTML, CSS, and JavaScript**

No frameworks, no build process, just simple and beautiful! 🚀
