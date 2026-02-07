// ===== Marco Haruni Website - Main JavaScript =====

// X-Style Dim/Light Mode Toggle
const themeToggle = document.querySelector('.theme-toggle');
const body = document.body;
const html = document.documentElement;

// Check for saved theme or default to light
const currentTheme = localStorage.getItem('theme') || 'light';
if (currentTheme === 'dim') {
    body.classList.add('dim-mode');
}

themeToggle.addEventListener('click', () => {
    body.classList.toggle('dim-mode');
    const theme = body.classList.contains('dim-mode') ? 'dim' : 'light';
    localStorage.setItem('theme', theme);

    // Smooth transition animation
    body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
});

// Mobile Menu Toggle
const mobileToggle = document.querySelector('.mobile-toggle');
const navMenu = document.querySelector('.nav-menu');

if (mobileToggle) {
    mobileToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        mobileToggle.classList.toggle('active');
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.navbar')) {
            navMenu.classList.remove('active');
            mobileToggle.classList.remove('active');
        }
    });

    // Close menu when clicking a link
    navMenu.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            mobileToggle.classList.remove('active');
        });
    });
}

// Social Link Enhanced Animations
const socialLinks = document.querySelectorAll('.social-link');

socialLinks.forEach(link => {
    link.addEventListener('mouseenter', function() {
        const social = this.dataset.social;

        // Add specific animations based on platform
        if (social === 'youtube') {
            this.style.animation = 'pulse 1.5s ease-in-out infinite';
        }
    });

    link.addEventListener('mouseleave', function() {
        this.style.animation = '';
    });

    // Click effect
    link.addEventListener('click', function(e) {
        // Create ripple effect
        const ripple = document.createElement('span');
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.6);
            width: 100px;
            height: 100px;
            margin-top: -50px;
            margin-left: -50px;
            animation: ripple 0.6s;
            pointer-events: none;
        `;

        const rect = this.getBoundingClientRect();
        ripple.style.left = (e.clientX - rect.left) + 'px';
        ripple.style.top = (e.clientY - rect.top) + 'px';

        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        from {
            opacity: 1;
            transform: scale(0);
        }
        to {
            opacity: 0;
            transform: scale(2);
        }
    }
`;
document.head.appendChild(style);

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe content cards
document.querySelectorAll('.content-card, .book-card').forEach(el => {
    observer.observe(el);
});

// Highlight active navigation item
const currentPath = window.location.pathname.split('/').pop() || 'index.html';
document.querySelectorAll('.nav-menu a').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || (currentPath === '' && href === 'index.html')) {
        link.classList.add('active');
    }
});

// Code syntax highlighting setup (for blog posts)
if (document.querySelector('pre code')) {
    // Prism.js will auto-highlight if loaded
    console.log('Code blocks detected - ready for syntax highlighting');
}

// Math rendering setup (for blog posts)
if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
        ]
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Alt + T = Toggle theme
    if (e.altKey && e.key === 't') {
        e.preventDefault();
        themeToggle.click();
    }

    // Alt + H = Go home
    if (e.altKey && e.key === 'h') {
        e.preventDefault();
        window.location.href = 'index.html';
    }
});

// Performance: Lazy load images
if ('loading' in HTMLImageElement.prototype) {
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
        img.src = img.dataset.src;
    });
} else {
    // Fallback for browsers that don't support lazy loading
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/lazysizes/5.3.2/lazysizes.min.js';
    document.body.appendChild(script);
}

// Console message for visitors
console.log('%c👋 Hi there!', 'font-size: 24px; font-weight: bold;');
console.log('%cInterested in AI? Let\'s connect!', 'font-size: 14px;');
console.log('%cEmail: marcoharuni95@gmail.com', 'font-size: 12px; color: #1d9bf0;');

// Analytics ready (add your tracking code here)
// Example: Google Analytics, Plausible, etc.
