// Minimal JS for mobile nav toggle and theme toggle.
// Lightweight, no dependencies.

(function () {
  const navToggle = document.getElementById('nav-toggle');
  const siteNav = document.getElementById('site-nav');
  const themeToggle = document.getElementById('theme-toggle');
  const THEME_KEY = 'mh:theme';

  // Mobile nav toggle
  if (navToggle && siteNav) {
    navToggle.addEventListener('click', () => {
      const expanded = navToggle.getAttribute('aria-expanded') === 'true';
      navToggle.setAttribute('aria-expanded', String(!expanded));
      if (!expanded) {
        siteNav.style.display = 'block';
      } else {
        siteNav.style.display = '';
      }
    });
  }

  // Theme management
  function setTheme(mode) {
    document.body.classList.remove('theme-light', 'theme-dark');
    document.body.classList.add(mode === 'dark' ? 'theme-dark' : 'theme-light');
    if (themeToggle) themeToggle.setAttribute('aria-pressed', String(mode === 'dark'));
    localStorage.setItem(THEME_KEY, mode);
  }

  const saved = localStorage.getItem(THEME_KEY);
  if (saved) setTheme(saved);
  else {
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const isDark = document.body.classList.contains('theme-dark');
      setTheme(isDark ? 'light' : 'dark');
    });
  }
})();
