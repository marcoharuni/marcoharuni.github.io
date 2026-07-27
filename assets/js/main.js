(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('[data-theme-toggle]');
  const menuButton = document.querySelector('[data-menu-toggle]');
  const mobileNav = document.querySelector('[data-mobile-nav]');
  const header = document.querySelector('.site-header');

  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)');
  let savedTheme = null;
  try { savedTheme = localStorage.getItem('theme'); } catch (_) {}
  const initialTheme = savedTheme || (systemPrefersDark.matches ? 'dark' : 'light');

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    try { localStorage.setItem('theme', theme); } catch (_) {}
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#111212' : '#fbfaf7');
    if (themeButton) {
      const next = theme === 'dark' ? 'light' : 'dark';
      themeButton.setAttribute('aria-label', `Use ${next} mode`);
      themeButton.setAttribute('title', `Use ${next} mode`);
      themeButton.dataset.currentTheme = theme;
    }
  };

  setTheme(initialTheme);

  themeButton?.addEventListener('click', () => {
    setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  menuButton?.addEventListener('click', () => {
    const expanded = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!expanded));
    mobileNav.hidden = expanded;
  });

  mobileNav?.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      menuButton?.setAttribute('aria-expanded', 'false');
      mobileNav.hidden = true;
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && mobileNav && !mobileNav.hidden) {
      mobileNav.hidden = true;
      menuButton?.setAttribute('aria-expanded', 'false');
      menuButton?.focus();
    }
  });

  const updateHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 4);
  updateHeader();
  window.addEventListener('scroll', updateHeader, { passive: true });
})();
