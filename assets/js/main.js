(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('[data-theme-toggle]');
  const menuButton = document.querySelector('[data-menu-toggle]');
  const mobileNav = document.querySelector('[data-mobile-nav]');
  const mobileNavWrap = document.querySelector('.mobile-nav-wrap');
  const header = document.querySelector('.site-header');
  const main = document.querySelector('main');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)');
  let menuTimer = null;

  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem('theme');
  } catch (_) {}

  const initialTheme = savedTheme || (systemPrefersDark.matches ? 'dark' : 'light');

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    root.style.colorScheme = theme;

    try {
      localStorage.setItem('theme', theme);
    } catch (_) {}

    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', theme === 'dark' ? '#111212' : '#fbfaf7');

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

  const openMenu = () => {
    if (!mobileNav || !mobileNavWrap || !menuButton) return;
    clearTimeout(menuTimer);
    mobileNav.hidden = false;
    mobileNav.setAttribute('aria-hidden', 'false');
    menuButton.setAttribute('aria-expanded', 'true');
    requestAnimationFrame(() => mobileNavWrap.classList.add('is-open'));
  };

  const closeMenu = ({ immediate = false, restoreFocus = false } = {}) => {
    if (!mobileNav || !mobileNavWrap || !menuButton) return;
    clearTimeout(menuTimer);
    mobileNavWrap.classList.remove('is-open');
    menuButton.setAttribute('aria-expanded', 'false');
    mobileNav.setAttribute('aria-hidden', 'true');

    const finish = () => {
      mobileNav.hidden = true;
      if (restoreFocus) menuButton.focus();
    };

    if (immediate || reducedMotion.matches) {
      finish();
    } else {
      menuTimer = window.setTimeout(finish, 230);
    }
  };

  menuButton?.addEventListener('click', () => {
    const expanded = menuButton.getAttribute('aria-expanded') === 'true';
    expanded ? closeMenu() : openMenu();
  });

  mobileNav?.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => closeMenu({ immediate: true }));
  });

  document.addEventListener('click', (event) => {
    if (
      mobileNavWrap?.classList.contains('is-open') &&
      !mobileNavWrap.contains(event.target) &&
      !menuButton?.contains(event.target)
    ) {
      closeMenu();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && mobileNavWrap?.classList.contains('is-open')) {
      closeMenu({ restoreFocus: true });
    }
  });

  window.addEventListener('resize', () => {
    if (window.innerWidth > 760) closeMenu({ immediate: true });
  });

  const updateHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 4);
  updateHeader();
  window.addEventListener('scroll', updateHeader, { passive: true });

  const shouldAnimateNavigation = (event, link) => {
    if (reducedMotion.matches || event.defaultPrevented || event.button !== 0) return false;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return false;
    if (link.hasAttribute('download') || link.target === '_blank') return false;

    const href = link.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) return false;

    const target = new URL(link.href, window.location.href);
    if (target.origin !== window.location.origin) return false;
    if (target.pathname.toLowerCase().endsWith('.pdf')) return false;
    if (target.pathname === window.location.pathname && target.search === window.location.search) return false;

    return true;
  };

  document.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (!link || !shouldAnimateNavigation(event, link)) return;

    event.preventDefault();
    root.classList.add('is-leaving');
    main?.setAttribute('aria-busy', 'true');

    window.setTimeout(() => {
      window.location.assign(link.href);
    }, 145);
  });

  window.addEventListener('pageshow', () => {
    root.classList.remove('is-leaving');
    main?.removeAttribute('aria-busy');
  });
})();
